import os
import logging
import uuid
from fastapi import FastAPI, HTTPException, Depends, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from backend.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

from backend.schemas import (
    CustomerInput, BatchCustomerInput, WebhookConfig,
    NotificationConfig, PredictionResponse, BatchPredictionResponse,
    AnomalyResponse, HistoryResponse,
)
from backend.model_utils import predict_future_spending, retrain_model, get_model_config, get_model_version
from backend.database import (
    save_prediction, save_batch_prediction, get_prediction_history,
    get_prediction_stats, get_cohort_data, get_rfm_segmentation,
    cleanup_old_predictions,
)
from backend.auth import verify_api_key, check_rate_limit, get_rate_limit_info
from backend.anomaly_detection import detect_individual_anomaly, get_anomaly_recommendations
from backend.webhooks import (
    send_webhook_notification, notify_at_risk_customer,
    notify_high_value_customer,
)
from backend.notifications import (
    notify_marketing_team_at_risk, notify_marketing_team_high_value,
    notify_marketing_team_batch_summary,
)
from backend.explainability import explain_prediction
from backend.cache import prediction_cache
from backend.metrics import metrics
from backend.circuit_breaker import circuit_breakers
from typing import List, Optional
from datetime import datetime
import asyncio
import time

# ─── CORS from environment ───
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",")

app = FastAPI(
    title="Customer Spending Prediction API",
    description="Advanced customer spending prediction with analytics, anomaly detection, and notifications",
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

logger.info(f"CORS configured for origins: {CORS_ORIGINS}")


# ─── Middleware ───

@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    """Request tracing, metrics, and rate limiting middleware."""
    start = time.time()
    request_id = str(uuid.uuid4())[:8]

    # Skip for docs and health
    skip_paths = {"/docs", "/openapi.json", "/redoc"}
    is_health = request.url.path == "/health" or request.url.path == "/metrics"

    if not is_health:
        # Rate limiting
        api_key = request.headers.get("X-API-Key")
        if api_key and not check_rate_limit(api_key):
            rate_info = get_rate_limit_info(api_key)
            metrics.inc_counter("rate_limit_exceeded")
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded",
                    "retry_after": rate_info["window_seconds"],
                    "limit": rate_info["limit"],
                    "remaining": rate_info["remaining"],
                },
                headers={"Retry-After": str(rate_info["window_seconds"])},
            )

    response = await call_next(request)

    # Add tracing headers
    duration_ms = (time.time() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{duration_ms:.0f}ms"

    if not is_health:
        metrics.record_api_request(
            request.method, request.url.path, response.status_code, duration_ms
        )

    # Rate limit headers
    api_key = request.headers.get("X-API-Key")
    if api_key and not is_health:
        rate_info = get_rate_limit_info(api_key)
        response.headers["X-RateLimit-Limit"] = str(rate_info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(rate_info["remaining"])

    return response


# ─── Health check ───

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_config = get_model_config()
    return {
        "status": "healthy",
        "version": "2.1.0",
        "model_version": get_model_version(),
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "api": "running",
            "database": "connected",
            "model": "loaded" if model_config else "fallback",
            "cache": prediction_cache.stats(),
            "circuit_breakers": circuit_breakers.all(),
        },
    }


# ─── V1 API routes ───

@app.post("/v1/predict", response_model=PredictionResponse)
async def predict_v1(
    data: CustomerInput,
    api_key: str = Depends(verify_api_key),
):
    """Predict future spending for a single customer."""
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] Prediction request: {data.customer_type}, spent={data.total_spent}")

    start = time.time()
    input_data = data.model_dump()
    input_data["last_purchase_date"] = str(input_data["last_purchase_date"])

    # Check cache
    cached = prediction_cache.get(input_data)
    if cached:
        logger.info(f"[{request_id}] Cache hit — returning cached prediction")
        metrics.inc_counter("cache_hits")
        return cached
    metrics.inc_counter("cache_misses")

    try:
        result = predict_future_spending(data)
        duration_ms = (time.time() - start) * 1000

        prediction_id = save_prediction(input_data, result, source="api", request_id=request_id)
        anomaly_result = detect_individual_anomaly(result)

        response = PredictionResponse(
            predicted_future_spending=result["prediction"],
            derived_metrics={
                "average_order_value": result["avg_order_value"],
                "recency_days": result["recency"],
                "CLV": result["CLV"],
                "churn_risk": result["churn_risk"],
                "persona": result["persona"],
                "anomaly_detection": anomaly_result,
            },
            recommendation=result["recommendation"],
            currency="INR",
            prediction_id=prediction_id,
            confidence_interval=result.get("confidence_interval"),
            model_version=result.get("model_version"),
        )

        # Cache the result
        prediction_cache.set(input_data, response)
        metrics.record_prediction(result["persona"], duration_ms)

        logger.info(
            f"[{request_id}] Prediction complete ({duration_ms:.0f}ms): ₹{result['prediction']:,.2f}, "
            f"persona={result['persona']}, churn={result['churn_risk']:.2f}"
        )
        return response

    except Exception as e:
        duration_ms = (time.time() - start) * 1000
        metrics.inc_counter("prediction_errors")
        logger.error(f"[{request_id}] Prediction failed ({duration_ms:.0f}ms): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Backward-compatible alias
@app.post("/predict", response_model=PredictionResponse, include_in_schema=False)
async def predict_compat(
    data: CustomerInput,
    api_key: str = Depends(verify_api_key),
):
    """Deprecated: use /v1/predict instead."""
    return await predict_v1(data, api_key)


@app.post("/v1/predict/explain")
async def predict_explain_v1(
    data: CustomerInput,
    api_key: str = Depends(verify_api_key),
):
    """
    Predict future spending with full explainability.

    Returns the prediction along with:
    - Feature contributions (SHAP-based or importance-weighted)
    - Plain-language explanation of why the prediction was made
    - Top positive and negative drivers
    - Actionable business insights
    """
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] Explain request: {data.customer_type}, spent={data.total_spent}")

    try:
        result = predict_future_spending(data)
        explanation = explain_prediction(data, result)

        input_data = data.model_dump()
        input_data["last_purchase_date"] = str(input_data["last_purchase_date"])
        prediction_id = save_prediction(input_data, result, source="api", request_id=request_id)

        return {
            "predicted_future_spending": result["prediction"],
            "confidence_interval": result.get("confidence_interval"),
            "model_version": result.get("model_version"),
            "persona": result["persona"],
            "churn_risk": result["churn_risk"],
            "recommendation": result["recommendation"],
            "explanation": explanation,
            "prediction_id": prediction_id,
            "currency": "INR",
        }
    except Exception as e:
        logger.error(f"[{request_id}] Explain failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@app.post("/predict/explain", include_in_schema=False)
async def predict_explain_compat(
    data: CustomerInput,
    api_key: str = Depends(verify_api_key),
):
    """Deprecated: use /v1/predict/explain instead."""
    return await predict_explain_v1(data, api_key)


@app.post("/v1/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_v1(
    data: BatchCustomerInput,
    api_key: str = Depends(verify_api_key),
):
    """Predict future spending for multiple customers."""
    logger.info(f"Batch prediction request: {len(data.customers)} customers")

    predictions = []
    input_data_list = []

    for customer in data.customers:
        result = predict_future_spending(customer)
        input_dict = customer.model_dump()
        input_dict["last_purchase_date"] = str(input_dict["last_purchase_date"])
        input_data_list.append(input_dict)

        prediction_id = save_prediction(input_dict, result, source="batch")

        predictions.append(PredictionResponse(
            predicted_future_spending=result["prediction"],
            derived_metrics={
                "average_order_value": result["avg_order_value"],
                "recency_days": result["recency"],
                "CLV": result["CLV"],
                "churn_risk": result["churn_risk"],
                "persona": result["persona"],
            },
            recommendation=result["recommendation"],
            currency="INR",
            prediction_id=prediction_id,
            confidence_interval=result.get("confidence_interval"),
            model_version=result.get("model_version"),
        ))

    batch_id = save_batch_prediction(len(data.customers), source=data.source or "api")

    spending_values = [p.predicted_future_spending for p in predictions]
    churn_values = [p.derived_metrics["churn_risk"] for p in predictions]
    clv_values = [p.derived_metrics["CLV"] for p in predictions]

    at_risk_count = sum(1 for c in churn_values if c > 0.6)
    high_value_count = sum(1 for c in clv_values if c > 100)

    summary = {
        "total_customers": len(predictions),
        "avg_predicted_spending": round(sum(spending_values) / len(spending_values), 2) if spending_values else 0,
        "total_predicted_spending": round(sum(spending_values), 2),
        "at_risk_count": at_risk_count,
        "high_value_count": high_value_count,
        "avg_churn_risk": round(sum(churn_values) / len(churn_values), 2) if churn_values else 0,
        "avg_clv": round(sum(clv_values) / len(clv_values), 2) if clv_values else 0,
    }

    logger.info(
        f"Batch complete: {len(predictions)} customers, "
        f"at_risk={at_risk_count}, high_value={high_value_count}"
    )

    return BatchPredictionResponse(
        batch_id=batch_id,
        total_customers=len(predictions),
        predictions=predictions,
        summary=summary,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse, include_in_schema=False)
async def predict_batch_compat(
    data: BatchCustomerInput,
    api_key: str = Depends(verify_api_key),
):
    """Deprecated: use /v1/predict/batch instead."""
    return await predict_batch_v1(data, api_key)


@app.get("/v1/history", response_model=HistoryResponse)
async def get_history_v1(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    api_key: str = Depends(verify_api_key),
):
    """Get prediction history with accurate pagination."""
    offset = (page - 1) * page_size
    predictions, total_count = get_prediction_history(limit=page_size, offset=offset)

    return HistoryResponse(
        predictions=predictions,
        total_count=total_count,
        page=page,
        page_size=page_size,
    )


@app.get("/history", response_model=HistoryResponse, include_in_schema=False)
async def get_history_compat(
    page: int = 1, page_size: int = 20, api_key: str = Depends(verify_api_key)
):
    """Deprecated: use /v1/history instead."""
    return await get_history_v1(page, page_size, api_key)


@app.get("/v1/analytics/stats")
async def get_stats_v1(api_key: str = Depends(verify_api_key)):
    """Get aggregate prediction statistics."""
    return get_prediction_stats()


@app.get("/analytics/stats", include_in_schema=False)
async def get_stats_compat(api_key: str = Depends(verify_api_key)):
    """Deprecated: use /v1/analytics/stats instead."""
    return await get_stats_v1(api_key)


@app.get("/v1/analytics/cohorts")
async def get_cohorts_v1(api_key: str = Depends(verify_api_key)):
    """Get cohort analysis data."""
    return get_cohort_data()


@app.get("/analytics/cohorts", include_in_schema=False)
async def get_cohorts_compat(api_key: str = Depends(verify_api_key)):
    """Deprecated."""
    return await get_cohorts_v1(api_key)


@app.get("/v1/analytics/rfm")
async def get_rfm_v1(api_key: str = Depends(verify_api_key)):
    """Get RFM segmentation."""
    return get_rfm_segmentation()


@app.get("/analytics/rfm", include_in_schema=False)
async def get_rfm_compat(api_key: str = Depends(verify_api_key)):
    """Deprecated."""
    return await get_rfm_v1(api_key)


# ─── Model management ───

@app.get("/v1/model/config")
async def get_model_config_v1(api_key: str = Depends(verify_api_key)):
    """Get current model configuration and version."""
    return get_model_config()


@app.post("/v1/model/retrain")
async def retrain_model_v1(
    training_data_path: str = Query("data/training_data.csv"),
    api_key: str = Depends(verify_api_key),
):
    """
    Retrain the model with new data.

    Requires training_data.csv in the data/ directory with columns:
    total_spent, total_orders, avg_order_value, recency, CLV, churn_risk,
    norm_spent, norm_orders, norm_recency, target_spending
    """
    logger.info("Model retraining requested")
    try:
        result = retrain_model(training_data_path)
        logger.info(f"Model retrained: version={result['version']}")
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Retraining failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")


# ─── Webhook testing ───

@app.post("/v1/webhook/test")
async def test_webhook_v1(
    config: WebhookConfig,
    api_key: str = Depends(verify_api_key),
):
    """Test a webhook configuration."""
    test_payload = {
        "event": "webhook.test",
        "timestamp": datetime.utcnow().isoformat(),
        "data": {
            "message": "This is a test webhook notification",
            "api_version": "2.1.0",
        },
    }

    cb = circuit_breakers.get("webhooks")
    if not cb.allow_request():
        return {"success": False, "error": "Circuit breaker is OPEN — webhook service failing"}

    result = await send_webhook_notification(
        config.url, "webhook.test", test_payload, config.headers
    )

    if result["success"]:
        cb.record_success()
    else:
        cb.record_failure()

    metrics.record_webhook("test", result["success"])
    return {
        "success": result["success"],
        "status_code": result.get("status_code"),
        "attempt": result.get("attempt", 1),
        "circuit_breaker": cb.info(),
        "message": "Webhook test completed",
    }


@app.post("/webhook/test", include_in_schema=False)
async def test_webhook_compat(
    config: WebhookConfig, api_key: str = Depends(verify_api_key)
):
    """Deprecated."""
    return await test_webhook_v1(config, api_key)


# ─── API key management ───

@app.get("/v1/api-keys/generate")
async def generate_api_key_v1():
    """Generate a new API key."""
    from backend.auth import generate_api_key
    new_key = generate_api_key()
    return {
        "api_key": new_key,
        "message": "Store this key securely. It won't be shown again.",
    }


@app.get("/api-keys/generate", include_in_schema=False)
async def generate_api_key_compat():
    """Deprecated."""
    return await generate_api_key_v1()


# ─── Data maintenance ───

@app.post("/v1/maintenance/cleanup")
async def cleanup_v1(
    days: int = Query(None, description="Retention period in days (default from env)"),
    api_key: str = Depends(verify_api_key),
):
    """Remove predictions older than the retention period."""
    deleted = cleanup_old_predictions(days)
    return {
        "deleted": deleted,
        "retention_days": days or int(os.getenv("DATA_RETENTION_DAYS", "365")),
    }


@app.post("/maintenance/cleanup", include_in_schema=False)
async def cleanup_compat(
    days: int = None, api_key: str = Depends(verify_api_key)
):
    """Deprecated."""
    return await cleanup_v1(days, api_key)


# ─── Monitoring & Observability ───

@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus-compatible metrics endpoint."""
    from fastapi.responses import PlainTextResponse
    metrics.set_gauge("cache_size", prediction_cache.stats()["size"])
    metrics.set_gauge("cache_hit_rate", prediction_cache.stats()["hit_rate"])
    return PlainTextResponse(content=metrics.to_prometheus(), media_type="text/plain")


@app.get("/v1/metrics")
async def metrics_json_v1(api_key: str = Depends(verify_api_key)):
    """JSON metrics endpoint for dashboard consumption."""
    return {
        "api_metrics": metrics.get_all(),
        "cache_stats": prediction_cache.stats(),
        "circuit_breakers": circuit_breakers.all(),
        "model_version": get_model_version(),
    }


@app.get("/v1/cache/stats")
async def cache_stats_v1(api_key: str = Depends(verify_api_key)):
    """Get cache statistics."""
    return prediction_cache.stats()


@app.post("/v1/cache/clear")
async def cache_clear_v1(api_key: str = Depends(verify_api_key)):
    """Clear the prediction cache."""
    prediction_cache.clear()
    return {"message": "Cache cleared"}


@app.get("/v1/circuit-breakers")
async def circuit_breakers_v1(api_key: str = Depends(verify_api_key)):
    """Get circuit breaker status."""
    return circuit_breakers.all()


@app.post("/v1/circuit-breakers/{name}/reset")
async def circuit_breaker_reset_v1(name: str, api_key: str = Depends(verify_api_key)):
    """Manually reset a circuit breaker."""
    cb = circuit_breakers.get(name)
    cb.reset()
    return cb.info()


@app.get("/v1/drift/detect")
async def detect_drift_v1(
    window: int = Query(50, description="Number of recent predictions to analyze"),
    api_key: str = Depends(verify_api_key),
):
    """
    Detect data drift by comparing recent predictions to historical patterns.

    Checks for distribution shifts in key metrics:
    - Spending amounts
    - Churn risk scores
    - Persona distribution
    - Customer types
    """
    predictions, total = get_prediction_history(limit=window, offset=0)
    if len(predictions) < 10:
        return {
            "drift_detected": False,
            "reason": "Insufficient data (need at least 10 predictions)",
            "sample_size": len(predictions),
        }

    # Analyze distributions
    import numpy as np
    spending = [p.get("predicted_spending", 0) for p in predictions]
    churn = [p.get("churn_risk", 0) for p in predictions]
    personas = [p.get("persona", "unknown") for p in predictions]
    customer_types = [p.get("customer_type", "unknown") for p in predictions]

    # Check for anomalies in distributions
    spend_arr = np.array(spending)
    churn_arr = np.array(churn)

    drift_flags = []

    # Check spending concentration
    spend_std = float(np.std(spend_arr))
    spend_mean = float(np.mean(spend_arr))
    cv = spend_std / spend_mean if spend_mean > 0 else 0  # Coefficient of variation
    if cv > 0.8:
        drift_flags.append({
            "metric": "spending_concentration",
            "severity": "high",
            "detail": f"Spending CV of {cv:.2f} indicates high variance",
        })

    # Check churn risk drift
    churn_mean = float(np.mean(churn_arr))
    if churn_mean > 0.7:
        drift_flags.append({
            "metric": "churn_risk_elevation",
            "severity": "medium",
            "detail": f"Average churn risk of {churn_mean:.0%} across recent predictions",
        })

    # Check persona distribution
    from collections import Counter
    persona_dist = Counter(personas)
    dominant_persona = persona_dist.most_common(1)[0]
    dominant_pct = dominant_persona[1] / len(personas)
    if dominant_pct > 0.8:
        drift_flags.append({
            "metric": "persona_imbalance",
            "severity": "low",
            "detail": f"{dominant_persona[0]} accounts for {dominant_pct:.0%} of predictions",
        })

    # Check customer type distribution
    type_dist = Counter(customer_types)
    if len(type_dist) == 1:
        drift_flags.append({
            "metric": "single_customer_type",
            "severity": "medium",
            "detail": f"All predictions are for '{list(type_dist.keys())[0]}' customers",
        })

    return {
        "drift_detected": len(drift_flags) > 0,
        "sample_size": len(predictions),
        "metrics": {
            "spending_mean": round(spend_mean, 2),
            "spending_std": round(spend_std, 2),
            "spending_cv": round(cv, 2),
            "churn_mean": round(churn_mean, 2),
            "persona_distribution": dict(persona_dist),
            "customer_type_distribution": dict(type_dist),
        },
        "flags": drift_flags,
        "recommendation": (
            "Review prediction patterns and consider model retraining"
            if drift_flags else
            "No significant drift detected"
        ),
    }
