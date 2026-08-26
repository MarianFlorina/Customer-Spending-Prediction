import os
import hmac
import hashlib
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
import httpx

logger = logging.getLogger(__name__)

WEBHOOK_TIMEOUT = int(os.getenv("WEBHOOK_TIMEOUT", "10"))
WEBHOOK_MAX_RETRIES = int(os.getenv("WEBHOOK_MAX_RETRIES", "3"))
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "")
WEBHOOK_RETRY_BASE_DELAY = float(os.getenv("WEBHOOK_RETRY_BASE_DELAY", "1.0"))


def _sign_payload(payload: str, secret: str) -> str:
    """Generate HMAC-SHA256 signature for webhook payload."""
    return hmac.new(
        secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def _verify_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Verify HMAC-SHA256 webhook signature."""
    if not secret:
        return True  # No secret configured = skip verification
    expected = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


async def send_webhook_notification(
    webhook_url: str,
    event_type: str,
    payload: Dict,
    headers: Optional[Dict] = None,
    secret: str = None,
    max_retries: int = None,
) -> Dict:
    """
    Send a webhook notification with retry logic and signature verification.

    Retries with exponential backoff on transient failures (5xx, timeouts).
    """
    if max_retries is None:
        max_retries = WEBHOOK_MAX_RETRIES

    effective_secret = secret or WEBHOOK_SECRET

    body_json = json.dumps(payload, default=str)
    body_bytes = body_json.encode("utf-8")

    default_headers = {
        "Content-Type": "application/json",
        "X-Webhook-Event": event_type,
        "X-Webhook-Timestamp": datetime.utcnow().isoformat(),
        "User-Agent": "FreebuffSpendingPredictor/2.0",
    }

    if effective_secret:
        signature = _sign_payload(body_json, effective_secret)
        default_headers["X-Webhook-Signature"] = f"sha256={signature}"

    if headers:
        default_headers.update(headers)

    last_error = None
    for attempt in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=WEBHOOK_TIMEOUT) as client:
                response = await client.post(
                    webhook_url, content=body_bytes, headers=default_headers
                )

                result = {
                    "success": response.status_code < 400,
                    "status_code": response.status_code,
                    "response": response.text[:500] if response.text else None,
                    "webhook_url": webhook_url,
                    "event_type": event_type,
                    "attempt": attempt + 1,
                }

                if result["success"]:
                    logger.info(
                        f"Webhook sent successfully: {event_type} -> {webhook_url} "
                        f"(attempt {attempt + 1})"
                    )
                    return result

                # Don't retry client errors (4xx) — they're permanent
                if 400 <= response.status_code < 500:
                    logger.warning(
                        f"Webhook permanent failure ({response.status_code}): "
                        f"{event_type} -> {webhook_url}"
                    )
                    return result

                # Server error — will retry
                last_error = f"HTTP {response.status_code}"
                logger.warning(
                    f"Webhook transient failure ({response.status_code}), "
                    f"attempt {attempt + 1}/{max_retries + 1}: {webhook_url}"
                )

        except httpx.TimeoutException:
            last_error = "timeout"
            logger.warning(
                f"Webhook timeout, attempt {attempt + 1}/{max_retries + 1}: {webhook_url}"
            )
        except Exception as e:
            last_error = str(e)
            logger.error(
                f"Webhook error: {e}, attempt {attempt + 1}/{max_retries + 1}: {webhook_url}"
            )

        # Exponential backoff before retry
        if attempt < max_retries:
            import asyncio
            delay = WEBHOOK_RETRY_BASE_DELAY * (2 ** attempt)
            await asyncio.sleep(delay)

    logger.error(
        f"Webhook failed after {max_retries + 1} attempts: {event_type} -> {webhook_url} "
        f"(last error: {last_error})"
    )
    return {
        "success": False,
        "status_code": None,
        "error": f"Failed after {max_retries + 1} attempts: {last_error}",
        "webhook_url": webhook_url,
        "event_type": event_type,
        "attempt": max_retries + 1,
    }


def notify_at_risk_customer(webhook_url: str, prediction: Dict, customer_data: Dict) -> Dict:
    """Build notification payload for at-risk customer."""
    payload = {
        "alert_type": "at_risk_customer",
        "customer": {
            "type": customer_data.get("customer_type"),
            "category": customer_data.get("product_category"),
            "total_spent": customer_data.get("total_spent"),
            "total_orders": customer_data.get("total_orders"),
        },
        "risk_metrics": {
            "churn_risk": prediction.get("churn_risk"),
            "persona": prediction.get("persona"),
            "predicted_spending": prediction.get("prediction"),
        },
        "recommended_action": prediction.get("recommendation"),
        "priority": "high" if prediction.get("churn_risk", 0) > 0.8 else "medium",
    }
    return {"event_type": "customer.at_risk", "payload": payload, "webhook_url": webhook_url}


def notify_high_value_customer(webhook_url: str, prediction: Dict, customer_data: Dict) -> Dict:
    """Build notification payload for high-value customer."""
    payload = {
        "alert_type": "high_value_customer",
        "customer": {
            "type": customer_data.get("customer_type"),
            "category": customer_data.get("product_category"),
            "total_spent": customer_data.get("total_spent"),
            "total_orders": customer_data.get("total_orders"),
        },
        "value_metrics": {
            "clv": prediction.get("clv"),
            "avg_order_value": prediction.get("avg_order_value"),
            "predicted_spending": prediction.get("prediction"),
        },
        "recommended_action": prediction.get("recommendation"),
        "priority": "high",
    }
    return {"event_type": "customer.high_value", "payload": payload, "webhook_url": webhook_url}


def notify_anomaly_detected(webhook_url: str, anomaly_result: Dict, customer_data: Optional[Dict] = None) -> Dict:
    """Build notification payload for anomaly detection."""
    payload = {
        "alert_type": "anomaly_detected",
        "anomaly_details": {
            "total_anomalies": anomaly_result.get("total_anomalies", 0),
            "anomaly_rate": anomaly_result.get("anomaly_rate", 0),
            "anomalies": anomaly_result.get("anomalies", []),
        },
        "risk_level": anomaly_result.get("risk_level", "low"),
        "customer": customer_data,
        "timestamp": datetime.utcnow().isoformat(),
    }
    return {"event_type": "system.anomaly_detected", "payload": payload, "webhook_url": webhook_url}


async def send_batch_webhook_notifications(webhook_url: str, notifications: List[Dict]) -> List[Dict]:
    """Send multiple webhook notifications."""
    results = []
    for notification in notifications:
        result = await send_webhook_notification(
            webhook_url=webhook_url,
            event_type=notification.get("event_type", "unknown"),
            payload=notification.get("payload", {}),
        )
        results.append(result)
    return results


WEBHOOK_EVENTS = {
    "prediction.completed": "A new prediction has been completed",
    "customer.at_risk": "A customer has been identified as at-risk",
    "customer.high_value": "A high-value customer has been identified",
    "system.anomaly_detected": "An anomaly has been detected in predictions",
    "batch.completed": "A batch prediction has been completed",
    "system.health_check": "System health check status",
}
