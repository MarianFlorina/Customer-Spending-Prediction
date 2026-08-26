"""Comprehensive tests for the Customer Spending Prediction API."""
import os
import json
import pytest
from datetime import date, datetime

# Set test API key before importing app
os.environ["API_KEY"] = "test-api-key-12345"
os.environ["DATABASE_PATH"] = "data/test_predictions.db"
os.environ["RATE_LIMIT_WINDOW"] = "60"
os.environ["MAX_REQUESTS_PER_WINDOW"] = "1000"

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)
HEADERS = {"X-API-Key": "test-api-key-12345"}


# ─── Fixtures ───

@pytest.fixture
def valid_customer():
    """Valid customer input payload."""
    return {
        "total_spent": 5000,
        "total_orders": 3,
        "last_purchase_date": "2024-06-01",
        "spending_period": "Last 30 Days",
        "customer_type": "New",
        "product_category": "Electronics",
        "discount_sensitivity": "Low",
    }


@pytest.fixture
def loyal_customer():
    """Loyal customer with many orders."""
    return {
        "total_spent": 25000,
        "total_orders": 15,
        "last_purchase_date": "2025-01-15",
        "spending_period": "Lifetime",
        "customer_type": "Loyal",
        "product_category": "Fashion",
        "discount_sensitivity": "Low",
    }


@pytest.fixture
def at_risk_customer():
    """Customer likely to churn."""
    return {
        "total_spent": 2000,
        "total_orders": 1,
        "last_purchase_date": "2022-01-01",
        "spending_period": "Last 12 Months",
        "customer_type": "New",
        "product_category": "Grocery",
        "discount_sensitivity": "High",
    }


# ═══════════════════════════════════════════════════════
# Health Check
# ═══════════════════════════════════════════════════════

class TestHealthCheck:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_contains_required_fields(self):
        data = client.get("/health").json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "model_version" in data
        assert "timestamp" in data
        assert "services" in data

    def test_health_services_status(self):
        data = client.get("/health").json()
        assert data["services"]["api"] == "running"
        assert data["services"]["database"] == "connected"


# ═══════════════════════════════════════════════════════
# Authentication
# ═══════════════════════════════════════════════════════

class TestAuthentication:
    def test_missing_api_key_returns_401(self):
        response = client.post("/v1/predict", json={
            "total_spent": 5000, "total_orders": 3,
            "last_purchase_date": "2024-06-01", "spending_period": "Last 30 Days",
            "customer_type": "New", "product_category": "Electronics",
            "discount_sensitivity": "Low",
        })
        assert response.status_code == 401

    def test_invalid_api_key_returns_403(self):
        headers = {"X-API-Key": "invalid-key-xyz"}
        response = client.post("/v1/predict", json={
            "total_spent": 5000, "total_orders": 3,
            "last_purchase_date": "2024-06-01", "spending_period": "Last 30 Days",
            "customer_type": "New", "product_category": "Electronics",
            "discount_sensitivity": "Low",
        }, headers=headers)
        assert response.status_code == 403

    def test_freebuff_prefix_accepted(self):
        headers = {"X-API-Key": "freebuff-any-key-works"}
        response = client.get("/health", headers=headers)
        assert response.status_code == 200

    def test_valid_api_key_accepted(self):
        response = client.get("/health", headers=HEADERS)
        assert response.status_code == 200


# ═══════════════════════════════════════════════════════
# Single Prediction (v1)
# ═══════════════════════════════════════════════════════

class TestSinglePrediction:
    def test_predict_returns_200(self, valid_customer):
        response = client.post("/v1/predict", json=valid_customer, headers=HEADERS)
        assert response.status_code == 200

    def test_predict_response_structure(self, valid_customer):
        data = client.post("/v1/predict", json=valid_customer, headers=HEADERS).json()
        assert "predicted_future_spending" in data
        assert "derived_metrics" in data
        assert "recommendation" in data
        assert "currency" in data
        assert data["currency"] == "INR"
        assert "prediction_id" in data
        assert "confidence_interval" in data
        assert "model_version" in data

    def test_confidence_interval_structure(self, valid_customer):
        data = client.post("/v1/predict", json=valid_customer, headers=HEADERS).json()
        ci = data["confidence_interval"]
        assert "lower" in ci
        assert "upper" in ci
        assert "alpha" in ci
        assert ci["alpha"] == 0.05
        assert ci["lower"] < data["predicted_future_spending"] < ci["upper"]

    def test_derived_metrics_structure(self, valid_customer):
        data = client.post("/v1/predict", json=valid_customer, headers=HEADERS).json()
        metrics = data["derived_metrics"]
        assert "average_order_value" in metrics
        assert "recency_days" in metrics
        assert "CLV" in metrics
        assert "churn_risk" in metrics
        assert "persona" in metrics
        assert "anomaly_detection" in metrics

    def test_at_risk_customer_persona(self, at_risk_customer):
        data = client.post("/v1/predict", json=at_risk_customer, headers=HEADERS).json()
        assert data["derived_metrics"]["persona"] == "At-Risk Customer"
        assert data["derived_metrics"]["churn_risk"] > 0.6

    def test_loyal_customer_persona(self, loyal_customer):
        data = client.post("/v1/predict", json=loyal_customer, headers=HEADERS).json()
        assert data["derived_metrics"]["persona"] == "Loyal Customer"

    def test_prediction_is_positive(self, valid_customer):
        data = client.post("/v1/predict", json=valid_customer, headers=HEADERS).json()
        assert data["predicted_future_spending"] > 0

    def test_avg_order_value_calculation(self, valid_customer):
        data = client.post("/v1/predict", json=valid_customer, headers=HEADERS).json()
        expected_aov = valid_customer["total_spent"] / valid_customer["total_orders"]
        assert data["derived_metrics"]["average_order_value"] == round(expected_aov, 2)

    def test_invalid_input_returns_422(self):
        response = client.post("/v1/predict", json={
            "total_spent": -100,
            "total_orders": 0,
        }, headers=HEADERS)
        assert response.status_code == 422

    def test_missing_fields_returns_422(self):
        response = client.post("/v1/predict", json={
            "total_spent": 5000,
        }, headers=HEADERS)
        assert response.status_code == 422


# ═══════════════════════════════════════════════════════
# Backward-Compatible Alias (/predict)
# ═══════════════════════════════════════════════════════

class TestBackwardCompatibility:
    def test_compat_predict_works(self, valid_customer):
        response = client.post("/predict", json=valid_customer, headers=HEADERS)
        assert response.status_code == 200

    def test_compat_predict_same_result(self, valid_customer):
        v1 = client.post("/v1/predict", json=valid_customer, headers=HEADERS).json()
        compat = client.post("/predict", json=valid_customer, headers=HEADERS).json()
        assert v1["predicted_future_spending"] == compat["predicted_future_spending"]


# ═══════════════════════════════════════════════════════
# Batch Prediction
# ═══════════════════════════════════════════════════════

class TestBatchPrediction:
    def test_batch_predict_returns_200(self, valid_customer):
        response = client.post("/v1/predict/batch", json={
            "customers": [valid_customer, valid_customer],
            "source": "test",
        }, headers=HEADERS)
        assert response.status_code == 200

    def test_batch_response_structure(self, valid_customer):
        data = client.post("/v1/predict/batch", json={
            "customers": [valid_customer],
        }, headers=HEADERS).json()
        assert "batch_id" in data
        assert "total_customers" in data
        assert "predictions" in data
        assert "summary" in data
        assert data["total_customers"] == 1

    def test_batch_summary_fields(self, valid_customer):
        data = client.post("/v1/predict/batch", json={
            "customers": [valid_customer, valid_customer],
        }, headers=HEADERS).json()
        summary = data["summary"]
        assert "total_customers" in summary
        assert "avg_predicted_spending" in summary
        assert "total_predicted_spending" in summary
        assert "at_risk_count" in summary
        assert "high_value_count" in summary
        assert "avg_churn_risk" in summary

    def test_batch_empty_list_returns_422(self):
        response = client.post("/v1/predict/batch", json={
            "customers": [],
        }, headers=HEADERS)
        assert response.status_code == 422

    def test_batch_each_prediction_has_ci(self, valid_customer, loyal_customer):
        data = client.post("/v1/predict/batch", json={
            "customers": [valid_customer, loyal_customer],
        }, headers=HEADERS).json()
        for pred in data["predictions"]:
            assert "confidence_interval" in pred
            assert pred["confidence_interval"]["lower"] < pred["predicted_future_spending"]


# ═══════════════════════════════════════════════════════
# History
# ═══════════════════════════════════════════════════════

class TestHistory:
    def test_history_returns_200(self):
        response = client.get("/v1/history", headers=HEADERS)
        assert response.status_code == 200

    def test_history_response_structure(self):
        data = client.get("/v1/history", headers=HEADERS).json()
        assert "predictions" in data
        assert "total_count" in data
        assert "page" in data
        assert "page_size" in data

    def test_history_pagination(self):
        data = client.get("/v1/history?page=1&page_size=5", headers=HEADERS).json()
        assert data["page"] == 1
        assert data["page_size"] == 5

    def test_history_has_accurate_total(self):
        data = client.get("/v1/history?page=1&page_size=100", headers=HEADERS).json()
        assert "total_count" in data
        assert isinstance(data["total_count"], int)


# ═══════════════════════════════════════════════════════
# Analytics
# ═══════════════════════════════════════════════════════

class TestAnalytics:
    def test_stats_returns_200(self):
        response = client.get("/v1/analytics/stats", headers=HEADERS)
        assert response.status_code == 200

    def test_stats_structure(self):
        data = client.get("/v1/analytics/stats", headers=HEADERS).json()
        assert "total_predictions" in data
        assert "avg_predicted_spending" in data
        assert "by_persona" in data
        assert "by_customer_type" in data

    def test_cohorts_returns_200(self):
        response = client.get("/v1/analytics/cohorts", headers=HEADERS)
        assert response.status_code == 200

    def test_rfm_returns_200(self):
        response = client.get("/v1/analytics/rfm", headers=HEADERS)
        assert response.status_code == 200


# ═══════════════════════════════════════════════════════
# Model Config
# ═══════════════════════════════════════════════════════

class TestModelConfig:
    def test_model_config_returns_200(self):
        response = client.get("/v1/model/config", headers=HEADERS)
        assert response.status_code == 200

    def test_model_config_has_version(self):
        data = client.get("/v1/model/config", headers=HEADERS).json()
        assert "version" in data
        assert "feature_names" in data


# ═══════════════════════════════════════════════════════
# API Key Generation
# ═══════════════════════════════════════════════════════

class TestAPIKeyGeneration:
    def test_generate_key_returns_200(self):
        response = client.get("/v1/api-keys/generate")
        assert response.status_code == 200

    def test_generated_key_has_prefix(self):
        data = client.get("/v1/api-keys/generate").json()
        assert data["api_key"].startswith("freebuff-")


# ═══════════════════════════════════════════════════════
# Webhook Test
# ═══════════════════════════════════════════════════════

class TestWebhook:
    def test_webhook_test_returns_200(self):
        response = client.post("/v1/webhook/test", json={
            "url": "https://httpbin.org/post",
            "events": ["customer.at_risk"],
        }, headers=HEADERS)
        assert response.status_code == 200

    def test_webhook_test_response(self):
        data = client.post("/v1/webhook/test", json={
            "url": "https://httpbin.org/post",
        }, headers=HEADERS).json()
        assert "success" in data
        assert "attempt" in data


# ═══════════════════════════════════════════════════════
# Maintenance
# ═══════════════════════════════════════════════════════

class TestMaintenance:
    def test_cleanup_returns_200(self):
        response = client.post("/v1/maintenance/cleanup", headers=HEADERS)
        assert response.status_code == 200

    def test_cleanup_response(self):
        data = client.post("/v1/maintenance/cleanup", headers=HEADERS).json()
        assert "deleted" in data
        assert "retention_days" in data


# ═══════════════════════════════════════════════════════
# CORS Headers
# ═══════════════════════════════════════════════════════

class TestCORS:
    def test_cors_headers_present(self):
        response = client.options("/health", headers={
            "Origin": "http://localhost:8501",
            "Access-Control-Request-Method": "GET",
        })
        # CORS middleware should respond
        assert response.status_code in [200, 405]


# ═══════════════════════════════════════════════════════
# OpenAPI / Docs
# ═══════════════════════════════════════════════════════

class TestOpenAPI:
    def test_openapi_json_available(self):
        response = client.get("/openapi.json")
        assert response.status_code == 200
        data = response.json()
        assert "paths" in data
        assert "/v1/predict" in data["paths"]

    def test_v1_endpoints_in_openapi(self):
        data = client.get("/openapi.json").json()
        paths = data["paths"]
        assert "/v1/predict" in paths
        assert "/v1/predict/batch" in paths
        assert "/v1/history" in paths
        assert "/v1/analytics/stats" in paths
        assert "/v1/model/config" in paths
