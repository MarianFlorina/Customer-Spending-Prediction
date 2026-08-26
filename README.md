# 💳 Customer Spending Prediction Platform

> ML-powered customer spending prediction & analytics platform with SHAP explainability, real-time drift detection, and production-grade monitoring.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.39+-FF4B4B?logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.5+-F7931E?logo=scikitlearn)
![Docker](https://img.shields.io/badge/Docker-24+-2496ED?logo=docker)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Features

### Core Prediction
- **Single & Batch Prediction** — Predict future spending for individual customers or up to 100 at once
- **Confidence Intervals** — 95% CI using tree-based uncertainty estimation
- **Customer Persona Classification** — At-Risk, Loyal, New/Occasional
- **Churn Risk Assessment** — Probability of customer attrition
- **CLV Calculation** — Customer Lifetime Value scoring

### Explainability 🔍
- **SHAP Integration** — TreeExplainer for exact feature contributions
- **Feature Waterfall Chart** — Visual breakdown of what drives each prediction
- **Plain-Language Summaries** — Human-readable explanations
- **Actionable Insights** — Auto-generated business recommendations with priority levels

### Analytics
- **Cohort Analysis** — Group customers by spending period
- **RFM Segmentation** — Recency, Frequency, Monetary analysis
- **Data Drift Detection** — Monitor distribution shifts in predictions
- **Historical Tracking** — SQLite-backed prediction history with export

### Production Hardening
- **API Key Authentication** — Secure access with HMAC key generation
- **Rate Limiting** — File-backed, per-key limits with headers
- **Response Caching** — SHA-256 keyed, 5-min TTL, thread-safe LRU
- **Circuit Breaker** — Auto-trips on webhook failures, auto-recovers
- **Prometheus Metrics** — Request counts, duration histograms, p50/p95/p99
- **Request Tracing** — X-Request-ID on every response
- **Structured Logging** — Rotating file handler with optional JSON format
- **Database Migrations** — Versioned schema with automatic upgrades
- **Data Retention** — Configurable cleanup with `POST /v1/maintenance/cleanup`

### Integrations
- **Webhook Notifications** — HMAC-signed, retry with exponential backoff
- **Email/SMS Alerts** — SMTP + SMS provider ready
- **Model Retraining** — Retrain with new data via API endpoint

---

## 📁 Project Structure

```
Customer-Spending-Prediction/
├── backend/
│   ├── main.py              # FastAPI application & routes
│   ├── schemas.py           # Pydantic models & enums
│   ├── model_utils.py       # ML model loading & prediction
│   ├── explainability.py    # SHAP-based explanations
│   ├── database.py          # SQLite with migrations
│   ├── auth.py              # API key authentication
│   ├── cache.py             # In-memory response cache
│   ├── metrics.py           # Prometheus metrics collector
│   ├── circuit_breaker.py   # Circuit breaker pattern
│   ├── webhooks.py          # Webhook delivery with retry
│   ├── notifications.py     # Email/SMS notifications
│   ├── anomaly_detection.py # IQR-based anomaly detection
│   └── logging_config.py    # Structured logging setup
├── frontend/
│   └── app.py               # Streamlit dashboard
├── models/
│   ├── spending_model.pkl   # Pre-trained GradientBoostingRegressor
│   └── feature_importance.csv
├── notebooks/
│   ├── 01_preprocessing.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_validation.ipynb
├── data/                    # Customer datasets
├── tests/
│   └── test_api.py          # 43 tests (all passing)
├── .github/workflows/
│   └── ci.yml               # GitHub Actions CI/CD
├── Dockerfile               # Backend (multi-stage)
├── Dockerfile.frontend      # Frontend (multi-stage)
├── docker-compose.yml       # Full-stack orchestration
├── requirements.txt
└── .env.example
```

---

## 🏃 Quick Start

### Prerequisites
- Python 3.10+
- pip

### 1. Clone & Install

```bash
git clone https://github.com/MarianFlorina/Customer-Spending-Prediction.git
cd Customer-Spending-Prediction

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate    # Mac/Linux

pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your settings (optional for dev)
```

### 3. Run

**Backend (FastAPI):**
```bash
uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

**Frontend (Streamlit):**
```bash
streamlit run frontend/app.py --server.port 8501
```

**Or use Docker:**
```bash
docker-compose up --build
```

### 4. Open

- **Dashboard:** http://localhost:8501
- **API Docs:** http://localhost:8000/docs
- **Health Check:** http://localhost:8000/health
- **Metrics:** http://localhost:8000/metrics

---

## 📡 API Endpoints

### Prediction
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/predict` | Single customer prediction |
| POST | `/v1/predict/explain` | Prediction with SHAP explainability |
| POST | `/v1/predict/batch` | Batch prediction (up to 100) |

### Analytics
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/v1/analytics/stats` | Aggregate statistics |
| GET | `/v1/analytics/cohorts` | Cohort analysis by spending period |
| GET | `/v1/analytics/rfm` | RFM segmentation |
| GET | `/v1/drift/detect` | Data drift detection |

### History & Maintenance
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/v1/history` | Paginated prediction history |
| POST | `/v1/maintenance/cleanup` | Remove old predictions |
| POST | `/v1/cache/clear` | Clear prediction cache |

### Model Management
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/v1/model/config` | Model version & config |
| POST | `/v1/model/retrain` | Retrain with new data |

### Monitoring
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/metrics` | Prometheus metrics |
| GET | `/v1/metrics` | JSON metrics |
| GET | `/v1/cache/stats` | Cache statistics |
| GET | `/v1/circuit-breakers` | Circuit breaker status |

### Webhooks & Auth
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/webhook/test` | Test webhook delivery |
| GET | `/v1/api-keys/generate` | Generate new API key |

> **Note:** All `/v1/` endpoints require `X-API-Key` header. Any key starting with `freebuff-` is accepted in dev mode.

---

## 📊 Example Usage

### Single Prediction
```bash
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: freebuff-dev-key" \
  -d '{
    "total_spent": 5000,
    "total_orders": 3,
    "last_purchase_date": "2024-06-01",
    "spending_period": "Last 30 Days",
    "customer_type": "New",
    "product_category": "Electronics",
    "discount_sensitivity": "Low"
  }'
```

### Prediction with Explainability
```bash
curl -X POST http://localhost:8000/v1/predict/explain \
  -H "Content-Type: application/json" \
  -H "X-API-Key: freebuff-dev-key" \
  -d '{
    "total_spent": 5000,
    "total_orders": 3,
    "last_purchase_date": "2024-06-01",
    "spending_period": "Last 30 Days",
    "customer_type": "New",
    "product_category": "Electronics",
    "discount_sensitivity": "Low"
  }'
```

### Data Drift Detection
```bash
curl "http://localhost:8000/v1/drift/detect?window=50" \
  -H "X-API-Key: freebuff-dev-key"
```

---

## 🧪 Testing

```bash
# Run all 43 tests
python -m pytest tests/test_api.py -v

# Run with coverage
python -m pytest tests/test_api.py -v --tb=short
```

### Test Coverage
| Suite | Tests | Status |
|-------|-------|--------|
| Health Check | 3 | ✅ |
| Authentication | 4 | ✅ |
| Single Prediction | 10 | ✅ |
| Backward Compatibility | 2 | ✅ |
| Batch Prediction | 5 | ✅ |
| History | 4 | ✅ |
| Analytics | 4 | ✅ |
| Model Config | 2 | ✅ |
| API Key Generation | 2 | ✅ |
| Webhook Test | 2 | ✅ |
| Maintenance | 2 | ✅ |
| CORS | 1 | ✅ |
| OpenAPI | 2 | ✅ |

---

## 🐳 Docker

### Build & Run
```bash
docker-compose up --build
```

### Services
| Service | Port | Description |
|---------|------|-------------|
| backend | 8000 | FastAPI API |
| frontend | 8501 | Streamlit Dashboard |

### Environment Variables
```bash
# Backend
API_KEY=your-secret-key
CORS_ORIGINS=http://localhost:8501
DATABASE_PATH=data/predictions.db
CACHE_TTL_SECONDS=300
LOG_LEVEL=INFO

# Webhooks
WEBHOOK_SECRET=your-webhook-secret
WEBHOOK_MAX_RETRIES=3

# Rate Limiting
RATE_LIMIT_WINDOW=60
MAX_REQUESTS_PER_WINDOW=100

# Notifications
EMAIL_ENABLED=false
SMS_ENABLED=false
```

---

## 📈 Model Details

- **Algorithm:** GradientBoostingRegressor (Scikit-learn)
- **Features:** 9 engineered features (spending, orders, recency, CLV, churn risk, normalized metrics)
- **Explainability:** SHAP TreeExplainer with fallback to feature importance weights
- **Confidence Intervals:** Tree-based uncertainty estimation (std deviation across estimators)

---

## 🏗️ Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Streamlit  │────▶│    FastAPI    │────▶│   Scikit-learn│
│  Dashboard   │     │    Backend    │     │     Model     │
└──────────────┘     └──────┬───────┘     └──────────────┘
                            │
                    ┌───────┴───────┐
                    │               │
              ┌─────┴─────┐  ┌─────┴─────┐
              │  SQLite   │  │   SHAP    │
              │ Database  │  │Explainer  │
              └───────────┘  └───────────┘
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) — Modern Python web framework
- [Streamlit](https://streamlit.io/) — Rapid ML app development
- [SHAP](https://shap.readthedocs.io/) — Model explainability
- [Scikit-learn](https://scikit-learn.org/) — Machine learning library
- [Plotly](https://plotly.com/python/) — Interactive visualizations

---

<p align="center">
  Built with ❤️ by <a href="https://github.com/MarianFlorina">MarianFlorina</a>
</p>
