import os
import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Tuple, Optional
import numpy as np
import joblib

logger = logging.getLogger(__name__)

# ─── Model Loading (graceful with fallback) ───

MODEL_PATH = os.getenv("MODEL_PATH", "models/spending_model.pkl")
MODEL_CONFIG_PATH = os.getenv("MODEL_CONFIG_PATH", "models/model_config.json")

_model = None
_model_config = None


def _load_model():
    """Load the ML model with graceful error handling."""
    global _model, _model_config

    try:
        _model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded successfully from {MODEL_PATH}: {type(_model).__name__}")
    except FileNotFoundError:
        logger.error(f"Model file not found at {MODEL_PATH}. Using fallback predictions.")
        _model = None
    except Exception as e:
        logger.error(f"Failed to load model: {e}. Using fallback predictions.")
        _model = None

    # Load model config (max constants, version, etc.)
    try:
        if Path(MODEL_CONFIG_PATH).exists():
            with open(MODEL_CONFIG_PATH, "r") as f:
                _model_config = json.load(f)
            logger.info(f"Model config loaded: version={_model_config.get('version', 'unknown')}")
        else:
            _model_config = _derive_default_config()
            logger.info("No model config found, using derived defaults")
    except Exception as e:
        logger.warning(f"Could not load model config: {e}, using defaults")
        _model_config = _derive_default_config()


def _derive_default_config() -> dict:
    """Derive default model configuration."""
    return {
        "version": "2.0.0",
        "created_at": datetime.utcnow().isoformat(),
        "max_spent": 12000,
        "max_orders": 12,
        "max_recency": 730,
        "confidence_interval_alpha": 0.05,
        "feature_names": [
            "total_spent", "total_orders", "avg_order_value", "recency",
            "CLV", "churn_risk", "norm_spent", "norm_orders", "norm_recency",
        ],
        "training_samples": None,
        "model_type": "GradientBoostingRegressor",
    }


def get_model_config() -> dict:
    """Get the current model configuration."""
    if _model_config is None:
        _load_model()
    return _model_config or _derive_default_config()


def get_model_version() -> str:
    """Get the current model version."""
    return get_model_config().get("version", "unknown")


# ─── Feature Engineering ───

MAX_SPENT = 12000
MAX_ORDERS = 12
MAX_RECENCY = 730


def _get_config_value(key: str, default) -> float:
    """Get a config value with fallback to default."""
    config = get_model_config()
    return config.get(key, default)


def predict_future_spending(data) -> Dict:
    """
    Predict future spending for a customer.

    Returns prediction with confidence intervals and all derived metrics.
    """
    # Feature Engineering
    avg_order_value = data.total_spent / data.total_orders
    recency = (date.today() - data.last_purchase_date).days

    max_spent = _get_config_value("max_spent", MAX_SPENT)
    max_orders = _get_config_value("max_orders", MAX_ORDERS)
    max_recency = _get_config_value("max_recency", MAX_RECENCY)

    norm_spent = min(data.total_spent / max_spent, 1)
    norm_orders = min(data.total_orders / max_orders, 1)
    norm_recency = 1 - min(recency / max_recency, 1)

    CLV = avg_order_value * data.total_orders * (1 / (recency + 1))

    churn_risk = (
        (recency / max_recency) * 0.6 +
        (1 - norm_orders) * 0.4
    )

    # Behavioral Adjustments
    type_multiplier = {"New": 0.9, "Returning": 1.0, "Loyal": 1.15}
    discount_multiplier = {"Low": 1.1, "Medium": 1.0, "High": 0.9}

    multiplier = (
        type_multiplier.get(data.customer_type, 1.0) *
        discount_multiplier.get(data.discount_sensitivity, 1.0)
    )

    X = np.array([[
        data.total_spent,
        data.total_orders,
        avg_order_value,
        recency,
        CLV,
        churn_risk,
        norm_spent,
        norm_orders,
        norm_recency,
    ]])

    # Model prediction with confidence interval
    prediction, ci_low, ci_high = _predict_with_confidence(X, multiplier)

    # Persona classification
    if churn_risk > 0.6:
        persona = "At-Risk Customer"
        recommendation = "Send re-engagement offers"
    elif data.total_orders >= 5:
        persona = "Loyal Customer"
        recommendation = "Upsell premium products"
    else:
        persona = "New / Occasional Customer"
        recommendation = "Onboarding & discounts"

    return {
        "prediction": round(float(prediction), 2),
        "confidence_interval": {
            "lower": round(float(ci_low), 2),
            "upper": round(float(ci_high), 2),
            "alpha": _get_config_value("confidence_interval_alpha", 0.05),
        },
        "avg_order_value": round(avg_order_value, 2),
        "recency": recency,
        "CLV": round(CLV, 2),
        "churn_risk": round(churn_risk, 2),
        "persona": persona,
        "recommendation": recommendation,
        "model_version": get_model_version(),
    }


def _predict_with_confidence(X: np.ndarray, multiplier: float) -> Tuple[float, float, float]:
    """
    Make prediction with confidence interval estimation.

    If the model supports `predict` with std estimation (e.g. GradientBoosting),
    we use it. Otherwise we fall back to a heuristic based on the model config.
    """
    alpha = _get_config_value("confidence_interval_alpha", 0.05)
    from scipy import stats

    if _model is not None:
        try:
            raw_pred = _model.predict(X)[0] * multiplier

            # For tree-based models, estimate uncertainty via tree predictions
            if hasattr(_model, "estimators_"):
                tree_preds = np.array([
                    tree.predict(X)[0] for tree in _model.estimators_.flatten()
                ])
                std = float(np.std(tree_preds)) * multiplier
            else:
                std = abs(raw_pred) * 0.1  # 10% heuristic fallback

            z_score = stats.norm.ppf(1 - alpha / 2)
            ci_low = raw_pred - z_score * std
            ci_high = raw_pred + z_score * std

            return raw_pred, ci_low, ci_high
        except Exception as e:
            logger.warning(f"Model prediction failed, using fallback: {e}")

    # Fallback: heuristic prediction
    base = float(X[0][0]) * multiplier * 0.95  # rough estimate
    std = base * 0.15
    z_score = stats.norm.ppf(1 - alpha / 2)
    ci_low = base - z_score * std
    ci_high = base + z_score * std
    return base, ci_low, ci_high


def retrain_model(training_data_path: str = "data/training_data.csv") -> Dict:
    """
    Retrain the model with new data.

    Returns training metrics and saves the new model.
    """
    import pandas as pd
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

    logger.info(f"Starting model retraining with data from {training_data_path}")

    if not Path(training_data_path).exists():
        raise FileNotFoundError(f"Training data not found at {training_data_path}")

    df = pd.read_csv(training_data_path)

    required_cols = [
        "total_spent", "total_orders", "avg_order_value", "recency",
        "CLV", "churn_risk", "norm_spent", "norm_orders", "norm_recency",
        "target_spending",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    feature_cols = [c for c in required_cols if c != "target_spending"]
    X = df[feature_cols].values
    y = df["target_spending"].values

    # Train
    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
    )
    model.fit(X, y)

    # Cross-validate
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")

    # Test metrics
    y_pred = model.predict(X)
    metrics = {
        "r2_score": round(float(r2_score(y, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y, y_pred))), 4),
        "mae": round(float(mean_absolute_error(y, y_pred)), 4),
        "cv_r2_mean": round(float(cv_scores.mean()), 4),
        "cv_r2_std": round(float(cv_scores.std()), 4),
        "training_samples": len(X),
    }

    # Derive new max constants from training data
    new_config = {
        "version": f"2.1.{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
        "created_at": datetime.utcnow().isoformat(),
        "max_spent": round(float(df["total_spent"].quantile(0.99)), 0),
        "max_orders": int(df["total_orders"].quantile(0.99)),
        "max_recency": int(df["recency"].quantile(0.99)) if "recency" in df.columns else MAX_RECENCY,
        "confidence_interval_alpha": 0.05,
        "feature_names": feature_cols,
        "training_samples": len(X),
        "model_type": type(model).__name__,
        "training_metrics": metrics,
    }

    # Save model and config
    backup_path = f"models/spending_model_{get_model_version()}.pkl"
    if Path(MODEL_PATH).exists():
        Path(MODEL_PATH).rename(backup_path)
        logger.info(f"Previous model backed up to {backup_path}")

    joblib.dump(model, MODEL_PATH)
    with open(MODEL_CONFIG_PATH, "w") as f:
        json.dump(new_config, f, indent=2)

    # Reload
    global _model, _model_config
    _model = model
    _model_config = new_config

    logger.info(f"Model retrained successfully. Version: {new_config['version']}")
    return {
        "status": "success",
        "version": new_config["version"],
        "metrics": metrics,
    }


# Initialize on import
_load_model()
