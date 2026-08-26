"""
Model Explainability Module — SHAP-based feature contribution analysis.

Provides human-readable explanations of why the model made a specific prediction,
including feature contributions, plain-language summaries, and actionable insights.
"""
import os
import json
import logging
import numpy as np
from typing import Dict, List, Optional
from datetime import date

logger = logging.getLogger(__name__)

# Feature names matching the model's training columns
FEATURE_NAMES = [
    "total_spent", "total_orders", "avg_order_value", "recency",
    "CLV", "churn_risk", "norm_spent", "norm_orders", "norm_recency",
]

# Human-readable feature descriptions
FEATURE_DESCRIPTIONS = {
    "total_spent": "Total amount the customer has spent",
    "total_orders": "Number of orders placed",
    "avg_order_value": "Average value per order",
    "recency": "Days since last purchase",
    "CLV": "Customer Lifetime Value score",
    "churn_risk": "Probability of customer leaving",
    "norm_spent": "Spending relative to typical customers",
    "norm_orders": "Order frequency relative to typical customers",
    "norm_recency": "Recency score (higher = more recent)",
}

# Impact descriptions for different contribution directions
IMPACT_TEMPLATES = {
    "total_spent": {
        "positive": "High total spending of ₹{value:,.0f} strongly increases the predicted future spending",
        "negative": "Low total spending of ₹{value:,.0f} limits the predicted future spending",
    },
    "total_orders": {
        "positive": "Frequent buyer with {value} orders — higher order count drives more spending",
        "negative": "Infrequent buyer with only {value} order(s) — low frequency limits growth",
    },
    "avg_order_value": {
        "positive": "Average order of ₹{value:,.0f} indicates strong purchasing power per transaction",
        "negative": "Low average order of ₹{value:,.0f} suggests price sensitivity",
    },
    "recency": {
        "positive": "Recent purchase ({value} days ago) — customer is actively engaged",
        "negative": "Last purchase was {value} days ago — long inactivity reduces predicted spending",
    },
    "CLV": {
        "positive": "High CLV of {value:.2f} indicates a valuable long-term customer",
        "negative": "Low CLV of {value:.2f} suggests limited historical engagement",
    },
    "churn_risk": {
        "positive": "Low churn risk ({value:.0%}) — stable customer likely to continue spending",
        "negative": "High churn risk ({value:.0%}) — customer may leave, reducing future spending",
    },
    "norm_spent": {
        "positive": "Spending is above average relative to the customer base",
        "negative": "Spending is below average relative to the customer base",
    },
    "norm_orders": {
        "positive": "Order frequency is above average",
        "negative": "Order frequency is below average",
    },
    "norm_recency": {
        "positive": "Customer recently active — strong engagement signal",
        "negative": "Customer has been inactive — weak engagement signal",
    },
}


def explain_prediction(data, prediction_result: Dict, model=None) -> Dict:
    """
    Generate a full explainability report for a single prediction.

    Uses SHAP TreeExplainer when available, falls back to feature importance
    heuristics when SHAP is unavailable.

    Args:
        data: CustomerInput object
        prediction_result: Output from predict_future_spending()
        model: Optional pre-loaded model (avoids re-loading)

    Returns:
        Explainability report with feature contributions, summary, and insights
    """
    from backend.model_utils import get_model_config

    # Compute feature values for this prediction
    avg_order_value = data.total_spent / data.total_orders
    recency = (date.today() - data.last_purchase_date).days

    config = get_model_config()
    max_spent = config.get("max_spent", 12000)
    max_orders = config.get("max_orders", 12)
    max_recency = config.get("max_recency", 730)

    norm_spent = min(data.total_spent / max_spent, 1)
    norm_orders = min(data.total_orders / max_orders, 1)
    norm_recency = 1 - min(recency / max_recency, 1)

    CLV = prediction_result.get("CLV", avg_order_value * data.total_orders * (1 / (recency + 1)))
    churn_risk = prediction_result.get("churn_risk", 0)

    feature_values = {
        "total_spent": data.total_spent,
        "total_orders": data.total_orders,
        "avg_order_value": round(avg_order_value, 2),
        "recency": recency,
        "CLV": round(CLV, 4),
        "churn_risk": round(churn_risk, 4),
        "norm_spent": round(norm_spent, 4),
        "norm_orders": round(norm_orders, 4),
        "norm_recency": round(norm_recency, 4),
    }

    X = np.array([[
        data.total_spent, data.total_orders, avg_order_value,
        recency, CLV, churn_risk, norm_spent, norm_orders, norm_recency,
    ]])

    # Try SHAP first, fall back to feature importance
    contributions = _compute_shap_contributions(X, model)
    if contributions is None:
        contributions = _compute_importance_contributions(X, feature_values)

    # Build per-feature explanations
    features = []
    for fname in FEATURE_NAMES:
        val = feature_values[fname]
        contrib = contributions.get(fname, 0.0)
        features.append({
            "name": fname,
            "description": FEATURE_DESCRIPTIONS.get(fname, fname),
            "value": _format_value(fname, val),
            "raw_value": val,
            "contribution": round(float(contrib), 4),
            "direction": "positive" if contrib > 0 else "negative" if contrib < 0 else "neutral",
            "magnitude": abs(float(contrib)),
        })

    # Sort by absolute contribution (most impactful first)
    features.sort(key=lambda f: f["magnitude"], reverse=True)

    # Top contributors
    top_positive = [f for f in features if f["direction"] == "positive"][:3]
    top_negative = [f for f in features if f["direction"] == "negative"][:3]

    # Generate natural language summary
    summary = _generate_summary(features, top_positive, top_negative, prediction_result)

    # Generate actionable insights
    insights = _generate_insights(features, prediction_result)

    return {
        "prediction": prediction_result.get("prediction", 0),
        "features": features,
        "top_positive_drivers": [_summarize_feature(f, "positive") for f in top_positive],
        "top_negative_drivers": [_summarize_feature(f, "negative") for f in top_negative],
        "summary": summary,
        "insights": insights,
        "feature_importance_ranking": [
            {"feature": f["name"], "importance": f["magnitude"]}
            for f in features
        ],
    }


def _compute_shap_contributions(X: np.ndarray, model=None) -> Optional[Dict[str, float]]:
    """Compute SHAP values for the prediction."""
    try:
        import shap

        if model is None:
            from backend.model_utils import _model
            model = _model

        if model is None:
            return None

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        contributions = {}
        for i, fname in enumerate(FEATURE_NAMES):
            contributions[fname] = float(shap_values[0][i])

        logger.info("SHAP explainability computed successfully")
        return contributions

    except Exception as e:
        logger.warning(f"SHAP computation failed, using fallback: {e}")
        return None


def _compute_importance_contributions(X: np.ndarray, feature_values: Dict) -> Dict[str, float]:
    """
    Fallback: estimate feature contributions using feature importance weights
    and normalized feature values.
    """
    # Feature importance weights (from the model's feature_importance.csv)
    importance_weights = {
        "avg_order_value": 0.424,
        "total_spent": 0.320,
        "norm_spent": 0.186,
        "CLV": 0.068,
        "norm_recency": 0.0004,
        "recency": 0.0003,
        "churn_risk": 0.0002,
        "total_orders": 0.0,
        "norm_orders": 0.0,
    }

    contributions = {}
    for fname in FEATURE_NAMES:
        weight = importance_weights.get(fname, 0.01)
        val = feature_values.get(fname, 0)

        # Scale contribution: importance × normalized value × direction
        # Positive features (higher = more spending)
        if fname in ("total_spent", "total_orders", "avg_order_value", "CLV", "norm_spent", "norm_orders", "norm_recency"):
            contributions[fname] = weight * val
        # Negative features (higher = less spending)
        elif fname in ("recency", "churn_risk"):
            contributions[fname] = -weight * val
        else:
            contributions[fname] = weight * val

    return contributions


def _format_value(feature_name: str, value) -> str:
    """Format a feature value for display."""
    if feature_name in ("total_spent", "avg_order_value", "CLV"):
        return f"₹{value:,.2f}"
    elif feature_name in ("churn_risk", "norm_spent", "norm_orders", "norm_recency"):
        return f"{value:.2%}" if value <= 1 else f"{value:.2f}"
    elif feature_name == "recency":
        return f"{value} days"
    elif feature_name == "total_orders":
        return str(int(value))
    return str(value)


def _summarize_feature(feature: Dict, direction: str) -> Dict:
    """Create a summary for a single feature contribution."""
    name = feature["name"]
    val = feature["raw_value"]

    template = IMPACT_TEMPLATES.get(name, {}).get(direction, "")
    if template:
        try:
            explanation = template.format(value=val)
        except (KeyError, ValueError):
            explanation = f"{feature['description']}: {feature['value']}"
    else:
        explanation = f"{feature['description']}: {feature['value']}"

    return {
        "feature": name,
        "value": feature["value"],
        "contribution": feature["contribution"],
        "explanation": explanation,
    }


def _generate_summary(
    features: List[Dict],
    top_positive: List[Dict],
    top_negative: List[Dict],
    prediction_result: Dict,
) -> Dict:
    """Generate a natural language summary of the prediction."""
    prediction = prediction_result.get("prediction", 0)
    persona = prediction_result.get("persona", "Unknown")
    churn = prediction_result.get("churn_risk", 0)

    # Build summary paragraphs
    paragraphs = []

    # Overall prediction
    paragraphs.append(
        f"The model predicts a future spending of ₹{prediction:,.2f} for this customer, "
        f"classifying them as a **{persona}**."
    )

    # Key drivers
    if top_positive:
        pos_features = ", ".join([f["name"].replace("_", " ") for f in top_positive[:2]])
        paragraphs.append(
            f"The strongest positive drivers are **{pos_features}** — "
            f"these factors push the prediction upward."
        )

    if top_negative:
        neg_features = ", ".join([f["name"].replace("_", " ") for f in top_negative[:2]])
        paragraphs.append(
            f"Key limiting factors include **{neg_features}** — "
            f"these reduce the predicted spending."
        )

    # Churn context
    if churn > 0.7:
        paragraphs.append(
            f"⚠️ With a churn risk of {churn:.0%}, there is a significant chance this customer "
            f"may reduce or stop spending. Retention efforts are recommended."
        )
    elif churn < 0.3:
        paragraphs.append(
            f"✅ With a low churn risk of {churn:.0%}, this customer shows strong engagement "
            f"and is likely to continue spending."
        )

    return {
        "paragraphs": paragraphs,
        "tldr": _generate_tldr(features, prediction, persona),
    }


def _generate_tldr(features: List[Dict], prediction: float, persona: str) -> str:
    """Generate a one-line TL;DR."""
    top = features[0] if features else None
    if top:
        return (
            f"₹{prediction:,.0f} predicted ({persona}) — "
            f"main driver: {top['name'].replace('_', ' ')} ({top['direction']})"
        )
    return f"₹{prediction:,.0f} predicted for a {persona}"


def _generate_insights(features: List[Dict], prediction_result: Dict) -> List[Dict]:
    """Generate actionable business insights based on the prediction."""
    insights = []
    persona = prediction_result.get("persona", "")
    churn = prediction_result.get("churn_risk", 0)
    clv = prediction_result.get("CLV", 0)
    recency = None

    for f in features:
        if f["name"] == "recency":
            recency = f["raw_value"]

    # Retention insight
    if churn > 0.6:
        insights.append({
            "type": "retention",
            "priority": "high",
            "title": "Customer Retention Risk",
            "description": f"Churn risk is {churn:.0%}. This customer needs immediate attention.",
            "actions": [
                "Send personalized re-engagement email within 24 hours",
                "Offer a loyalty discount or exclusive deal",
                "Assign a dedicated account manager for high-touch outreach",
            ],
        })

    # Upsell insight
    if persona == "Loyal Customer" or clv > 100:
        insights.append({
            "type": "upsell",
            "priority": "medium",
            "title": "Upsell Opportunity",
            "description": "This customer has strong purchasing history — prime for upselling.",
            "actions": [
                "Recommend premium product bundles",
                "Enroll in VIP/loyalty program",
                "Offer early access to new products",
            ],
        })

    # Recency insight
    if recency and recency > 180:
        insights.append({
            "type": "reactivation",
            "priority": "high",
            "title": "Reactivation Needed",
            "description": f"Last purchase was {recency} days ago — customer may have disengaged.",
            "actions": [
                "Send 'We miss you' campaign with special offer",
                "Survey to understand why they stopped purchasing",
                "Retarget with relevant product ads",
            ],
        })

    # New customer insight
    if persona == "New / Occasional Customer":
        insights.append({
            "type": "onboarding",
            "priority": "medium",
            "title": "Nurture New Customer",
            "description": "This is a new customer — focus on building the relationship.",
            "actions": [
                "Send welcome series with product recommendations",
                "Offer second-purchase discount",
                "Provide excellent post-purchase support",
            ],
        })

    # High value insight
    if clv > 200:
        insights.append({
            "type": "high_value",
            "priority": "high",
            "title": "High-Value Customer",
            "description": f"CLV of ₹{clv:,.0f} — this customer represents significant revenue potential.",
            "actions": [
                "Provide white-glove customer service",
                "Create personalized product recommendations",
                "Offer exclusive membership benefits",
            ],
        })

    # Default insight if none triggered
    if not insights:
        insights.append({
            "type": "general",
            "priority": "low",
            "title": "Standard Engagement",
            "description": "This customer fits a typical spending pattern.",
            "actions": [
                "Continue standard marketing communications",
                "Monitor for changes in spending behavior",
                "Include in regular promotional campaigns",
            ],
        })

    return insights
