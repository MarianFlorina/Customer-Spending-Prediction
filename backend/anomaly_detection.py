import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta

def detect_spending_anomalies(predictions: List[Dict]) -> Dict:
    """
    Detect anomalies in spending patterns across multiple predictions.
    
    Args:
        predictions: List of prediction results
        
    Returns:
        Dictionary containing anomaly detection results
    """
    if not predictions:
        return {"anomalies": [], "statistics": {}}
    
    # Extract relevant metrics
    spending_values = [p.get('predicted_spending', 0) for p in predictions]
    churn_risks = [p.get('churn_risk', 0) for p in predictions]
    clv_values = [p.get('clv', 0) for p in predictions]
    order_values = [p.get('avg_order_value', 0) for p in predictions]
    
    # Calculate statistics
    stats = {
        "spending": _calculate_stats(spending_values),
        "churn_risk": _calculate_stats(churn_risks),
        "clv": _calculate_stats(clv_values),
        "order_value": _calculate_stats(order_values)
    }
    
    # Detect anomalies using IQR method
    anomalies = []
    
    for i, pred in enumerate(predictions):
        pred_anomalies = []
        
        # Check spending anomaly
        spending = pred.get('predicted_spending', 0)
        if _is_anomaly(spending, stats['spending']):
            pred_anomalies.append({
                "metric": "predicted_spending",
                "value": spending,
                "severity": _get_severity(spending, stats['spending'])
            })
        
        # Check churn risk anomaly
        churn = pred.get('churn_risk', 0)
        if _is_anomaly(churn, stats['churn_risk']):
            pred_anomalies.append({
                "metric": "churn_risk",
                "value": churn,
                "severity": _get_severity(churn, stats['churn_risk'])
            })
        
        # Check CLV anomaly
        clv = pred.get('clv', 0)
        if _is_anomaly(clv, stats['clv']):
            pred_anomalies.append({
                "metric": "clv",
                "value": clv,
                "severity": _get_severity(clv, stats['clv'])
            })
        
        if pred_anomalies:
            anomalies.append({
                "prediction_index": i,
                "customer_type": pred.get('customer_type', 'Unknown'),
                "anomalies": pred_anomalies
            })
    
    return {
        "anomalies": anomalies,
        "statistics": stats,
        "total_anomalies": len(anomalies),
        "anomaly_rate": round(len(anomalies) / len(predictions) * 100, 2) if predictions else 0
    }

def detect_individual_anomaly(prediction: Dict, historical_stats: Dict = None) -> Dict:
    """
    Detect if a single prediction is anomalous compared to historical data.
    
    Args:
        prediction: Single prediction result
        historical_stats: Historical statistics for comparison
        
    Returns:
        Dictionary indicating if anomaly detected and details
    """
    anomalies = []
    
    # If no historical stats provided, use default thresholds
    if historical_stats is None:
        historical_stats = {
            "spending": {"mean": 5000, "std": 2000, "iqr": 3000, "q1": 3500, "q3": 6500},
            "churn_risk": {"mean": 0.5, "std": 0.2, "iqr": 0.3, "q1": 0.35, "q3": 0.65},
            "clv": {"mean": 100, "std": 50, "iqr": 75, "q1": 62.5, "q3": 137.5},
            "order_value": {"mean": 1000, "std": 400, "iqr": 600, "q1": 700, "q3": 1300}
        }
    
    # Check each metric
    spending = prediction.get('predicted_spending', 0)
    if _is_anomaly(spending, historical_stats['spending']):
        anomalies.append({
            "metric": "predicted_spending",
            "value": spending,
            "severity": _get_severity(spending, historical_stats['spending']),
            "suggestion": "Consider reviewing customer profile for data accuracy"
        })
    
    churn = prediction.get('churn_risk', 0)
    if _is_anomaly(churn, historical_stats['churn_risk']):
        anomalies.append({
            "metric": "churn_risk",
            "value": churn,
            "severity": _get_severity(churn, historical_stats['churn_risk']),
            "suggestion": "Immediate retention action recommended" if churn > 0.8 else "Monitor closely"
        })
    
    clv = prediction.get('clv', 0)
    if _is_anomaly(clv, historical_stats['clv']):
        anomalies.append({
            "metric": "clv",
            "value": clv,
            "severity": _get_severity(clv, historical_stats['clv']),
            "suggestion": "High-value customer - prioritize engagement" if clv > historical_stats['clv']['mean'] else "Low CLV - consider re-engagement"
        })
    
    return {
        "is_anomalous": len(anomalies) > 0,
        "anomalies": anomalies,
        "risk_level": _calculate_risk_level(anomalies)
    }

def get_anomaly_recommendations(anomalies: List[Dict]) -> List[str]:
    """
    Generate recommendations based on detected anomalies.
    
    Args:
        anomalies: List of detected anomalies
        
    Returns:
        List of actionable recommendations
    """
    recommendations = []
    
    for anomaly in anomalies:
        metric = anomaly.get('metric', '')
        severity = anomaly.get('severity', 'low')
        
        if metric == 'predicted_spending':
            if severity == 'high':
                recommendations.append("Investigate potential data quality issues or fraud")
            elif severity == 'medium':
                recommendations.append("Verify customer purchase history accuracy")
        
        elif metric == 'churn_risk':
            if severity == 'high':
                recommendations.append("Initiate immediate customer retention campaign")
                recommendations.append("Assign dedicated account manager")
            elif severity == 'medium':
                recommendations.append("Schedule personalized outreach within 48 hours")
        
        elif metric == 'clv':
            if severity == 'high':
                recommendations.append("Enroll in VIP customer program")
                recommendations.append("Offer exclusive loyalty rewards")
            elif severity == 'medium':
                recommendations.append("Create targeted upsell opportunities")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_recommendations = []
    for rec in recommendations:
        if rec not in seen:
            seen.add(rec)
            unique_recommendations.append(rec)
    
    return unique_recommendations

def _calculate_stats(values: List[float]) -> Dict:
    """Calculate statistical measures for a list of values."""
    if not values:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "iqr": 0}
    
    arr = np.array(values)
    q1 = np.percentile(arr, 25)
    q3 = np.percentile(arr, 75)
    iqr = q3 - q1
    
    return {
        "mean": round(float(np.mean(arr)), 2),
        "std": round(float(np.std(arr)), 2),
        "min": round(float(np.min(arr)), 2),
        "max": round(float(np.max(arr)), 2),
        "iqr": round(float(iqr), 2),
        "q1": round(float(q1), 2),
        "q3": round(float(q3), 2)
    }

def _is_anomaly(value: float, stats: Dict) -> bool:
    """Check if a value is an anomaly using IQR method."""
    if stats['iqr'] == 0:
        return False
    
    lower_bound = stats['q1'] - 1.5 * stats['iqr']
    upper_bound = stats['q3'] + 1.5 * stats['iqr']
    
    return value < lower_bound or value > upper_bound

def _get_severity(value: float, stats: Dict) -> str:
    """Determine the severity of an anomaly."""
    if stats['iqr'] == 0:
        return 'low'
    
    lower_bound = stats['q1'] - 1.5 * stats['iqr']
    upper_bound = stats['q3'] + 1.5 * stats['iqr']
    
    # Check for extreme anomalies (beyond 3x IQR)
    extreme_lower = stats['q1'] - 3 * stats['iqr']
    extreme_upper = stats['q3'] + 3 * stats['iqr']
    
    if value < extreme_lower or value > extreme_upper:
        return 'high'
    elif value < lower_bound or value > upper_bound:
        return 'medium'
    else:
        return 'low'

def _calculate_risk_level(anomalies: List[Dict]) -> str:
    """Calculate overall risk level based on anomalies."""
    if not anomalies:
        return 'low'
    
    high_count = sum(1 for a in anomalies if a.get('severity') == 'high')
    medium_count = sum(1 for a in anomalies if a.get('severity') == 'medium')
    
    if high_count > 0:
        return 'high'
    elif medium_count > 0:
        return 'medium'
    else:
        return 'low'
