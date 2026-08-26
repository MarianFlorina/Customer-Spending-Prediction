import os
from typing import Dict, List, Optional
from datetime import datetime
import json

# Notification configuration
EMAIL_ENABLED = os.getenv("EMAIL_ENABLED", "false").lower() == "true"
SMS_ENABLED = os.getenv("SMS_ENABLED", "false").lower() == "true"

# SMTP configuration (for email)
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")

# SMS configuration (placeholder - integrate with Twilio, AWS SNS, etc.)
SMS_API_KEY = os.getenv("SMS_API_KEY", "")
SMS_API_SECRET = os.getenv("SMS_API_SECRET", "")

def send_email_notification(
    to_email: str,
    subject: str,
    body: str,
    html_body: Optional[str] = None
) -> Dict:
    """
    Send an email notification.
    
    Args:
        to_email: Recipient email address
        subject: Email subject
        body: Plain text body
        html_body: Optional HTML body
        
    Returns:
        Dictionary with send status
    """
    if not EMAIL_ENABLED:
        return {
            "success": False,
            "error": "Email notifications disabled",
            "recipient": to_email
        }
    
    # In production, implement actual SMTP sending
    # This is a placeholder implementation
    try:
        # import smtplib
        # from email.mime.text import MIMEText
        # from email.mime.multipart import MIMEMultipart
        
        # msg = MIMEMultipart('alternative')
        # msg['Subject'] = subject
        # msg['From'] = SMTP_USER
        # msg['To'] = to_email
        
        # msg.attach(MIMEText(body, 'plain'))
        # if html_body:
        #     msg.attach(MIMEText(html_body, 'html'))
        
        # with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        #     server.starttls()
        #     server.login(SMTP_USER, SMTP_PASSWORD)
        #     server.send_message(msg)
        
        return {
            "success": True,
            "recipient": to_email,
            "subject": subject,
            "timestamp": datetime.utcnow().isoformat(),
            "message": "Email sent successfully (simulated)"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "recipient": to_email
        }

def send_sms_notification(
    phone_number: str,
    message: str
) -> Dict:
    """
    Send an SMS notification.
    
    Args:
        phone_number: Recipient phone number
        message: SMS message
        
    Returns:
        Dictionary with send status
    """
    if not SMS_ENABLED:
        return {
            "success": False,
            "error": "SMS notifications disabled",
            "recipient": phone_number
        }
    
    # In production, integrate with SMS provider (Twilio, AWS SNS, etc.)
    # This is a placeholder implementation
    try:
        return {
            "success": True,
            "recipient": phone_number,
            "message": "SMS sent successfully (simulated)",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "recipient": phone_number
        }

def notify_marketing_team_at_risk(
    recipients: List[Dict],
    prediction: Dict,
    customer_data: Dict
) -> List[Dict]:
    """
    Notify marketing team about at-risk customers.
    
    Args:
        recipients: List of recipient dictionaries with 'email' and/or 'phone'
        prediction: Prediction result
        customer_data: Original customer data
        
    Returns:
        List of notification results
    """
    results = []
    
    churn_risk = prediction.get('churn_risk', 0)
    persona = prediction.get('persona', 'Unknown')
    recommendation = prediction.get('recommendation', 'No recommendation')
    
    subject = f"🚨 At-Risk Customer Alert - {persona}"
    
    body = f"""
Customer At-Risk Alert

Customer Type: {customer_data.get('customer_type', 'Unknown')}
Category: {customer_data.get('product_category', 'Unknown')}
Total Spent: ₹{customer_data.get('total_spent', 0):,.2f}
Total Orders: {customer_data.get('total_orders', 0)}

Risk Assessment:
- Churn Risk: {churn_risk:.2%}
- Persona: {persona}
- Recommended Action: {recommendation}

Please take immediate action to retain this customer.
    """.strip()
    
    html_body = f"""
<html>
<body>
<h2>🚨 Customer At-Risk Alert</h2>
<table>
<tr><td><strong>Customer Type:</strong></td><td>{customer_data.get('customer_type', 'Unknown')}</td></tr>
<tr><td><strong>Category:</strong></td><td>{customer_data.get('product_category', 'Unknown')}</td></tr>
<tr><td><strong>Total Spent:</strong></td><td>₹{customer_data.get('total_spent', 0):,.2f}</td></tr>
<tr><td><strong>Total Orders:</strong></td><td>{customer_data.get('total_orders', 0)}</td></tr>
</table>
<h3>Risk Assessment</h3>
<ul>
<li><strong>Churn Risk:</strong> {churn_risk:.2%}</li>
<li><strong>Persona:</strong> {persona}</li>
<li><strong>Recommended Action:</strong> {recommendation}</li>
</ul>
<p>Please take immediate action to retain this customer.</p>
</body>
</html>
    """.strip()
    
    sms_message = f"🚨 At-Risk Customer: {persona}. Churn Risk: {churn_risk:.0%}. Action: {recommendation}"
    
    for recipient in recipients:
        if 'email' in recipient:
            email_result = send_email_notification(
                to_email=recipient['email'],
                subject=subject,
                body=body,
                html_body=html_body
            )
            results.append({"type": "email", "recipient": recipient['email'], **email_result})
        
        if 'phone' in recipient:
            sms_result = send_sms_notification(
                phone_number=recipient['phone'],
                message=sms_message
            )
            results.append({"type": "sms", "recipient": recipient['phone'], **sms_result})
    
    return results

def notify_marketing_team_high_value(
    recipients: List[Dict],
    prediction: Dict,
    customer_data: Dict
) -> List[Dict]:
    """
    Notify marketing team about high-value customers.
    
    Args:
        recipients: List of recipient dictionaries
        prediction: Prediction result
        customer_data: Original customer data
        
    Returns:
        List of notification results
    """
    results = []
    
    clv = prediction.get('clv', 0)
    persona = prediction.get('persona', 'Unknown')
    recommendation = prediction.get('recommendation', 'No recommendation')
    
    subject = f"⭐ High-Value Customer Alert - {persona}"
    
    body = f"""
High-Value Customer Alert

Customer Type: {customer_data.get('customer_type', 'Unknown')}
Category: {customer_data.get('product_category', 'Unknown')}
Total Spent: ₹{customer_data.get('total_spent', 0):,.2f}
Total Orders: {customer_data.get('total_orders', 0)}

Value Assessment:
- Customer Lifetime Value: ₹{clv:,.2f}
- Persona: {persona}
- Recommended Action: {recommendation}

This customer represents significant value. Prioritize engagement.
    """.strip()
    
    sms_message = f"⭐ High-Value Customer: {persona}. CLV: ₹{clv:,.2f}. Action: {recommendation}"
    
    for recipient in recipients:
        if 'email' in recipient:
            email_result = send_email_notification(
                to_email=recipient['email'],
                subject=subject,
                body=body
            )
            results.append({"type": "email", "recipient": recipient['email'], **email_result})
        
        if 'phone' in recipient:
            sms_result = send_sms_notification(
                phone_number=recipient['phone'],
                message=sms_message
            )
            results.append({"type": "sms", "recipient": recipient['phone'], **sms_result})
    
    return results

def notify_marketing_team_batch_summary(
    recipients: List[Dict],
    batch_stats: Dict
) -> List[Dict]:
    """
    Notify marketing team with batch prediction summary.
    
    Args:
        recipients: List of recipient dictionaries
        batch_stats: Batch prediction statistics
        
    Returns:
        List of notification results
    """
    results = []
    
    total_customers = batch_stats.get('total_customers', 0)
    at_risk_count = batch_stats.get('at_risk_count', 0)
    high_value_count = batch_stats.get('high_value_count', 0)
    avg_churn_risk = batch_stats.get('avg_churn_risk', 0)
    
    subject = f"📊 Batch Prediction Summary - {total_customers} Customers Analyzed"
    
    body = f"""
Batch Prediction Summary

Total Customers Analyzed: {total_customers}
At-Risk Customers: {at_risk_count} ({at_risk_count/total_customers*100:.1f}%)
High-Value Customers: {high_value_count} ({high_value_count/total_customers*100:.1f}%)
Average Churn Risk: {avg_churn_risk:.2%}

Key Insights:
- {at_risk_count} customers need immediate retention efforts
- {high_value_count} customers are candidates for upselling
- Overall portfolio health: {"Good" if avg_churn_risk < 0.5 else "Needs Attention"}
    """.strip()
    
    sms_message = f"📊 Batch Summary: {total_customers} customers. {at_risk_count} at-risk, {high_value_count} high-value."
    
    for recipient in recipients:
        if 'email' in recipient:
            email_result = send_email_notification(
                to_email=recipient['email'],
                subject=subject,
                body=body
            )
            results.append({"type": "email", "recipient": recipient['email'], **email_result})
        
        if 'phone' in recipient:
            sms_result = send_sms_notification(
                phone_number=recipient['phone'],
                message=sms_message
            )
            results.append({"type": "sms", "recipient": recipient['phone'], **sms_result})
    
    return results
