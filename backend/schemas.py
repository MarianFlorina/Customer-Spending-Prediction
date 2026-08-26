from pydantic import BaseModel, Field
from datetime import date
from typing import List, Optional, Dict
from enum import Enum

class CustomerType(str, Enum):
    NEW = "New"
    RETURNING = "Returning"
    LOYAL = "Loyal"

class DiscountSensitivity(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"

class SpendingPeriod(str, Enum):
    LAST_30_DAYS = "Last 30 Days"
    LAST_6_MONTHS = "Last 6 Months"
    LAST_12_MONTHS = "Last 12 Months"
    LIFETIME = "Lifetime"

class ProductCategory(str, Enum):
    ELECTRONICS = "Electronics"
    FASHION = "Fashion"
    GROCERY = "Grocery"
    HOME = "Home"
    MIXED = "Mixed"

class CustomerInput(BaseModel):
    total_spent: float = Field(..., ge=0, description="Total amount spent by customer")
    total_orders: int = Field(..., ge=1, description="Total number of orders")
    last_purchase_date: date = Field(..., description="Date of last purchase")
    spending_period: SpendingPeriod = Field(..., description="Spending period timeframe")
    customer_type: CustomerType = Field(..., description="Customer loyalty type")
    product_category: ProductCategory = Field(..., description="Primary product category")
    discount_sensitivity: DiscountSensitivity = Field(..., description="Sensitivity to discounts")

class BatchCustomerInput(BaseModel):
    customers: List[CustomerInput] = Field(..., min_length=1, max_length=100, description="List of customers to predict")
    source: Optional[str] = Field(default="api", description="Source of the batch request")

class WebhookConfig(BaseModel):
    url: str = Field(..., description="Webhook URL to send notifications")
    events: List[str] = Field(default=["customer.at_risk", "customer.high_value"], description="Events to trigger webhook")
    headers: Optional[Dict[str, str]] = Field(default=None, description="Optional custom headers")

class NotificationRecipient(BaseModel):
    email: Optional[str] = Field(default=None, description="Email address for notifications")
    phone: Optional[str] = Field(default=None, description="Phone number for SMS notifications")
    name: Optional[str] = Field(default=None, description="Recipient name")

class NotificationConfig(BaseModel):
    recipients: List[NotificationRecipient] = Field(..., min_length=1, description="List of notification recipients")
    notify_at_risk: bool = Field(default=True, description="Send notifications for at-risk customers")
    notify_high_value: bool = Field(default=True, description="Send notifications for high-value customers")
    notify_batch_summary: bool = Field(default=True, description="Send batch summary notifications")

class ConfidenceInterval(BaseModel):
    lower: float
    upper: float
    alpha: float = 0.05

class PredictionResponse(BaseModel):
    predicted_future_spending: float
    derived_metrics: Dict
    recommendation: str
    currency: str = "INR"
    prediction_id: Optional[int] = None
    confidence_interval: Optional[ConfidenceInterval] = None
    model_version: Optional[str] = None

class BatchPredictionResponse(BaseModel):
    batch_id: int
    total_customers: int
    predictions: List[PredictionResponse]
    summary: Dict

class AnomalyResponse(BaseModel):
    is_anomalous: bool
    anomalies: List[Dict]
    risk_level: str
    recommendations: List[str]

class HistoryResponse(BaseModel):
    predictions: List[Dict]
    total_count: int
    page: int
    page_size: int
