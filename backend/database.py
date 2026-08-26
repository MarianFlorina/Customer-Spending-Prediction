import os
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

DATABASE_PATH = os.getenv("DATABASE_PATH", "data/predictions.db")
DATA_RETENTION_DAYS = int(os.getenv("DATA_RETENTION_DAYS", "365"))
MIGRATION_VERSION = 3  # Bump when schema changes


def _get_connection() -> sqlite3.Connection:
    """Get a database connection with proper settings."""
    Path(DATABASE_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Initialize the database with schema and run migrations."""
    conn = _get_connection()
    cursor = conn.cursor()

    # ─── Schema v1: Base tables ───
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            total_spent REAL NOT NULL,
            total_orders INTEGER NOT NULL,
            last_purchase_date DATE,
            spending_period TEXT,
            customer_type TEXT,
            product_category TEXT,
            discount_sensitivity TEXT,
            predicted_spending REAL,
            avg_order_value REAL,
            recency_days INTEGER,
            clv REAL,
            churn_risk REAL,
            persona TEXT,
            recommendation TEXT,
            request_source TEXT DEFAULT 'api',
            model_version TEXT DEFAULT 'unknown',
            confidence_lower REAL,
            confidence_upper REAL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS batch_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            batch_size INTEGER,
            request_source TEXT DEFAULT 'api',
            status TEXT DEFAULT 'completed'
        )
    """)

    # ─── Schema v2: Migration table ───
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # ─── Indexes for query performance ───
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_persona ON predictions(persona)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_customer_type ON predictions(customer_type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_category ON predictions(product_category)")

    # Run migrations
    _run_migrations(conn)

    conn.commit()
    conn.close()
    logger.info("Database initialized successfully")


def _run_migrations(conn: sqlite3.Connection):
    """Run pending database migrations."""
    cursor = conn.cursor()

    cursor.execute("SELECT version FROM schema_migrations ORDER BY version DESC LIMIT 1")
    row = cursor.fetchone()
    current_version = row[0] if row else 0

    if current_version < 1:
        # Migration 1: Already applied in CREATE TABLE above
        cursor.execute("INSERT OR IGNORE INTO schema_migrations (version) VALUES (1)")
        logger.info("Applied migration v1")

    if current_version < 2:
        # Migration 2: Add model_version and confidence columns if missing
        try:
            cursor.execute("ALTER TABLE predictions ADD COLUMN model_version TEXT DEFAULT 'unknown'")
        except sqlite3.OperationalError:
            pass  # Column already exists
        try:
            cursor.execute("ALTER TABLE predictions ADD COLUMN confidence_lower REAL")
        except sqlite3.OperationalError:
            pass
        try:
            cursor.execute("ALTER TABLE predictions ADD COLUMN confidence_upper REAL")
        except sqlite3.OperationalError:
            pass
        cursor.execute("INSERT OR IGNORE INTO schema_migrations (version) VALUES (2)")
        logger.info("Applied migration v2")

    if current_version < 3:
        # Migration 3: Add request_id for idempotency
        try:
            cursor.execute("ALTER TABLE predictions ADD COLUMN request_id TEXT")
        except sqlite3.OperationalError:
            pass
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_request_id ON predictions(request_id)")
        cursor.execute("INSERT OR IGNORE INTO schema_migrations (version) VALUES (3)")
        logger.info("Applied migration v3")


def save_prediction(input_data: dict, result: dict, source: str = "api", request_id: str = None) -> int:
    """Save a single prediction to the database."""
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions (
            total_spent, total_orders, last_purchase_date, spending_period,
            customer_type, product_category, discount_sensitivity,
            predicted_spending, avg_order_value, recency_days, clv,
            churn_risk, persona, recommendation, request_source,
            model_version, confidence_lower, confidence_upper, request_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        input_data["total_spent"], input_data["total_orders"],
        input_data["last_purchase_date"], input_data["spending_period"],
        input_data["customer_type"], input_data["product_category"],
        input_data["discount_sensitivity"], result["prediction"],
        result["avg_order_value"], result["recency"], result["CLV"],
        result["churn_risk"], result["persona"], result["recommendation"],
        source, result.get("model_version", "unknown"),
        result.get("confidence_interval", {}).get("lower"),
        result.get("confidence_interval", {}).get("upper"),
        request_id,
    ))

    prediction_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return prediction_id


def save_batch_prediction(batch_size: int, source: str = "api") -> int:
    """Save a batch prediction record."""
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO batch_predictions (batch_size, request_source)
        VALUES (?, ?)
    """, (batch_size, source))

    batch_id = cursor.lastrowid
    conn.commit()
    conn.close()

    return batch_id


def get_prediction_history(
    limit: int = 100, offset: int = 0
) -> Tuple[List[Dict], int]:
    """Retrieve prediction history with accurate total count."""
    conn = _get_connection()
    cursor = conn.cursor()

    # Total count
    cursor.execute("SELECT COUNT(*) FROM predictions")
    total_count = cursor.fetchone()[0]

    # Paginated results
    cursor.execute("""
        SELECT * FROM predictions
        ORDER BY timestamp DESC
        LIMIT ? OFFSET ?
    """, (limit, offset))

    rows = cursor.fetchall()
    conn.close()

    return [dict(row) for row in rows], total_count


def cleanup_old_predictions(days: int = None) -> int:
    """Remove predictions older than the retention period."""
    if days is None:
        days = DATA_RETENTION_DAYS

    conn = _get_connection()
    cursor = conn.cursor()

    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    cursor.execute("DELETE FROM predictions WHERE timestamp < ?", (cutoff,))
    deleted = cursor.rowcount
    conn.commit()
    conn.close()

    if deleted > 0:
        logger.info(f"Cleaned up {deleted} predictions older than {days} days")
    return deleted


def get_prediction_stats() -> Dict:
    """Get aggregate statistics about predictions."""
    conn = _get_connection()
    cursor = conn.cursor()

    stats = {}

    cursor.execute("SELECT COUNT(*) FROM predictions")
    stats["total_predictions"] = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(predicted_spending) FROM predictions")
    result = cursor.fetchone()[0]
    stats["avg_predicted_spending"] = round(result, 2) if result else 0

    cursor.execute("SELECT SUM(predicted_spending) FROM predictions")
    result = cursor.fetchone()[0]
    stats["total_predicted_spending"] = round(result, 2) if result else 0

    cursor.execute("""
        SELECT persona, COUNT(*) as count FROM predictions GROUP BY persona
    """)
    stats["by_persona"] = {row[0]: row[1] for row in cursor.fetchall()}

    cursor.execute("""
        SELECT customer_type, COUNT(*) as count FROM predictions GROUP BY customer_type
    """)
    stats["by_customer_type"] = {row[0]: row[1] for row in cursor.fetchall()}

    cursor.execute("""
        SELECT product_category, COUNT(*) as count FROM predictions GROUP BY product_category
    """)
    stats["by_category"] = {row[0]: row[1] for row in cursor.fetchall()}

    cursor.execute("SELECT AVG(churn_risk) FROM predictions")
    result = cursor.fetchone()[0]
    stats["avg_churn_risk"] = round(result, 2) if result else 0

    cursor.execute("SELECT AVG(clv) FROM predictions")
    result = cursor.fetchone()[0]
    stats["avg_clv"] = round(result, 2) if result else 0

    cursor.execute("SELECT COUNT(*) FROM batch_predictions")
    stats["total_batch_predictions"] = cursor.fetchone()[0]

    conn.close()
    return stats


def get_cohort_data() -> List[Dict]:
    """Get cohort analysis data grouped by spending period."""
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            spending_period,
            COUNT(*) as customer_count,
            AVG(predicted_spending) as avg_predicted,
            AVG(churn_risk) as avg_churn_risk,
            AVG(clv) as avg_clv,
            AVG(avg_order_value) as avg_order_value
        FROM predictions
        GROUP BY spending_period
        ORDER BY customer_count DESC
    """)

    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_rfm_segmentation() -> List[Dict]:
    """Get RFM (Recency, Frequency, Monetary) segmentation data."""
    conn = _get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            CASE
                WHEN recency_days <= 30 THEN 'Recent'
                WHEN recency_days <= 90 THEN 'Active'
                WHEN recency_days <= 180 THEN 'At-Risk'
                ELSE 'Churned'
            END as recency_segment,
            CASE
                WHEN total_orders >= 10 THEN 'Champion'
                WHEN total_orders >= 5 THEN 'Loyal'
                WHEN total_orders >= 2 THEN 'Potential'
                ELSE 'New'
            END as frequency_segment,
            CASE
                WHEN total_spent >= 10000 THEN 'High Value'
                WHEN total_spent >= 5000 THEN 'Medium Value'
                WHEN total_spent >= 1000 THEN 'Low Value'
                ELSE 'Minimal'
            END as monetary_segment,
            COUNT(*) as customer_count,
            AVG(predicted_spending) as avg_predicted_spending
        FROM predictions
        GROUP BY recency_segment, frequency_segment, monetary_segment
        ORDER BY customer_count DESC
    """)

    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


# Initialize database on import
init_db()
