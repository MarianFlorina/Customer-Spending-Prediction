import os
import streamlit as st
import requests
from datetime import date, datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# ═══════════════════════════════════════════════════════════════
# Page Config
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="SpendSense — Customer Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
REQUIRED_CSV_COLUMNS = {"total_spent", "total_orders", "last_purchase_date"}
OPTIONAL_CSV_COLUMNS = {"spending_period", "customer_type", "product_category", "discount_sensitivity"}

# ═══════════════════════════════════════════════════════════════
# Professional CSS — Dark Theme with Accent Colors & Mobile
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ─── Global ─── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg-primary: #0a0e17;
    --bg-secondary: #111827;
    --bg-card: #1a2332;
    --bg-card-hover: #1f2b3d;
    --accent: #6366f1;
    --accent-glow: rgba(99, 102, 241, 0.3);
    --green: #10b981;
    --green-bg: rgba(16, 185, 129, 0.1);
    --red: #ef4444;
    --red-bg: rgba(239, 68, 68, 0.1);
    --yellow: #f59e0b;
    --yellow-bg: rgba(245, 158, 11, 0.1);
    --blue: #3b82f6;
    --blue-bg: rgba(59, 130, 246, 0.1);
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --border: #1e293b;
    --border-light: #334155;
    --radius: 12px;
    --radius-sm: 8px;
    --shadow: 0 4px 6px -1px rgba(0,0,0,0.3), 0 2px 4px -2px rgba(0,0,0,0.2);
    --shadow-lg: 0 10px 15px -3px rgba(0,0,0,0.4), 0 4px 6px -4px rgba(0,0,0,0.3);
}

/* ─── Streamlit Overrides ─── */
.stApp {
    background: var(--bg-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--text-primary) !important;
}

/* ─── Cards ─── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    padding: 20px !important;
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow) !important;
    transition: all 0.2s ease !important;
}
[data-testid="stMetric"]:hover {
    border-color: var(--accent) !important;
    box-shadow: 0 0 20px var(--accent-glow) !important;
    transform: translateY(-2px) !important;
}
[data-testid="stMetric"] label {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}
[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
}

/* ─── Headings ─── */
h1, h2, h3, h4, h5, h6 {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}

/* ─── Buttons ─── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent) 0%, #818cf8 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: 12px 28px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.5) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ─── Inputs ─── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div,
.stDateInput > div > div > input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
}

/* ─── Expander ─── */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border) !important;
    font-weight: 500 !important;
    color: var(--text-primary) !important;
}

/* ─── Tabs ─── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px !important;
    background: var(--bg-secondary) !important;
    padding: 4px !important;
    border-radius: var(--radius-sm) !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 6px !important;
    padding: 10px 20px !important;
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    border: none !important;
    transition: all 0.2s ease !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: white !important;
}

/* ─── Success/Warning/Error Boxes ─── */
.success-box {
    background: var(--green-bg);
    border: 1px solid var(--green);
    border-left: 4px solid var(--green);
    border-radius: var(--radius-sm);
    padding: 20px 24px;
    margin: 16px 0;
}
.warning-box {
    background: var(--yellow-bg);
    border: 1px solid var(--yellow);
    border-left: 4px solid var(--yellow);
    border-radius: var(--radius-sm);
    padding: 20px 24px;
    margin: 16px 0;
}
.info-box {
    background: var(--blue-bg);
    border: 1px solid var(--blue);
    border-left: 4px solid var(--blue);
    border-radius: var(--radius-sm);
    padding: 20px 24px;
    margin: 16px 0;
}
.danger-box {
    background: var(--red-bg);
    border: 1px solid var(--red);
    border-left: 4px solid var(--red);
    border-radius: var(--radius-sm);
    padding: 20px 24px;
    margin: 16px 0;
}

/* ─── Prediction Banner ─── */
.prediction-banner {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(16, 185, 129, 0.15) 100%);
    border: 1px solid rgba(99, 102, 241, 0.3);
    border-radius: var(--radius);
    padding: 32px;
    margin: 20px 0;
    text-align: center;
}
.prediction-banner h3 {
    color: var(--accent) !important;
    font-size: 1rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    margin-bottom: 8px !important;
}
.prediction-banner h1 {
    font-size: 2.8rem !important;
    background: linear-gradient(135deg, var(--accent) 0%, var(--green) 100%);
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    margin: 8px 0 !important;
}
.prediction-banner .ci-text {
    color: var(--text-muted) !important;
    font-size: 0.9rem !important;
    margin-top: 4px !important;
}

/* ─── Stat Pills ─── */
.stat-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 0.85rem;
    color: var(--text-secondary);
    margin: 4px;
}
.stat-pill .value {
    color: var(--text-primary);
    font-weight: 600;
}

/* ─── Divider ─── */
.section-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 24px 0;
}

/* ─── Footer ─── */
.footer {
    text-align: center;
    padding: 24px;
    color: var(--text-muted);
    font-size: 0.85rem;
    border-top: 1px solid var(--border);
    margin-top: 40px;
}

/* ─── Responsive ─── */
@media (max-width: 768px) {
    .prediction-banner h1 {
        font-size: 2rem !important;
    }
    [data-testid="stMetric"] {
        padding: 14px !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.2rem !important;
    }
    .stColumns > div {
        min-width: 100% !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# Session State
# ═══════════════════════════════════════════════════════════════
defaults = {
    "predicted": False,
    "predictions_history": [],
    "api_key": os.getenv("API_KEY", ""),
    "current_page": "Single Prediction",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════

def make_api_request(endpoint, method="GET", data=None, params=None):
    """API request with structured error handling."""
    headers = {}
    if st.session_state.api_key:
        headers["X-API-Key"] = st.session_state.api_key
    url = f"{API_BASE_URL}{endpoint}"
    try:
        if method == "GET":
            return requests.get(url, headers=headers, params=params, timeout=15)
        return requests.post(url, headers=headers, json=data, timeout=15)
    except requests.exceptions.ConnectionError:
        st.error("🔌 **Connection failed** — backend is not running. Start it with `uvicorn backend.main:app`.")
        return None
    except requests.exceptions.Timeout:
        st.error("⏱️ **Request timed out** — server may be overloaded.")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"⚠️ **Request error:** {e}")
        return None


def handle_error(response):
    """Display structured error from API response."""
    if response.status_code == 429:
        st.warning("⏱️ **Rate limit exceeded.** Please wait before trying again.")
        return False
    elif response.status_code == 401:
        st.error("🔒 **Authentication failed.** Enter a valid API key in the sidebar.")
        return False
    elif response.status_code == 403:
        st.error("🚫 **Access denied.** Your API key may be invalid.")
        return False
    elif response.status_code >= 500:
        st.error(f"💥 **Server error** ({response.status_code}). Please try again later.")
        return False
    else:
        try:
            detail = response.json().get("detail", response.text[:200])
        except Exception:
            detail = response.text[:200]
        st.error(f"❌ **Error** ({response.status_code}): {detail}")
        return False


def validate_csv(df):
    """Validate CSV columns for batch upload."""
    df_cols = set(c.strip().lower() for c in df.columns)
    missing = REQUIRED_CSV_COLUMNS - df_cols
    if missing:
        st.error(
            f"❌ **Missing required columns:** `{', '.join(sorted(missing))}`\n\n"
            f"Required: `{', '.join(sorted(REQUIRED_CSV_COLUMNS))}`\n\n"
            f"Optional: `{', '.join(sorted(OPTIONAL_CSV_COLUMNS))}`"
        )
        return False
    return True


def render_persona_badge(persona):
    """Render colored persona badge."""
    colors = {
        "At-Risk Customer": ("var(--red)", "var(--red-bg)"),
        "Loyal Customer": ("var(--green)", "var(--green-bg)"),
        "New / Occasional Customer": ("var(--yellow)", "var(--yellow-bg)"),
    }
    color, bg = colors.get(persona, ("var(--text-secondary)", "var(--bg-card)"))
    return f'<span style="background:{bg};border:1px solid {color};color:{color};padding:4px 12px;border-radius:20px;font-size:0.85rem;font-weight:600;">{persona}</span>'


# ═══════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:10px 0 20px 0;">
        <div style="font-size:2.5rem;">📊</div>
        <h2 style="margin:0;background:linear-gradient(135deg,#6366f1,#818cf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">SpendSense</h2>
        <p style="color:var(--text-muted);font-size:0.8rem;margin:4px 0 0 0;">Customer Analytics Platform</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # API Config
    st.markdown("**🔑 API Configuration**")
    api_key = st.text_input("API Key", value=st.session_state.api_key, type="password", placeholder="Enter your API key")
    if api_key != st.session_state.api_key:
        st.session_state.api_key = api_key
        st.rerun()

    api_url = st.text_input("API URL", value=API_BASE_URL, placeholder="http://localhost:8000")

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Navigation
    st.markdown("**📚 Navigation**")
    pages = {
        "🔮 Single Prediction": "Single Prediction",
        "📊 Batch Prediction": "Batch Prediction",
        "📈 Analytics": "Analytics",
        "📋 History": "History",
        "⚙️ Settings": "Settings",
    }
    selected = st.radio(
        "nav",
        list(pages.keys()),
        index=list(pages.keys()).index(f"{'🔮 Single Prediction' if st.session_state.current_page == 'Single Prediction' else '📊 Batch Prediction' if st.session_state.current_page == 'Batch Prediction' else '📈 Analytics' if st.session_state.current_page == 'Analytics' else '📋 History' if st.session_state.current_page == 'History' else '⚙️ Settings'}"),
        label_visibility="collapsed",
    )
    st.session_state.current_page = pages[selected]

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # Status
    health = make_api_request("/health")
    if health and health.status_code == 200:
        hd = health.json()
        model_ver = hd.get("model_version", "?")
        st.markdown(f"""
        <div style="background:var(--green-bg);border:1px solid var(--green);border-radius:var(--radius-sm);padding:12px;text-align:center;">
            <div style="font-size:1.2rem;">✅</div>
            <div style="color:var(--green);font-weight:600;font-size:0.85rem;">Connected</div>
            <div style="color:var(--text-muted);font-size:0.75rem;margin-top:4px;">Model: {model_ver}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:var(--red-bg);border:1px solid var(--red);border-radius:var(--radius-sm);padding:12px;text-align:center;">
            <div style="font-size:1.2rem;">❌</div>
            <div style="color:var(--red);font-weight:600;font-size:0.85rem;">Offline</div>
            <div style="color:var(--text-muted);font-size:0.75rem;margin-top:4px;">Check backend</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# Main Content
# ═══════════════════════════════════════════════════════════════

page = st.session_state.current_page

# ─── Header ───
st.markdown("""
<div style="margin-bottom:24px;">
    <h1 style="margin:0;font-size:2rem;">Customer Spending Prediction</h1>
    <p style="color:var(--text-muted);margin:4px 0 0 0;font-size:0.95rem;">ML-powered insights for smarter marketing decisions</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SINGLE PREDICTION
# ═══════════════════════════════════════════════════════════════
if page == "Single Prediction":
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            spending_period = st.selectbox("📅 Spending Period", ["Last 30 Days", "Last 6 Months", "Last 12 Months", "Lifetime"])
            total_spent = st.number_input("💰 Total Spending (₹)", 0.0, 100000.0, 5000.0, step=100.0)
            total_orders = st.number_input("🛒 Total Orders", 1, 100, 3)
            last_purchase_date = st.date_input("📆 Last Purchase Date", date(2024, 6, 1))
        with col2:
            customer_type = st.selectbox("👤 Customer Type", ["New", "Returning", "Loyal"])
            product_category = st.selectbox("📦 Primary Category", ["Electronics", "Fashion", "Grocery", "Home", "Mixed"])
            discount_sensitivity = st.selectbox("🏷️ Discount Sensitivity", ["Low", "Medium", "High"])

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    if st.button("🔮  Predict Future Spending", use_container_width=True):
        with st.spinner("🤖 Analyzing customer data..."):
            payload = {
                "total_spent": total_spent, "total_orders": total_orders,
                "last_purchase_date": str(last_purchase_date),
                "spending_period": spending_period, "customer_type": customer_type,
                "product_category": product_category, "discount_sensitivity": discount_sensitivity,
            }
            response = make_api_request("/v1/predict/explain", method="POST", data=payload)

            if response and response.status_code == 200:
                data = response.json()
                st.session_state.predictions_history.append({"timestamp": datetime.now().isoformat(), "result": data})
                explanation = data.get("explanation", {})

                # Prediction Banner
                ci = data.get("confidence_interval", {})
                ci_html = f'<div class="ci-text">95% Confidence: ₹{ci["lower"]:,.2f} — ₹{ci["upper"]:,.2f}</div>' if ci else ""

                st.markdown(f"""
                <div class="prediction-banner">
                    <h3>💰 Predicted Future Spending</h3>
                    <h1>₹{data['predicted_future_spending']:,.2f}</h1>
                    {ci_html}
                </div>
                """, unsafe_allow_html=True)

                metrics = {
                    "persona": data.get("persona", "Unknown"),
                    "average_order_value": total_spent / total_orders,
                    "churn_risk": data.get("churn_risk", 0),
                    "CLV": 0,
                }

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric("👤 Persona", metrics["persona"])
                with c2:
                    st.metric("💰 Avg Order Value", f"₹{metrics['average_order_value']:,.2f}")
                with c3:
                    delta_color = "inverse" if metrics["churn_risk"] > 0.6 else "normal"
                    st.metric("⚠️ Churn Risk", f"{metrics['churn_risk']:.0%}", delta="High" if metrics["churn_risk"] > 0.6 else "Low", delta_color=delta_color)
                with c4:
                    st.metric("📈 Model", f"v{data.get('model_version', '?')}")

                st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

                # ─── Explainability Section ───
                if explanation:
                    st.header("🔍 Why This Prediction?")

                    # TL;DR
                    tldr = explanation.get("summary", {}).get("tldr", "")
                    if tldr:
                        st.markdown(f"""
                        <div class="info-box">
                            <strong>📌 TL;DR:</strong> {tldr}
                        </div>
                        """, unsafe_allow_html=True)

                    # Feature Contributions Waterfall Chart
                    features = explanation.get("features", [])
                    if features:
                        st.subheader("Feature Contributions")

                        c1, c2 = st.columns([3, 2])
                        with c1:
                            feat_names = [f["name"].replace("_", " ").title() for f in features]
                            feat_contribs = [f["contribution"] for f in features]
                            colors = ["#10b981" if c > 0 else "#ef4444" for c in feat_contribs]

                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                y=feat_names[::-1],
                                x=feat_contribs[::-1],
                                orientation="h",
                                marker_color=colors[::-1],
                                text=[f"{c:+,.0f}" for c in feat_contribs[::-1]],
                                textposition="auto",
                                textfont=dict(color="white", size=11),
                            ))
                            fig.update_layout(
                                template="plotly_dark", height=350,
                                margin=dict(t=10, b=10, l=120, r=20),
                                title=dict(text="Feature Impact on Prediction", font=dict(size=14)),
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                xaxis=dict(title="Contribution (₹)", gridcolor="rgba(255,255,255,0.05)"),
                                yaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        with c2:
                            st.markdown("**Feature Values**")
                            for f in features[:6]:
                                direction_icon = "🟢" if f["direction"] == "positive" else "🔴" if f["direction"] == "negative" else "⚪"
                                st.markdown(
                                    f"{direction_icon} **{f['name'].replace('_', ' ').title()}**: {f['value']} "
                                    f"*(impact: {f['contribution']:+,.0f})*"
                                )

                    # Natural Language Summary
                    paragraphs = explanation.get("summary", {}).get("paragraphs", [])
                    if paragraphs:
                        st.subheader("📋 Plain-Language Summary")
                        for p in paragraphs:
                            st.markdown(p)

                    # Top Drivers
                    top_pos = explanation.get("top_positive_drivers", [])
                    top_neg = explanation.get("top_negative_drivers", [])

                    if top_pos or top_neg:
                        st.subheader("🎯 Key Drivers")
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**🟢 What's pushing spending UP:**")
                            for d in top_pos:
                                st.markdown(f"- {d.get('explanation', d['feature'])}")
                        with c2:
                            st.markdown("**🔴 What's pulling spending DOWN:**")
                            for d in top_neg:
                                st.markdown(f"- {d.get('explanation', d['feature'])}")

                    # Actionable Insights
                    insights = explanation.get("insights", [])
                    if insights:
                        st.subheader("🚀 Actionable Insights")
                        for insight in insights:
                            priority = insight.get("priority", "low")
                            priority_color = {"high": "var(--red)", "medium": "var(--yellow)", "low": "var(--text-muted)"}.get(priority, "var(--text-muted)")
                            icon = {"retention": "🛡️", "upsell": "📈", "reactivation": "🔄", "onboarding": "🎉", "high_value": "👑", "general": "💡"}.get(insight.get("type", ""), "💡")

                            st.markdown(f"""
                            <div style="background:var(--bg-card);border:1px solid var(--border);border-left:4px solid {priority_color};border-radius:var(--radius-sm);padding:16px 20px;margin:12px 0;">
                                <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
                                    <span style="font-size:1.2rem;">{icon}</span>
                                    <strong style="color:var(--text-primary);">{insight['title']}</strong>
                                    <span style="background:{priority_color};color:white;padding:2px 8px;border-radius:10px;font-size:0.7rem;text-transform:uppercase;font-weight:600;">{priority}</span>
                                </div>
                                <p style="color:var(--text-secondary);margin:0 0 8px 0;font-size:0.9rem;">{insight['description']}</p>
                            """, unsafe_allow_html=True)
                            if insight.get("actions"):
                                for action in insight["actions"]:
                                    st.markdown(f"  - {action}")
                            st.markdown("</div>", unsafe_allow_html=True)

                    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

                # Charts
                c1, c2 = st.columns(2)
                with c1:
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        x=["Past Spending", "Predicted Spending"],
                        y=[total_spent, data["predicted_future_spending"]],
                        marker_color=["#6366f1", "#10b981"],
                        text=[f"₹{total_spent:,.0f}", f"₹{data['predicted_future_spending']:,.0f}"],
                        textposition="auto", textfont=dict(color="white", size=14),
                    ))
                    fig.update_layout(
                        template="plotly_dark", height=380, margin=dict(t=40, b=20, l=20, r=20),
                        title=dict(text="Spending Comparison", font=dict(size=16)),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        yaxis=dict(title="Amount (₹)", gridcolor="rgba(255,255,255,0.05)"),
                        xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with c2:
                    fig = go.Figure()
                    fig.add_trace(go.Indicator(
                        mode="gauge+number+delta", value=metrics["churn_risk"] * 100,
                        title={"text": "Churn Risk", "font": {"size": 16}},
                        delta={"reference": 50, "increasing": {"color": "#ef4444"}, "decreasing": {"color": "#10b981"}},
                        gauge={
                            "axis": {"range": [0, 100], "tickcolor": "white"},
                            "bar": {"color": "#ef4444" if metrics["churn_risk"] > 0.6 else "#10b981"},
                            "steps": [
                                {"range": [0, 30], "color": "rgba(16,185,129,0.15)"},
                                {"range": [30, 60], "color": "rgba(245,158,11,0.15)"},
                                {"range": [60, 100], "color": "rgba(239,68,68,0.15)"},
                            ],
                            "threshold": {"line": {"color": "white", "width": 3}, "thickness": 0.8, "value": 70},
                        },
                    ))
                    fig.update_layout(
                        height=380, margin=dict(t=40, b=20, l=30, r=30),
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="white"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Recommendation
                st.markdown(f"""
                <div class="info-box">
                    <strong>💡 Recommendation:</strong> {data['recommendation']}
                </div>
                """, unsafe_allow_html=True)

            elif response:
                handle_error(response)


# ═══════════════════════════════════════════════════════════════
# BATCH PREDICTION
# ═══════════════════════════════════════════════════════════════
elif page == "Batch Prediction":
    tab1, tab2 = st.tabs(["📁 Upload CSV", "✏️ Manual Entry"])

    with tab1:
        uploaded = st.file_uploader("Upload CSV with customer data", type=["csv"], help="Required columns: total_spent, total_orders, last_purchase_date")
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                st.dataframe(df.head(10), use_container_width=True)

                if not validate_csv(df):
                    st.stop()

                if st.button("🔄  Process Batch", use_container_width=True):
                    with st.spinner("🤖 Processing batch predictions..."):
                        customers = []
                        for _, row in df.iterrows():
                            col_map = {c.strip().lower(): c for c in df.columns}
                            customers.append({
                                "total_spent": float(row[col_map.get("total_spent", "total_spent")]),
                                "total_orders": int(row[col_map.get("total_orders", "total_orders")]),
                                "last_purchase_date": str(row[col_map.get("last_purchase_date", "last_purchase_date")]),
                                "spending_period": row.get(col_map.get("spending_period", "spending_period"), "Last 30 Days"),
                                "customer_type": row.get(col_map.get("customer_type", "customer_type"), "New"),
                                "product_category": row.get(col_map.get("product_category", "product_category"), "Mixed"),
                                "discount_sensitivity": row.get(col_map.get("discount_sensitivity", "discount_sensitivity"), "Medium"),
                            })

                        response = make_api_request("/v1/predict/batch", method="POST", data={"customers": customers[:100]})
                        if response and response.status_code == 200:
                            bd = response.json()
                            summary = bd["summary"]

                            st.markdown(f"""
                            <div class="success-box">
                                <strong>✅ Processed {bd['total_customers']} customers</strong> successfully
                            </div>
                            """, unsafe_allow_html=True)

                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Total Customers", summary["total_customers"])
                            c2.metric("Avg Predicted", f"₹{summary['avg_predicted_spending']:,.0f}")
                            c3.metric("At-Risk", summary["at_risk_count"])
                            c4.metric("High-Value", summary["high_value_count"])

                            results = []
                            for p in bd["predictions"]:
                                ci = p.get("confidence_interval", {})
                                results.append({
                                    "Predicted ₹": f"₹{p['predicted_future_spending']:,.0f}",
                                    "CI Low": f"₹{ci['lower']:,.0f}" if ci else "—",
                                    "CI High": f"₹{ci['upper']:,.0f}" if ci else "—",
                                    "Persona": p["derived_metrics"]["persona"],
                                    "Churn": f"{p['derived_metrics']['churn_risk']:.0%}",
                                    "CLV": f"₹{p['derived_metrics']['CLV']:,.0f}",
                                })
                            st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)

                            csv_out = pd.DataFrame(results).to_csv(index=False)
                            st.download_button("📥 Download Results", csv_out, "batch_predictions.csv", "text/csv")
                        elif response:
                            handle_error(response)
            except pd.errors.EmptyDataError:
                st.error("❌ CSV file is empty.")
            except pd.errors.ParserError:
                st.error("❌ Could not parse CSV. Check the format.")
            except Exception as e:
                st.error(f"❌ Error: {e}")

    with tab2:
        num = st.number_input("Number of customers", 1, 10, 3)
        customers = []
        for i in range(num):
            with st.expander(f"👤 Customer {i + 1}", expanded=(i == 0)):
                c1, c2 = st.columns(2)
                with c1:
                    customers.append({
                        "total_spent": st.number_input("Total Spending (₹)", 0.0, 100000.0, 5000.0, key=f"s{i}"),
                        "total_orders": st.number_input("Total Orders", 1, 100, 3, key=f"o{i}"),
                        "last_purchase_date": str(st.date_input("Last Purchase", date(2024, 6, 1), key=f"d{i}")),
                    })
                with c2:
                    customers[-1].update({
                        "spending_period": "Last 30 Days",
                        "customer_type": st.selectbox("Type", ["New", "Returning", "Loyal"], key=f"t{i}"),
                        "product_category": st.selectbox("Category", ["Electronics", "Fashion", "Grocery", "Home", "Mixed"], key=f"c{i}"),
                        "discount_sensitivity": st.selectbox("Discount", ["Low", "Medium", "High"], key=f"dc{i}"),
                    })

        if st.button("🔮  Process Manual Batch", use_container_width=True):
            with st.spinner("Processing..."):
                response = make_api_request("/v1/predict/batch", method="POST", data={"customers": customers})
                if response and response.status_code == 200:
                    bd = response.json()
                    st.success(f"✅ Processed {bd['total_customers']} customers")
                    st.json(bd["summary"])
                elif response:
                    handle_error(response)


# ═══════════════════════════════════════════════════════════════
# ANALYTICS
# ═══════════════════════════════════════════════════════════════
elif page == "Analytics":
    tab1, tab2, tab3 = st.tabs(["📈 Statistics", "👥 Cohorts", "🎯 RFM Segmentation"])

    with tab1:
        resp = make_api_request("/v1/analytics/stats")
        if resp and resp.status_code == 200:
            s = resp.json()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Predictions", f"{s.get('total_predictions', 0):,}")
            c2.metric("Avg Spending", f"₹{s.get('avg_predicted_spending', 0):,.0f}")
            c3.metric("Avg Churn Risk", f"{s.get('avg_churn_risk', 0):.0%}")
            c4.metric("Batch Jobs", s.get("total_batch_predictions", 0))

            c1, c2 = st.columns(2)
            with c1:
                if s.get("by_persona"):
                    fig = px.pie(values=list(s["by_persona"].values()), names=list(s["by_persona"].keys()),
                                 template="plotly_dark", hole=0.4,
                                 color_discrete_sequence=["#6366f1", "#10b981", "#f59e0b"])
                    fig.update_layout(margin=dict(t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", height=350)
                    st.plotly_chart(fig, use_container_width=True)
            with c2:
                if s.get("by_customer_type"):
                    fig = px.bar(x=list(s["by_customer_type"].keys()), y=list(s["by_customer_type"].values()),
                                 template="plotly_dark", color=list(s["by_customer_type"].keys()),
                                 color_discrete_sequence=["#6366f1", "#818cf8", "#a5b4fc"])
                    fig.update_layout(margin=dict(t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", height=350, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
        elif resp:
            handle_error(resp)

    with tab2:
        resp = make_api_request("/v1/analytics/cohorts")
        if resp and resp.status_code == 200:
            cohorts = resp.json()
            if cohorts:
                df = pd.DataFrame(cohorts)
                c1, c2 = st.columns(2)
                with c1:
                    fig = px.bar(df, x="spending_period", y="customer_count", template="plotly_dark",
                                 color_discrete_sequence=["#6366f1"])
                    fig.update_layout(margin=dict(t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", height=350)
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    fig = px.bar(df, x="spending_period", y="avg_predicted", template="plotly_dark",
                                 color_discrete_sequence=["#10b981"])
                    fig.update_layout(margin=dict(t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", height=350)
                    st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("📊 No cohort data yet. Make some predictions first!")
        elif resp:
            handle_error(resp)

    with tab3:
        resp = make_api_request("/v1/analytics/rfm")
        if resp and resp.status_code == 200:
            rfm = resp.json()
            if rfm:
                st.dataframe(pd.DataFrame(rfm), use_container_width=True, hide_index=True)
            else:
                st.info("📊 No RFM data yet. Make some predictions first!")
        elif resp:
            handle_error(resp)


# ═══════════════════════════════════════════════════════════════
# HISTORY
# ═══════════════════════════════════════════════════════════════
elif page == "History":
    page_num = st.number_input("Page", 1, 100, 1)
    resp = make_api_request("/v1/history", params={"page": page_num, "page_size": 20})
    if resp and resp.status_code == 200:
        h = resp.json()
        total = h.get("total_count", 0)
        st.info(f"📋 Showing page {page_num} — **{total}** total predictions in database")

        if h["predictions"]:
            df = pd.DataFrame(h["predictions"])
            display = [c for c in ["timestamp", "total_spent", "total_orders", "predicted_spending",
                                    "persona", "churn_risk", "model_version"] if c in df.columns]
            st.dataframe(df[display], use_container_width=True, hide_index=True)

            csv = df.to_csv(index=False)
            st.download_button("📥 Export to CSV", csv, f"predictions_p{page_num}.csv", "text/csv")
        else:
            st.info("No predictions on this page.")
    elif resp:
        handle_error(resp)


# ═══════════════════════════════════════════════════════════════
# SETTINGS
# ═══════════════════════════════════════════════════════════════
elif page == "Settings":
    tab1, tab2 = st.tabs(["🔑 API Settings", "🧹 Maintenance"])

    with tab1:
        st.subheader("API Configuration")
        st.code(f"API URL: {API_BASE_URL}\nAPI Key: {'*' * 8 if st.session_state.api_key else '(not set)'}")

        st.markdown("---")
        st.subheader("Generate New API Key")
        if st.button("🔑  Generate Key"):
            resp = make_api_request("/v1/api-keys/generate")
            if resp and resp.status_code == 200:
                new_key = resp.json()["api_key"]
                st.code(new_key)
                st.warning("🔑 Save this key securely — it won't be shown again.")
            elif resp:
                handle_error(resp)

        st.markdown("---")
        st.subheader("Model Info")
        resp = make_api_request("/v1/model/config")
        if resp and resp.status_code == 200:
            st.json(resp.json())

    with tab2:
        st.subheader("Data Maintenance")
        days = st.number_input("Retention Period (days)", 30, 3650, 365)
        if st.button("🧹  Clean Old Predictions"):
            resp = make_api_request("/v1/maintenance/cleanup", method="POST", params={"days": days})
            if resp and resp.status_code == 200:
                deleted = resp.json().get("deleted", 0)
                st.success(f"✅ Deleted **{deleted}** predictions older than {days} days")
            elif resp:
                handle_error(resp)


# ─── Footer ───
st.markdown("""
<div class="footer">
    <strong>SpendSense</strong> v2.1 · Built with FastAPI + Streamlit + Scikit-learn<br>
    <span style="font-size:0.75rem;">ML-powered customer spending prediction & analytics platform</span>
</div>
""", unsafe_allow_html=True)
