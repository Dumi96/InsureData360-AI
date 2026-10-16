from __future__ import annotations
from live_prediction import render_live_prediction

import json
from datetime import datetime
import os
import re
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg
import streamlit as st
from dotenv import load_dotenv

# ============================================================
# 1. APPLICATION CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

ANOMALY_FILE = PROCESSED_DIR / "claim_anomaly_results.csv"
SEVERITY_METRICS_FILE = MODELS_DIR / "claim_severity_metrics.json"

load_dotenv(PROJECT_ROOT / ".env")

st.set_page_config(
    page_title="InsureData360-AI | Insurance Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)



# ============================================================
# 2. PROFESSIONAL DASHBOARD STYLING
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --navy: #102A43;
        --blue: #1D4ED8;
        --blue-dark: #1E40AF;
        --blue-light: #EFF6FF;
        --text: #172033;
        --muted: #64748B;
        --border: #DCE5F0;
        --surface: #FFFFFF;
        --background: #F4F7FB;
        --tab-inactive: #E8EEF6;
        --tab-active: #1D4ED8;
    }

    /* --------------------------------------------------------
       GLOBAL APPLICATION
    -------------------------------------------------------- */

    /* Animated full-app live wallpaper. Keep it on the app background
       itself so it can never sit on top of Streamlit's dashboard widgets. */
    .stApp {
        background-color: #EEF4FF !important;
        background-image:
            radial-gradient(ellipse at 12% 18%, rgba(56, 189, 248, .25) 0%, transparent 34%),
            radial-gradient(ellipse at 84% 24%, rgba(129, 140, 248, .23) 0%, transparent 32%),
            radial-gradient(ellipse at 56% 88%, rgba(45, 212, 191, .22) 0%, transparent 36%),
            linear-gradient(-45deg, #EAF2FF 0%, #F4F7FF 24%, #E5FAF7 50%, #F1ECFF 76%, #EAF2FF 100%);
        background-size: 160% 160%, 160% 160%, 160% 160%, 420% 420%;
        background-position: 0% 0%, 100% 10%, 50% 100%, 0% 50%;
        animation: wallpaperShift 28s ease-in-out infinite;
        color: var(--text);
    }

    /* Streamlit's content containers remain transparent, not hidden/covered. */
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {
        background: transparent !important;
    }

    [data-testid="stHeader"] {
        background: rgba(244, 247, 251, 0.68);
        backdrop-filter: blur(12px);
    }

    /* --------------------------------------------------------
       SIDEBAR
    -------------------------------------------------------- */

    [data-testid="stSidebar"] {
        background: linear-gradient(155deg, #0B1730, #102A43 42%, #163C63 72%, #0B1730);
        background-size: 220% 220%;
        animation: sidebarFlow 24s ease-in-out infinite;
        border-right: 1px solid rgba(191, 219, 254, .16);
    }

    [data-testid="stSidebar"] * {
        color: #F8FAFC;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.18);
    }

    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #FFFFFF;
    }

    /* --------------------------------------------------------
       HERO BANNER
    -------------------------------------------------------- */

    .hero {
        position: relative;
        overflow: hidden;
        padding: 28px 30px;
        border-radius: 18px;
        margin-bottom: 24px;
        color: white;
        background: linear-gradient(
            115deg,
            #102A43 0%,
            #174EA6 48%,
            #2563EB 100%
        );
        background-size: 180% 180%;
        animation: heroGradient 12s ease infinite;
        box-shadow: 0 8px 24px rgba(16, 42, 67, 0.15);
    }

    .hero h1 {
        font-size: clamp(25px, 3vw, 36px);
        line-height: 1.2;
        margin: 0 0 10px 0;
        color: white;
        font-weight: 800;
    }

    .hero p {
        margin: 0;
        color: #E6F0FF;
        font-size: 15px;
        max-width: 850px;
        line-height: 1.6;
    }

    .hero-topline {
        text-transform: uppercase;
        font-size: 11px;
        letter-spacing: 2px;
        font-weight: 800;
        color: #BFDBFE;
        margin-bottom: 12px;
    }

    .hero-icon {
        position: absolute;
        right: 35px;
        top: 24px;
        font-size: 70px;
        opacity: 0.22;
        animation: shieldFloat 4s ease-in-out infinite;
    }

    /* --------------------------------------------------------
       SECTION LABELS
    -------------------------------------------------------- */

    .section-label {
        font-size: 12px;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 1.1px;
        color: #64748B;
        margin: 8px 0 10px 0;
    }

    /* --------------------------------------------------------
       METRIC CARDS
    -------------------------------------------------------- */

    .metric-card {
        background: #FFFFFF;
        border: 1px solid #DCE5F0;
        border-radius: 14px;
        padding: 16px 18px;
        min-height: 118px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.035);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 18px rgba(15, 23, 42, 0.07);
    }

    .metric-label {
        color: #64748B;
        font-size: 13px;
        font-weight: 650;
        margin-bottom: 10px;
    }

    .metric-value {
        color: #102A43;
        font-size: clamp(23px, 2vw, 30px);
        line-height: 1.15;
        font-weight: 800;
        overflow-wrap: anywhere;
    }

    .metric-note {
        margin-top: 8px;
        color: #64748B;
        font-size: 11px;
    }

    /* --------------------------------------------------------
       CHARTS
    -------------------------------------------------------- */

    [data-testid="stPlotlyChart"] {
        background: #FFFFFF;
        border: 1px solid #DCE5F0;
        border-radius: 14px;
        padding: 8px;
        margin-bottom: 16px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.045);
        animation: chartAppear 0.45s ease-out;
    }

    /* --------------------------------------------------------
       DATA TABLES
    -------------------------------------------------------- */

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        border: 1px solid #DCE5F0;
        border-radius: 12px;
        overflow: hidden;
        background: #FFFFFF;
    }

    /* --------------------------------------------------------
       CLAIMS & PAYMENTS TABS
       Improves tab label contrast, clickability and selection.
    -------------------------------------------------------- */

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid #DCE5F0;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #E8EEF6 !important;
        color: #1F2937 !important;
        border: 1px solid #CBD5E1 !important;
        border-bottom: none !important;
        border-radius: 9px 9px 0 0 !important;
        padding: 10px 18px !important;
        min-height: 44px;
        font-weight: 700 !important;
        opacity: 1 !important;
        transition:
            background-color 0.2s ease,
            color 0.2s ease;
    }

    /* Tab text can be rendered in nested elements. */
    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span,
    .stTabs [data-baseweb="tab"] div {
        color: #1F2937 !important;
        font-weight: 700 !important;
    }

    /* Hover: make the tab visibly interactive. */
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #DBEAFE !important;
        border-color: #93C5FD !important;
    }

    .stTabs [data-baseweb="tab"]:hover p,
    .stTabs [data-baseweb="tab"]:hover span,
    .stTabs [data-baseweb="tab"]:hover div {
        color: #1E3A8A !important;
    }

    /* Selected tab: blue background with white text. */
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #1D4ED8 !important;
        border-color: #1D4ED8 !important;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] p,
    .stTabs [data-baseweb="tab"][aria-selected="true"] span,
    .stTabs [data-baseweb="tab"][aria-selected="true"] div {
        color: #FFFFFF !important;
    }

    /* Keyboard focus visibility. */
    .stTabs [data-baseweb="tab"]:focus-visible {
        outline: 3px solid #93C5FD !important;
        outline-offset: 2px !important;
    }

    /* --------------------------------------------------------
       BUTTONS
    -------------------------------------------------------- */

    .stButton > button,
    .stDownloadButton > button {
        background-color: #1D4ED8 !important;
        color: #FFFFFF !important;
        border: 1px solid #1D4ED8 !important;
        border-radius: 9px !important;
        font-weight: 750 !important;
        min-height: 40px;
        transition:
            background-color 0.2s ease,
            transform 0.2s ease;
    }

    .stButton > button *,
    .stDownloadButton > button *,
    .stButton > button p,
    .stDownloadButton > button p,
    .stButton > button span,
    .stDownloadButton > button span,
    .stButton > button div,
    .stDownloadButton > button div {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        background-color: #1E40AF !important;
        border-color: #1E40AF !important;
        transform: translateY(-1px);
    }

    .stButton > button:focus,
    .stDownloadButton > button:focus {
        outline: 3px solid #93C5FD !important;
        outline-offset: 2px;
    }

    /* --------------------------------------------------------
       ALERTS
    -------------------------------------------------------- */

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    /* --------------------------------------------------------
       INFORMATION NOTES
    -------------------------------------------------------- */

    .info-note {
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-radius: 10px;
        padding: 12px 14px;
        color: #1E3A8A;
        font-size: 13px;
        line-height: 1.55;
        margin: 8px 0 16px 0;
    }

    /* --------------------------------------------------------
       FOOTER
    -------------------------------------------------------- */

    .footer {
        text-align: center;
        color: #64748B;
        font-size: 11px;
        padding: 22px 0 10px 0;
        margin-top: 30px;
        border-top: 1px solid #DCE5F0;
    }

    /* --------------------------------------------------------
       ANIMATIONS
    -------------------------------------------------------- */

    @keyframes wallpaperShift {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }

    @keyframes sidebarFlow {
        0%, 100% { background-position: 0% 20%; }
        50% { background-position: 100% 80%; }
    }

    @keyframes heroGradient {
        0%, 100% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
    }

    @keyframes shieldFloat {
        0%, 100% {
            transform: translateY(0px);
        }
        50% {
            transform: translateY(-7px);
        }
    }

    @keyframes chartAppear {
        from {
            opacity: 0.45;
            transform: translateY(5px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Respect reduced-motion accessibility settings. */
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation: none !important;
            transition: none !important;
            scroll-behavior: auto !important;
        }
    }

    /* ========================================================
       ENHANCED COMMAND-CENTER VISUAL LAYER
    ======================================================== */
    .hero {
        border: 1px solid rgba(191, 219, 254, .28);
        box-shadow: 0 16px 42px rgba(16, 42, 67, .20);
    }
    .hero::before {
        content: "";
        position: absolute;
        inset: -80% -20%;
        background: linear-gradient(110deg, transparent 38%, rgba(255,255,255,.12) 48%, transparent 58%);
        transform: translateX(-65%) rotate(10deg);
        animation: heroShimmer 9s ease-in-out infinite;
        pointer-events: none;
    }
    .hero::after {
        content: "";
        position: absolute;
        width: 240px;
        height: 240px;
        right: 7%;
        top: -120px;
        border: 1px solid rgba(191,219,254,.18);
        border-radius: 50%;
        box-shadow: 0 0 0 24px rgba(191,219,254,.035), 0 0 0 50px rgba(191,219,254,.025);
        pointer-events: none;
    }
    .metric-card {
        position: relative;
        overflow: hidden;
        border-top: 3px solid #3B82F6;
        box-shadow: 0 7px 22px rgba(15,23,42,.055);
    }
    .metric-card::after {
        content: "";
        position: absolute;
        right: -22px;
        bottom: -36px;
        width: 95px;
        height: 95px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(59,130,246,.12), transparent 68%);
        pointer-events: none;
    }
    .section-label {
        display: flex;
        align-items: center;
        gap: 9px;
    }
    .section-label::before {
        content: "";
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #10B981;
        box-shadow: 0 0 0 4px rgba(16,185,129,.13);
    }
    .live-status {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        border: 1px solid #BBF7D0;
        border-radius: 999px;
        padding: 7px 12px;
        color: #166534;
        background: #F0FDF4;
        font-size: 12px;
        font-weight: 800;
    }
    .live-status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #22C55E;
        box-shadow: 0 0 0 0 rgba(34,197,94,.45);
        animation: livePulse 1.8s infinite;
    }
    .feature-card {
        border: 1px solid #DCE5F0;
        border-radius: 15px;
        padding: 17px;
        background: linear-gradient(145deg, #FFFFFF, #F8FBFF);
        min-height: 120px;
        box-shadow: 0 5px 16px rgba(15,23,42,.04);
        transition: transform .2s ease, box-shadow .2s ease;
    }
    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 25px rgba(15,23,42,.09);
    }
    .feature-card .feature-icon { font-size: 25px; margin-bottom: 8px; }
    .feature-card .feature-title { color: #102A43; font-weight: 800; font-size: 14px; }
    .feature-card .feature-copy { color: #64748B; font-size: 12px; line-height: 1.55; margin-top: 5px; }
    @keyframes heroShimmer {
        0%, 60%, 100% { transform: translateX(-70%) rotate(10deg); }
        30% { transform: translateX(70%) rotate(10deg); }
    }
    @keyframes livePulse {
        0% { box-shadow: 0 0 0 0 rgba(34,197,94,.45); }
        70% { box-shadow: 0 0 0 7px rgba(34,197,94,0); }
        100% { box-shadow: 0 0 0 0 rgba(34,197,94,0); }
    }

    /* Responsive styling for smaller screens. */
    @media (max-width: 700px) {
        .hero {
            padding: 22px;
        }

        .hero-icon {
            right: 15px;
            top: 12px;
            font-size: 45px;
        }

        .stTabs [data-baseweb="tab"] {
            padding: 8px 12px !important;
            min-height: 42px;
        }
    }
    
/* ============================================================
   GLOBAL FORM READABILITY — ALL DASHBOARD PAGES
============================================================ */

/* Main page widget labels */
.stApp [data-testid="stWidgetLabel"],
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stWidgetLabel"] label,
.stApp .stRadio label,
.stApp .stCheckbox label {
    color: #172033 !important;
    font-weight: 650 !important;
}

/* Supporting descriptions and help text */
.stApp [data-testid="stCaptionContainer"],
.stApp [data-testid="stCaptionContainer"] p,
.stApp [data-testid="stHelp"],
.stApp [data-testid="stTooltipIcon"] {
    color: #475569 !important;
}

/* Text, number, date and other input controls */
.stApp input,
.stApp textarea {
    color: #172033 !important;
    background-color: #F1F5F9 !important;
    border: 1px solid #94A3B8 !important;
    border-radius: 8px !important;
    caret-color: #1D4ED8 !important;
}

/* Select boxes and dropdown controls */
.stApp [data-baseweb="select"] > div {
    background-color: #F1F5F9 !important;
    border-color: #94A3B8 !important;
    color: #172033 !important;
}

/* Text displayed inside selected dropdown values */
.stApp [data-baseweb="select"] *,
.stApp [data-baseweb="select"] input {
    color: #172033 !important;
}

/* Placeholder text */
.stApp input::placeholder,
.stApp textarea::placeholder {
    color: #64748B !important;
    opacity: 1 !important;
}

/* Focused fields */
.stApp input:focus,
.stApp textarea:focus,
.stApp [data-baseweb="select"]:focus-within {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.18) !important;
}

/* Dropdown menu and options */
[data-baseweb="popover"],
[data-baseweb="menu"],
[data-baseweb="popover"] ul {
    background-color: #FFFFFF !important;
}

[data-baseweb="menu"] *,
[data-baseweb="popover"] [role="option"] {
    color: #172033 !important;
}

[data-baseweb="popover"] [role="option"]:hover {
    background-color: #DBEAFE !important;
    color: #1E3A8A !important;
}

/* ============================================================
   TABS — CLEAR SELECTED AND UNSELECTED STATES
============================================================ */

.stTabs [data-baseweb="tab-list"] {
    gap: 8px !important;
    border-bottom: 2px solid #CBD5E1 !important;
}

/* Inactive tab */
.stTabs [data-baseweb="tab"] {
    background-color: #E2E8F0 !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 9px 9px 0 0 !important;
    min-height: 46px !important;
    padding: 10px 18px !important;
    opacity: 1 !important;
}

/* Inactive tab text */
.stTabs [data-baseweb="tab"]:not([aria-selected="true"]),
.stTabs [data-baseweb="tab"]:not([aria-selected="true"]) p,
.stTabs [data-baseweb="tab"]:not([aria-selected="true"]) span,
.stTabs [data-baseweb="tab"]:not([aria-selected="true"]) div {
    color: #172033 !important;
    font-weight: 700 !important;
}

/* Hover state */
.stTabs [data-baseweb="tab"]:hover {
    background-color: #BFDBFE !important;
    border-color: #60A5FA !important;
}

.stTabs [data-baseweb="tab"]:hover p,
.stTabs [data-baseweb="tab"]:hover span,
.stTabs [data-baseweb="tab"]:hover div {
    color: #1E3A8A !important;
}

/* Selected tab */
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background-color: #1D4ED8 !important;
    border-color: #1D4ED8 !important;
    opacity: 1 !important;
}

.stTabs [data-baseweb="tab"][aria-selected="true"],
.stTabs [data-baseweb="tab"][aria-selected="true"] p,
.stTabs [data-baseweb="tab"][aria-selected="true"] span,
.stTabs [data-baseweb="tab"][aria-selected="true"] div {
    color: #FFFFFF !important;
    font-weight: 800 !important;
}

/* Visible keyboard focus */
.stTabs [data-baseweb="tab"]:focus-visible {
    outline: 3px solid #60A5FA !important;
    outline-offset: 2px !important;
}

/* Stronger selected-tab indicator */
.stTabs [data-baseweb="tab-highlight"] {
    background-color: #1D4ED8 !important;
    height: 4px !important;
}

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. DATABASE CONNECTION AND READ-ONLY QUERIES
# ============================================================

ALLOWED_TABLES = {
    "customers",
    "policies",
    "premium_payments",
    "claims",
}


def get_db_config() -> dict[str, Any]:
    """Read PostgreSQL settings from environment variables."""
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "dbname": os.getenv("DB_NAME", ""),
        "user": os.getenv("DB_USER", ""),
        "password": os.getenv("DB_PASSWORD", ""),
        "connect_timeout": 5,
    }


def run_query(query: str, params: tuple | None = None) -> pd.DataFrame:
    """
    Run a read-only query and return the result as a DataFrame.

    The SQL is restricted to SELECT/WITH statements. Table names used by
    read_table() are checked against ALLOWED_TABLES.
    """
    sql = query.strip()

    if not re.match(r"^(SELECT|WITH)\b", sql, flags=re.IGNORECASE):
        raise ValueError("Only SELECT or WITH read queries are permitted.")

    config = get_db_config()

    if not config["dbname"] or not config["user"]:
        raise RuntimeError(
            "Database configuration is incomplete. Check DB_NAME and DB_USER "
            "in your project .env file."
        )

    with psycopg.connect(**config) as connection:
        # Set the transaction to read-only as an additional safety measure.
        connection.execute("SET TRANSACTION READ ONLY")
        return pd.read_sql_query(sql, connection, params=params)


@st.cache_data(ttl=60, show_spinner=False)
def read_table(table_name: str) -> pd.DataFrame:
    """Read a permitted insurance table from PostgreSQL."""
    if table_name not in ALLOWED_TABLES:
        raise ValueError(f"Table is not permitted: {table_name}")

    # Table names come only from the allowlist above.
    query = f'SELECT * FROM "{table_name}" LIMIT 100000'
    return run_query(query)


# ============================================================
# 4. DATA NORMALISATION HELPERS
# ============================================================

def find_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    """Find a column without depending on its case or punctuation."""
    if df.empty:
        return None

    def normalise(value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", str(value).lower())

    lookup = {normalise(column): column for column in df.columns}

    for candidate in candidates:
        match = lookup.get(normalise(candidate))
        if match is not None:
            return match

    return None


def series_or_default(
    df: pd.DataFrame,
    candidates: list[str],
    default: Any = None,
) -> pd.Series:
    """Return a matched column, or a default Series aligned to the data."""
    column = find_column(df, candidates)

    if column is not None:
        return df[column]

    return pd.Series(default, index=df.index, dtype="object")


def standardise_data(
    customers: pd.DataFrame,
    policies: pd.DataFrame,
    payments: pd.DataFrame,
    claims: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Create consistent column names and add useful customer/product fields
    from related tables when join keys are available.
    """
    customers = customers.copy()
    policies = policies.copy()
    payments = payments.copy()
    claims = claims.copy()

    def rename_if_found(
        df: pd.DataFrame,
        target: str,
        candidates: list[str],
    ) -> None:
        source = find_column(df, candidates)
        if source is not None and source != target:
            if target not in df.columns:
                df.rename(columns={source: target}, inplace=True)

    # Customer fields
    for target, candidates in {
        "customer_id": ["customer_id", "customerid", "client_id", "clientid"],
        "age": ["age", "customer_age"],
        "customer_type": ["customer_type", "client_type", "segment"],
        "gender": ["gender", "sex"],
        "city": ["city", "town"],
    }.items():
        rename_if_found(customers, target, candidates)

    # Policy fields
    for target, candidates in {
        "policy_id": ["policy_id", "policyid", "policy_number", "policyno"],
        "customer_id": ["customer_id", "customerid", "client_id"],
        "product": ["product", "product_name", "insurance_type", "policy_type"],
        "annual_premium": [
            "annual_premium", "premium_amount", "premium", "yearly_premium"
        ],
        "policy_status": ["policy_status", "status"],
        "policy_start_date": [
            "policy_start_date", "start_date", "inception_date", "effective_date"
        ],
        "policy_end_date": ["policy_end_date", "end_date", "expiry_date"],
    }.items():
        rename_if_found(policies, target, candidates)

    # Payment fields
    for target, candidates in {
        "payment_id": ["payment_id", "paymentid", "receipt_id", "receipt_no"],
        "policy_id": ["policy_id", "policyid", "policy_number", "policyno"],
        "payment_amount": [
            "payment_amount", "amount", "premium_paid", "paid_amount"
        ],
        "payment_status": ["payment_status", "status"],
        "payment_date": ["payment_date", "paid_date", "transaction_date"],
    }.items():
        rename_if_found(payments, target, candidates)

    # Claim fields
    for target, candidates in {
        "claim_id": ["claim_id", "claimid", "claim_number", "claim_no"],
        "policy_id": ["policy_id", "policyid", "policy_number", "policyno"],
        "claim_amount": [
            "claim_amount", "amount", "claimed_amount", "loss_amount"
        ],
        "claim_status": ["claim_status", "status"],
        "claim_date": ["claim_date", "date_of_claim", "reported_date"],
    }.items():
        rename_if_found(claims, target, candidates)

    # Convert recognised date and amount columns.
    for df, columns in [
        (policies, ["policy_start_date", "policy_end_date"]),
        (payments, ["payment_date"]),
        (claims, ["claim_date"]),
    ]:
        for column in columns:
            if column in df.columns:
                df[column] = pd.to_datetime(df[column], errors="coerce")

    for df, columns in [
        (policies, ["annual_premium"]),
        (payments, ["payment_amount"]),
        (claims, ["claim_amount"]),
    ]:
        for column in columns:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")

    # Add customer attributes to policies, without overwriting policy fields.
    if (
        not customers.empty
        and not policies.empty
        and "customer_id" in customers.columns
        and "customer_id" in policies.columns
    ):
        customer_fields = [
            column for column in
            ["customer_id", "customer_type", "age", "gender", "city"]
            if column in customers.columns
        ]
        customer_lookup = customers[customer_fields].drop_duplicates(
            subset=["customer_id"]
        )

        missing_fields = [
            column for column in customer_fields
            if column != "customer_id" and column not in policies.columns
        ]

        if missing_fields:
            policies = policies.merge(
                customer_lookup[["customer_id"] + missing_fields],
                on="customer_id",
                how="left",
            )

    # Add product/customer information to claims through policy_id.
    if (
        not policies.empty
        and not claims.empty
        and "policy_id" in policies.columns
        and "policy_id" in claims.columns
    ):
        enrich_fields = [
            column for column in
            ["policy_id", "product", "customer_id", "annual_premium"]
            if column in policies.columns
        ]
        policy_lookup = policies[enrich_fields].drop_duplicates(
            subset=["policy_id"]
        )

        missing_fields = [
            column for column in enrich_fields
            if column != "policy_id" and column not in claims.columns
        ]

        if missing_fields:
            claims = claims.merge(
                policy_lookup[["policy_id"] + missing_fields],
                on="policy_id",
                how="left",
            )

    # Add product information to payments.
    if (
        not policies.empty
        and not payments.empty
        and "policy_id" in policies.columns
        and "policy_id" in payments.columns
        and "product" in policies.columns
        and "product" not in payments.columns
    ):
        policy_products = policies[
            ["policy_id", "product"]
        ].drop_duplicates(subset=["policy_id"])

        payments = payments.merge(
            policy_products,
            on="policy_id",
            how="left",
        )

    return {
        "customers": customers,
        "policies": policies,
        "premium_payments": payments,
        "claims": claims,
    }


@st.cache_data(ttl=60, show_spinner=False)
def load_data() -> tuple[dict[str, pd.DataFrame], str | None]:
    """Load all four database tables and return an error message if needed."""
    try:
        raw = {
            "customers": read_table("customers"),
            "policies": read_table("policies"),
            "premium_payments": read_table("premium_payments"),
            "claims": read_table("claims"),
        }

        return standardise_data(
            raw["customers"],
            raw["policies"],
            raw["premium_payments"],
            raw["claims"],
        ), None

    except Exception as exc:
        return {
            "customers": pd.DataFrame(),
            "policies": pd.DataFrame(),
            "premium_payments": pd.DataFrame(),
            "claims": pd.DataFrame(),
        }, str(exc)


# ============================================================
# 5. SHARED PRESENTATION HELPERS
# ============================================================

def show_banner(
    section: str,
    description: str,
    illustration: str = "🛡️",
) -> None:
    """Display the animated dashboard header."""
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-topline">InsureData360-AI · DCW Company PLC</div>
            <div class="hero-icon">{illustration}</div>
            <h1>{section}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_footer() -> None:
    st.markdown(
        """
        <div class="footer">
            InsureData360-AI · Insurance Analytics & Machine Learning<br>
            Built by Dumindu Wickramaarachchi · © 2026<br>
            Demonstration dashboard using synthetic insurance data
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_currency(value: Any) -> str:
    try:
        number = float(value)
        if pd.isna(number):
            return "—"
        return f"{number:,.2f}"
    except (TypeError, ValueError):
        return "—"


def show_metric(
    label: str,
    value: Any,
    note: str = "",
    prefix: str = "",
) -> None:
    """Display a consistent bordered KPI card."""
    display_value = f"{prefix}{value}"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{display_value}</div>
            <div class="metric-note">{note or "&nbsp;"}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_empty(message: str) -> None:
    st.info(message)


def add_plot_style(
    fig: go.Figure,
    title: str | None = None,
    height: int = 360,
) -> go.Figure:
    """Apply readable fonts, high contrast and consistent chart styling."""
    fig.update_layout(
        title=dict(
            text=title or "",
            font=dict(size=17, color="#172033"),
            x=0.02,
            xanchor="left",
        ),
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="Arial, sans-serif",
            size=12,
            color="#334155",
        ),
        margin=dict(l=24, r=24, t=65, b=28),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#E2E8F0",
            borderwidth=1,
            font=dict(color="#334155", size=11),
        ),
        hoverlabel=dict(
            bgcolor="#102A43",
            font_color="#FFFFFF",
            font_size=12,
        ),
        transition=dict(duration=450, easing="cubic-in-out"),
    )

    fig.update_xaxes(
        title_font=dict(color="#334155", size=12),
        tickfont=dict(color="#334155", size=11),
        showline=True,
        linecolor="#CBD5E1",
        gridcolor="#EEF2F7",
        zerolinecolor="#CBD5E1",
    )

    fig.update_yaxes(
        title_font=dict(color="#334155", size=12),
        tickfont=dict(color="#334155", size=11),
        showline=False,
        gridcolor="#EEF2F7",
        zerolinecolor="#CBD5E1",
    )

    return fig


def show_chart(fig: go.Figure, key: str | None = None) -> None:
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displaylogo": False,
            "responsive": True,
            "scrollZoom": False,
        },
        key=key,
    )


def get_amount(
    df: pd.DataFrame,
    candidates: list[str],
) -> pd.Series:
    """Return a numeric Series for the first matching amount column."""
    column = find_column(df, candidates)
    if column is None:
        return pd.Series(0.0, index=df.index, dtype="float64")

    return pd.to_numeric(df[column], errors="coerce").fillna(0.0)


def get_text(
    df: pd.DataFrame,
    candidates: list[str],
    default: str = "Unknown",
) -> pd.Series:
    """Return a clean text Series from a matching column."""
    column = find_column(df, candidates)
    if column is None:
        return pd.Series(default, index=df.index, dtype="object")

    return df[column].fillna(default).astype(str)


def show_dataframe(
    df: pd.DataFrame,
    empty_message: str = "No records are available.",
) -> None:
    if df.empty:
        show_empty(empty_message)
    else:
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# 6. ANOMALY FLAG NORMALISATION
# ============================================================

TRUE_TOKENS = {
    "true", "yes", "y", "1", "anomaly", "unusual",
    "outlier", "flagged", "review", "needsreview",
}
FALSE_TOKENS = {
    "false", "no", "n", "0", "normal", "typical",
    "inlier", "notanomaly", "notoutlier", "notflagged",
}


def normalise_token(value: Any) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9-]", "", str(value).strip().lower())


def get_anomaly_mask(
    series: pd.Series,
    column_name: str = "",
) -> pd.Series:
    """
    Convert common model output formats into a boolean anomaly mask.

    Isolation Forest typically uses -1 for anomalies and +1 for inliers.
    Boolean-style columns use explicit true/false tokens.
    """
    values = series.copy()
    name = re.sub(r"[^a-z0-9]", "", column_name.lower())

    boolean_column = any(
        marker in name
        for marker in ["isanomaly", "anomalyflag", "isoutlier", "flagged"]
    )

    numeric = pd.to_numeric(values, errors="coerce")
    unique_numeric = set(numeric.dropna().unique().tolist())

    if (
        not boolean_column
        and unique_numeric
        and unique_numeric.issubset({-1, 1})
    ):
        result = numeric.eq(-1)
        return result.fillna(False).astype(bool)

    def classify(value: Any) -> bool:
        token = normalise_token(value)

        if not token:
            return False

        # Resolve negative labels before positive substring checks.
        if token in FALSE_TOKENS:
            return False

        if token.startswith("not") and any(
            word in token for word in ["anomaly", "outlier", "flagged"]
        ):
            return False

        if token in TRUE_TOKENS:
            return True

        # Allow compound labels such as "high_risk_anomaly".
        positive_words = ["anomaly", "unusual", "outlier", "flagged", "review"]
        return any(word in token for word in positive_words)

    return values.map(classify).fillna(False).astype(bool)


# ============================================================
# 7. CLAIM-SEVERITY MODEL METRICS
# ============================================================

METRIC_ALIASES = {
    "mae": {
        "mae",
        "meanabsoluteerror",
        "testmae",
        "validationmae",
        "valmae",
    },
    "rmse": {
        "rmse",
        "rootmeansquarederror",
        "testrmse",
        "validationrmse",
        "valrmse",
    },
    "r2": {
        "r2",
        "r2score",
        "rsquared",
        "r2test",
        "testr2",
        "r2scoretest",
        "r2scoretestset",
        "r2scorevalidation",
        "r2validation",
    },
}


def normalise_metric_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def scan_json_for_metrics(data: Any) -> dict[str, float]:
    """
    Recursively search dictionaries/lists for MAE, RMSE and R² aliases.
    Handles nested structures such as {"test_metrics": {"MAE": 123}}.
    """
    found: dict[str, float] = {}

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key, value in item.items():
                normalized = normalise_metric_key(key)

                for metric_name, aliases in METRIC_ALIASES.items():
                    if normalized in aliases:
                        try:
                            number = float(value)
                            if pd.notna(number):
                                found.setdefault(metric_name, number)
                        except (TypeError, ValueError):
                            pass

                if isinstance(value, (dict, list)):
                    visit(value)

        elif isinstance(item, list):
            for value in item:
                visit(value)

    visit(data)
    return found


def load_severity_metrics() -> tuple[dict[str, float], str | None]:
    if not SEVERITY_METRICS_FILE.exists():
        return {}, (
            f"Metrics file not found: {SEVERITY_METRICS_FILE}. "
            "Run the claim-severity training/evaluation pipeline to create it."
        )

    try:
        with SEVERITY_METRICS_FILE.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        metrics = scan_json_for_metrics(payload)

        if not metrics:
            return {}, (
                "The metrics JSON was read, but MAE, RMSE and R² keys were "
                "not recognised. Expected keys include mae, rmse, r2, "
                "mean_absolute_error, root_mean_squared_error and r2_score."
            )

        return metrics, None

    except (OSError, json.JSONDecodeError) as exc:
        return {}, f"Unable to read model metrics JSON: {exc}"


# ============================================================
# 8. SIDEBAR NAVIGATION AND DATA LOADING
# ============================================================

with st.sidebar:
    st.markdown("## 🛡️ InsureData360-AI")
    st.caption("Insurance intelligence workspace")
    st.divider()

    page = st.radio(
        "Navigate",
        [
            "Overview",
            "Claims & Payments",
            "ML Insights",
            "Data Quality",
            "Warehouse",
        ],
        index=0,
    )

    st.divider()
    st.markdown("### Data connection")

    if st.button("↻ Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    motion_effects = st.checkbox(
        "✨ Motion effects",
        value=True,
        key="motion_effects",
        help="Turn off animated interface effects if you prefer reduced motion.",
    )
    if not motion_effects:
        st.markdown(
            """<style>
            .stApp, [data-testid="stSidebar"],
            .hero, .hero::before, .hero-icon, .metric-card, [data-testid="stPlotlyChart"],
            .live-status-dot, .prediction-result, .feature-card {
                animation: none !important;
                transition: none !important;
                transform: none !important;
            }
            </style>""",
            unsafe_allow_html=True,
        )

    st.caption("Database cache: up to 60 seconds")
    st.caption("Use synthetic/demo data only.")


with st.spinner("Loading insurance data..."):
    data, db_error = load_data()

customers = data["customers"]
policies = data["policies"]
payments = data["premium_payments"]
claims = data["claims"]

database_ready = db_error is None

if not database_ready:
    st.warning(
        "The dashboard could not load all database tables. "
        "Pages remain available, but database-driven charts may be empty."
    )
    with st.expander("Database connection details"):
        st.code(db_error or "Unknown database error")
        st.markdown(
            """
            Check the following:
            - PostgreSQL is running.
            - The database name, username and password in `.env` are correct.
            - The tables `customers`, `policies`, `premium_payments` and `claims`
              exist in the configured database.
            - Your Python environment has `psycopg[binary]` installed.
            """
        )


# ============================================================
# 9. OVERVIEW PAGE
# ============================================================

if page == "Overview":
    show_banner(
        "Insurance Intelligence Overview",
        "A consolidated view of customers, policy portfolios, premiums and "
        "claims, with machine-learning outputs where available.",
        "🛡️",
    )

    st.markdown(
        '<div class="live-status"><span class="live-status-dot"></span> DASHBOARD SESSION ACTIVE</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"Last rendered: {datetime.now().strftime('%d %b %Y · %H:%M:%S')} · Data refresh is manual from the sidebar.")

    total_customers = len(customers)
    total_policies = len(policies)
    total_claims = len(claims)
    total_premium = get_amount(
        policies, ["annual_premium", "premium_amount", "premium"]
    ).sum()
    total_claim_amount = get_amount(
        claims, ["claim_amount", "amount", "claimed_amount"]
    ).sum()

    anomaly_count = 0
    anomaly_share = 0.0

    if ANOMALY_FILE.exists():
        try:
            anomaly_df = pd.read_csv(ANOMALY_FILE)
            anomaly_column = find_column(
                anomaly_df,
                [
                    "is_anomaly", "anomaly_flag", "is_outlier",
                    "flagged", "prediction", "anomaly",
                    "anomaly_label", "review_status",
                ],
            )

            if anomaly_column:
                anomaly_mask = get_anomaly_mask(
                    anomaly_df[anomaly_column],
                    anomaly_column,
                )
                anomaly_count = int(anomaly_mask.sum())
                anomaly_share = (
                    anomaly_count / len(anomaly_df) * 100
                    if len(anomaly_df) else 0.0
                )
        except Exception:
            pass

    st.markdown('<div class="section-label">Portfolio snapshot</div>',
                unsafe_allow_html=True)

    metric_cols = st.columns(4)
    with metric_cols[0]:
        show_metric("Customers", f"{total_customers:,}", "Customer records")
    with metric_cols[1]:
        show_metric("Policies", f"{total_policies:,}", "Policy records")
    with metric_cols[2]:
        show_metric("Total annual premium", format_currency(total_premium),
                    "Currency units from source data")
    with metric_cols[3]:
        show_metric("Claims", f"{total_claims:,}", "Claim records")

    metric_cols2 = st.columns(3)
    with metric_cols2[0]:
        show_metric("Claim amount total", format_currency(total_claim_amount),
                    "Sum of recorded claim amounts")
    with metric_cols2[1]:
        show_metric("Flagged records", f"{anomaly_count:,}",
                    "From available model output")
    with metric_cols2[2]:
        show_metric("Flagged share", f"{anomaly_share:.1f}%",
                    "Of model-output rows")

    # Interactive scenario modelling: transparent arithmetic, not a model forecast.
    with st.expander("🧮 Premium portfolio scenario simulator", expanded=False):
        st.caption(
            "Explore a hypothetical premium change against the current policy premium total. "
            "This is a what-if calculation, not a predicted business outcome."
        )
        premium_change_pct = st.slider(
            "Hypothetical annual premium adjustment",
            min_value=-30,
            max_value=50,
            value=5,
            step=1,
            format="%d%%",
            key="portfolio_premium_scenario",
        )
        scenario_total = float(total_premium) * (1 + premium_change_pct / 100)
        scenario_delta = scenario_total - float(total_premium)
        scenario_cols = st.columns(3)
        with scenario_cols[0]:
            show_metric("Current portfolio premium", format_currency(total_premium), "Source policy records")
        with scenario_cols[1]:
            show_metric("Scenario premium", format_currency(scenario_total), f"{premium_change_pct:+d}% scenario")
        with scenario_cols[2]:
            show_metric("Scenario difference", format_currency(scenario_delta), "Calculated change")

    feature_cols = st.columns(3)
    feature_cards = [
        ("🧠", "ML-assisted screening", "Flag unusual claim patterns for human review; flags are not proof of fraud."),
        ("📈", "Portfolio analytics", "Compare product premiums, claim status mix and claim-value trends."),
        ("📤", "Export-ready insights", "Download filtered claim registers and an executive snapshot for reporting."),
    ]
    for feature_col, (feature_icon, feature_title, feature_copy) in zip(feature_cols, feature_cards):
        with feature_col:
            st.markdown(
                f'<div class="feature-card"><div class="feature-icon">{feature_icon}</div>'
                f'<div class="feature-title">{feature_title}</div><div class="feature-copy">{feature_copy}</div></div>',
                unsafe_allow_html=True,
            )
    st.write("")

    # One-click executive snapshot export uses the data already loaded on this page.
    executive_snapshot = pd.DataFrame([
        {"Metric": "Customers", "Value": total_customers},
        {"Metric": "Policies", "Value": total_policies},
        {"Metric": "Annual premium total", "Value": float(total_premium)},
        {"Metric": "Claims", "Value": total_claims},
        {"Metric": "Claim amount total", "Value": float(total_claim_amount)},
        {"Metric": "Flagged records", "Value": anomaly_count},
        {"Metric": "Flagged share percent", "Value": round(anomaly_share, 2)},
        {"Metric": "Scenario premium adjustment percent", "Value": int(st.session_state.get("portfolio_premium_scenario", 5))},
        {"Metric": "Scenario annual premium", "Value": float(total_premium) * (1 + int(st.session_state.get("portfolio_premium_scenario", 5)) / 100)},
    ])
    st.download_button(
        "⬇️ Download executive snapshot (CSV)",
        data=executive_snapshot.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"insuredata360_executive_snapshot_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        key="overview_executive_snapshot",
    )

    # Status counts are displayed only when a recognisable status field exists.
    overview_status_col = find_column(claims, ["claim_status", "status"])
    if overview_status_col and not claims.empty:
        status_values = claims[overview_status_col].fillna("Unknown").astype(str).str.strip().str.lower()
        paid_count = int(status_values.isin({"paid", "settled", "closed-paid", "completed"}).sum())
        pending_count = int(status_values.str.contains(r"pending|in progress|under review|processing", regex=True, na=False).sum())
        rejected_count = int(status_values.str.contains(r"reject|declin|denied", regex=True, na=False).sum())
        status_cols = st.columns(3)
        with status_cols[0]:
            show_metric("Paid / settled", f"{paid_count:,}", "Based on claim status labels")
        with status_cols[1]:
            show_metric("Pending / in review", f"{pending_count:,}", "Based on claim status labels")
        with status_cols[2]:
            show_metric("Rejected / declined", f"{rejected_count:,}", "Based on claim status labels")
        st.caption("Status groupings use recognised text labels in the source data; unmatched labels are not forced into these categories.")

    st.markdown('<div class="section-label">Portfolio analytics</div>',
                unsafe_allow_html=True)

    left, right = st.columns(2)

    with left:
        if not policies.empty:
            product_col = find_column(
                policies, ["product", "product_name", "policy_type"]
            )

            if product_col:
                product_data = policies.copy()
                product_data["Product"] = (
                    product_data[product_col].fillna("Unknown").astype(str)
                )
                product_data["Annual premium"] = get_amount(
                    product_data,
                    ["annual_premium", "premium_amount", "premium"],
                )
                grouped = (
                    product_data.groupby("Product", as_index=False)
                    .agg(
                        Policies=("Product", "size"),
                        Premium=("Annual premium", "sum"),
                    )
                    .sort_values("Premium", ascending=False)
                )

                fig = px.bar(
                    grouped,
                    x="Product",
                    y="Premium",
                    color="Product",
                    hover_data=["Policies"],
                    labels={"Premium": "Annual premium"},
                )
                fig.update_layout(showlegend=False)
                add_plot_style(fig, "Annual premium by product")
                show_chart(fig, key="overview_premium_product")
            else:
                show_empty("No product column was found in the policy data.")
        else:
            show_empty("No policy records are available.")

    with right:
        if not claims.empty:
            status_col = find_column(
                claims, ["claim_status", "status"]
            )

            if status_col:
                status_data = (
                    claims[status_col]
                    .fillna("Unknown")
                    .astype(str)
                    .value_counts()
                    .rename_axis("Claim status")
                    .reset_index(name="Claims")
                )

                fig = px.pie(
                    status_data,
                    names="Claim status",
                    values="Claims",
                    hole=0.58,
                )
                add_plot_style(fig, "Claim status distribution")
                show_chart(fig, key="overview_claim_status")
            else:
                show_empty("No claim-status column was found.")
        else:
            show_empty("No claim records are available.")

    left, right = st.columns(2)

    with left:
        claim_date_col = find_column(
            claims, ["claim_date", "date_of_claim", "reported_date"]
        )
        claim_amount_col = find_column(
            claims, ["claim_amount", "amount", "claimed_amount"]
        )

        if (
            not claims.empty
            and claim_date_col
            and claim_amount_col
        ):
            trend = claims.copy()
            trend[claim_date_col] = pd.to_datetime(
                trend[claim_date_col], errors="coerce"
            )
            trend[claim_amount_col] = pd.to_numeric(
                trend[claim_amount_col], errors="coerce"
            )
            trend = trend.dropna(
                subset=[claim_date_col, claim_amount_col]
            )

            if not trend.empty:
                trend["Month"] = (
                    trend[claim_date_col].dt.to_period("M").astype(str)
                )
                monthly = (
                    trend.groupby("Month", as_index=False)
                    .agg(
                        Claim_amount=(claim_amount_col, "sum"),
                        Claim_count=(claim_amount_col, "size"),
                    )
                    .sort_values("Month")
                )

                fig = px.line(
                    monthly,
                    x="Month",
                    y="Claim_amount",
                    markers=True,
                    hover_data=["Claim_count"],
                    labels={"Claim_amount": "Claim amount"},
                )
                fig.update_traces(line_width=3)
                add_plot_style(fig, "Monthly claim amount")
                show_chart(fig, key="overview_claim_trend")
            else:
                show_empty("There are no valid claim dates or amounts.")
        else:
            show_empty("Claim dates and amounts are needed for a monthly trend.")

    with right:
        if not claims.empty and claim_amount_col:
            claim_product_col = find_column(
                claims, ["product", "product_name", "policy_type"]
            )

            if claim_product_col:
                product_claims = claims.copy()
                product_claims["Product"] = (
                    product_claims[claim_product_col]
                    .fillna("Unknown")
                    .astype(str)
                )
                product_claims["Claim amount"] = get_amount(
                    product_claims,
                    ["claim_amount", "amount", "claimed_amount"],
                )

                grouped = (
                    product_claims.groupby("Product", as_index=False)
                    .agg(
                        Claim_amount=("Claim amount", "sum"),
                        Claims=("Claim amount", "size"),
                    )
                    .sort_values("Claim_amount", ascending=False)
                )

                fig = px.bar(
                    grouped,
                    x="Product",
                    y="Claim_amount",
                    hover_data=["Claims"],
                    labels={"Claim_amount": "Claim amount"},
                )
                add_plot_style(fig, "Claim amount by product")
                show_chart(fig, key="overview_claim_product")
            else:
                show_empty("No product field is available for claims.")
        else:
            show_empty("Claim amount data is unavailable.")

    st.markdown("### 🛰️ Recent claim activity")
    if not claims.empty:
        event_df = claims.copy()
        event_date_col = find_column(event_df, ["claim_date", "date_of_claim", "reported_date", "created_at"])
        event_status_col = find_column(event_df, ["claim_status", "status"])
        event_amount_col = find_column(event_df, ["claim_amount", "amount", "claimed_amount"])
        if event_date_col:
            event_df["_event_date"] = pd.to_datetime(event_df[event_date_col], errors="coerce")
            event_df = event_df.sort_values("_event_date", ascending=False, na_position="last")
        else:
            event_df["_event_date"] = pd.NaT
        activity = pd.DataFrame()
        activity["Date"] = event_df["_event_date"].dt.strftime("%Y-%m-%d %H:%M").fillna("Date unavailable")
        activity["Status"] = event_df[event_status_col].fillna("Unknown").astype(str) if event_status_col else "Not provided"
        activity["Claim amount"] = pd.to_numeric(event_df[event_amount_col], errors="coerce") if event_amount_col else pd.NA
        activity = activity.head(8)
        st.dataframe(activity, use_container_width=True, hide_index=True)
        st.download_button(
            "Download recent activity CSV",
            data=activity.to_csv(index=False).encode("utf-8-sig"),
            file_name="insuredata360_recent_claim_activity.csv",
            mime="text/csv",
            key="overview_recent_activity_download",
        )
    else:
        show_empty("No claim records are available for the activity feed.")

    st.markdown("### Recent policy records")
    if not policies.empty:
        st.dataframe(policies.tail(10), use_container_width=True, hide_index=True)
    else:
        show_empty("No policy records are available.")


# ============================================================
# 10. CLAIMS & PAYMENTS PAGE
# ============================================================

elif page == "Claims & Payments":
    show_banner(
        "Claims & Payments",
        "Explore claim volumes, claim values, claim statuses and premium-payment "
        "records in one place.",
        "📊",
    )

    claims_tab, payments_tab = st.tabs(["Claims", "Premium payments"])

    with claims_tab:
        claim_amount_col = find_column(
            claims, ["claim_amount", "amount", "claimed_amount"]
        )
        claim_status_col = find_column(claims, ["claim_status", "status"])
        claim_date_col = find_column(
            claims, ["claim_date", "date_of_claim", "reported_date"]
        )

        claim_total = get_amount(
            claims, ["claim_amount", "amount", "claimed_amount"]
        ).sum()
        claim_mean = (
            get_amount(
                claims, ["claim_amount", "amount", "claimed_amount"]
            ).mean()
            if not claims.empty else 0.0
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            show_metric("Claim records", f"{len(claims):,}")
        with c2:
            show_metric("Total claim amount", format_currency(claim_total))
        with c3:
            show_metric("Average claim amount", format_currency(claim_mean))

        left, right = st.columns(2)

        with left:
            if claim_status_col and not claims.empty:
                counts = (
                    claims[claim_status_col]
                    .fillna("Unknown")
                    .astype(str)
                    .value_counts()
                    .rename_axis("Status")
                    .reset_index(name="Claims")
                )
                fig = px.bar(
                    counts,
                    x="Status",
                    y="Claims",
                    color="Status",
                )
                fig.update_layout(showlegend=False)
                add_plot_style(fig, "Claims by status")
                show_chart(fig, key="claims_status_chart")
            else:
                show_empty("Claim status data is unavailable.")

        with right:
            if claim_amount_col and not claims.empty:
                amounts = get_amount(
                    claims, ["claim_amount", "amount", "claimed_amount"]
                )
                chart_data = pd.DataFrame({"Claim amount": amounts})
                fig = px.histogram(
                    chart_data,
                    x="Claim amount",
                    nbins=30,
                    labels={"Claim amount": "Claim amount"},
                )
                add_plot_style(fig, "Claim amount distribution")
                show_chart(fig, key="claims_amount_histogram")
            else:
                show_empty("Claim amount data is unavailable.")

        if claim_date_col and not claims.empty:
            dated = claims.copy()
            dated[claim_date_col] = pd.to_datetime(
                dated[claim_date_col], errors="coerce"
            )
            dated = dated.dropna(subset=[claim_date_col])

            if not dated.empty:
                dated["Month"] = (
                    dated[claim_date_col].dt.to_period("M").astype(str)
                )
                monthly = (
                    dated.groupby("Month")
                    .size()
                    .reset_index(name="Claims")
                    .sort_values("Month")
                )
                fig = px.line(
                    monthly,
                    x="Month",
                    y="Claims",
                    markers=True,
                )
                add_plot_style(fig, "Monthly claim volume")
                show_chart(fig, key="claims_monthly_volume")

        st.markdown("### Claim register")
        if claims.empty:
            show_empty("No claim records are available.")
        else:
            register = claims.copy()
            filter_cols = st.columns([1.4, 1, 1, 1])
            with filter_cols[0]:
                register_search = st.text_input(
                    "Search claim records",
                    placeholder="Claim ID, policy, customer…",
                    key="claim_register_search",
                )
            with filter_cols[1]:
                if claim_status_col:
                    status_options = sorted(register[claim_status_col].fillna("Unknown").astype(str).unique().tolist())
                    chosen_statuses = st.multiselect(
                        "Claim status", status_options, default=status_options,
                        key="claim_register_status",
                    )
                else:
                    chosen_statuses = []
                    st.caption("No status field")
            with filter_cols[2]:
                date_from = date_to = None
                if claim_date_col:
                    valid_dates = pd.to_datetime(register[claim_date_col], errors="coerce").dropna()
                    if not valid_dates.empty:
                        date_from = st.date_input(
                            "From date", value=valid_dates.min().date(),
                            key="claim_register_from",
                        )
                    else:
                        st.caption("No valid dates")
            with filter_cols[3]:
                if claim_date_col and date_from is not None and not valid_dates.empty:
                    date_to = st.date_input(
                        "To date", value=valid_dates.max().date(),
                        min_value=date_from,
                        key="claim_register_to",
                    )

            if claim_status_col:
                register = register[register[claim_status_col].fillna("Unknown").astype(str).isin(chosen_statuses)]
            if claim_date_col and date_from is not None and date_to is not None:
                parsed_dates = pd.to_datetime(register[claim_date_col], errors="coerce")
                register = register.loc[
                    parsed_dates.dt.date.ge(date_from) & parsed_dates.dt.date.le(date_to)
                ]
            if register_search.strip():
                search_value = register_search.strip().lower()
                text_columns = register.select_dtypes(include=["object", "string"]).columns
                matches = pd.Series(False, index=register.index, dtype=bool)
                for col in text_columns:
                    matches |= register[col].astype("string").str.lower().str.contains(
                        re.escape(search_value), na=False, regex=True
                    )
                register = register.loc[matches]

            st.caption(f"{len(register):,} matching claims · showing up to 500 rows")
            st.dataframe(register.head(500), use_container_width=True, hide_index=True)
            st.download_button(
                "Download filtered claims (CSV)",
                data=register.to_csv(index=False).encode("utf-8"),
                file_name="insuredata360_filtered_claims.csv",
                mime="text/csv",
                key="download_filtered_claim_register",
            )

    with payments_tab:
        payment_amount_col = find_column(
            payments, ["payment_amount", "amount", "premium_paid"]
        )
        payment_status_col = find_column(
            payments, ["payment_status", "status"]
        )

        total_payments = get_amount(
            payments, ["payment_amount", "amount", "premium_paid"]
        ).sum()

        c1, c2, c3 = st.columns(3)
        with c1:
            show_metric("Payment records", f"{len(payments):,}")
        with c2:
            show_metric("Total payment amount", format_currency(total_payments))
        with c3:
            show_metric(
                "Payment statuses",
                f"{payments[payment_status_col].nunique():,}"
                if payment_status_col and not payments.empty else "—",
            )

        left, right = st.columns(2)

        with left:
            if payment_status_col and not payments.empty:
                counts = (
                    payments[payment_status_col]
                    .fillna("Unknown")
                    .astype(str)
                    .value_counts()
                    .rename_axis("Status")
                    .reset_index(name="Payments")
                )
                fig = px.pie(
                    counts,
                    names="Status",
                    values="Payments",
                    hole=0.5,
                )
                add_plot_style(fig, "Payment status distribution")
                show_chart(fig, key="payments_status_chart")
            else:
                show_empty("Payment status data is unavailable.")

        with right:
            if payment_amount_col and not payments.empty:
                amounts = get_amount(
                    payments, ["payment_amount", "amount", "premium_paid"]
                )
                chart_data = pd.DataFrame({"Payment amount": amounts})
                fig = px.histogram(
                    chart_data,
                    x="Payment amount",
                    nbins=30,
                )
                add_plot_style(fig, "Payment amount distribution")
                show_chart(fig, key="payments_amount_chart")
            else:
                show_empty("Payment amount data is unavailable.")

        st.markdown("### Payment records")
        show_dataframe(payments.tail(100), "No payment records are available.")


# ============================================================
# 11. MACHINE-LEARNING INSIGHTS PAGE
# ============================================================

elif page == "ML Insights":
    show_banner(
        "Machine Learning Insights",
        "Review claim-anomaly outputs and claim-severity model evaluation metrics. "
        "Model flags identify records for investigation; they do not prove fraud.",
        "🤖",
    )

    render_live_prediction()
    st.divider()

    st.markdown("### Claim anomaly detection")

    if not ANOMALY_FILE.exists():
        st.warning(
            f"Anomaly results were not found at `{ANOMALY_FILE}`. "
            "Run your anomaly-detection pipeline to generate the CSV."
        )
        anomaly_df = pd.DataFrame()

    else:
        try:
            anomaly_df = pd.read_csv(ANOMALY_FILE)
        except Exception as exc:
            st.error(f"Could not read the anomaly-results CSV: {exc}")
            anomaly_df = pd.DataFrame()

    if not anomaly_df.empty:
        anomaly_col = find_column(
            anomaly_df,
            [
                "is_anomaly", "anomaly_flag", "is_outlier",
                "flagged", "prediction", "anomaly",
                "anomaly_label", "review_status",
            ],
        )

        if anomaly_col:
            anomaly_mask = get_anomaly_mask(
                anomaly_df[anomaly_col],
                anomaly_col,
            )
        else:
            anomaly_mask = pd.Series(
                False, index=anomaly_df.index, dtype=bool
            )
            st.warning(
                "No recognised anomaly-label column was found. "
                "All records are treated as typical for display only. "
                "Check the column names in the model-output CSV."
            )

        anomaly_count = int(anomaly_mask.sum())
        typical_count = int((~anomaly_mask).sum())
        anomaly_pct = (
            anomaly_count / len(anomaly_df) * 100
            if len(anomaly_df) else 0
        )

        m1, m2, m3 = st.columns(3)
        with m1:
            show_metric("Records analysed", f"{len(anomaly_df):,}")
        with m2:
            show_metric("Flagged for review", f"{anomaly_count:,}")
        with m3:
            show_metric("Flagged share", f"{anomaly_pct:.1f}%")

        st.markdown(
            """
            <div class="info-note">
                <strong>Interpretation:</strong> an anomaly flag means a record
                differs from patterns learned or defined by the model. It is a
                review signal, not a fraud determination. Validate flagged records
                with business rules and authorised human review.
            </div>
            """,
            unsafe_allow_html=True,
        )

        amount_col = find_column(
            anomaly_df,
            [
                "claim_amount", "amount", "claimed_amount",
                "loss_amount", "predicted_claim_amount",
            ],
        )

        if amount_col:
            chart_data = anomaly_df.copy()
            chart_data["Claim amount"] = pd.to_numeric(
                chart_data[amount_col], errors="coerce"
            )
            chart_data["Review category"] = anomaly_mask.map(
                {
                    True: "Flagged for review",
                    False: "Typical",
                }
            )
            chart_data = chart_data.dropna(subset=["Claim amount"])

            if not chart_data.empty:
                fig = px.histogram(
                    chart_data,
                    x="Claim amount",
                    color="Review category",
                    barmode="overlay",
                    opacity=0.65,
                    nbins=30,
                    category_orders={
                        "Review category": [
                            "Typical",
                            "Flagged for review",
                        ]
                    },
                    labels={
                        "Claim amount": "Claim amount",
                        "count": "Number of records",
                    },
                    color_discrete_map={
                        "Typical": "#2563EB",
                        "Flagged for review": "#F97316",
                    },
                    hover_data=["Review category"],
                )
                fig.update_layout(
                    xaxis_title="Claim amount",
                    yaxis_title="Number of records",
                    legend_title_text="Model classification",
                    bargap=0.08,
                )
                add_plot_style(
                    fig,
                    "Claim amount distribution: typical vs flagged",
                    height=410,
                )
                show_chart(fig, key="ml_anomaly_distribution")

                st.caption(
                    "Overlapping histograms compare the claim-amount patterns "
                    "of flagged and typical records. Overlap is expected; "
                    "claim amount alone does not determine whether a record "
                    "is anomalous."
                )
            else:
                show_empty("No numeric claim amounts are available to plot.")
        else:
            show_empty(
                "No claim-amount field was found in the anomaly-results CSV."
            )

        st.markdown("### Records flagged for review")

        flagged_records = anomaly_df.loc[anomaly_mask].copy()

        if flagged_records.empty:
            st.success("No records were flagged by the loaded output.")
        else:
            st.dataframe(
                flagged_records.head(100),
                use_container_width=True,
                hide_index=True,
            )

        st.download_button(
            label="⬇ Download anomaly results (CSV)",
            data=anomaly_df.to_csv(index=False).encode("utf-8"),
            file_name="claim_anomaly_results.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_anomaly_results",
        )

    st.divider()
    st.markdown("### Claim-severity model evaluation")

    metrics, metrics_error = load_severity_metrics()

    metric_cols = st.columns(3)

    with metric_cols[0]:
        show_metric(
            "MAE",
            format_currency(metrics["mae"]) if "mae" in metrics else "Not available",
            "Mean Absolute Error · lower is generally better",
        )

    with metric_cols[1]:
        show_metric(
            "RMSE",
            format_currency(metrics["rmse"]) if "rmse" in metrics else "Not available",
            "Root Mean Squared Error · lower is generally better",
        )

    with metric_cols[2]:
        show_metric(
            "R²",
            f"{metrics['r2']:.4f}" if "r2" in metrics else "Not available",
            "Coefficient of determination · context matters",
        )

    if metrics_error:
        st.info(metrics_error)

    if metrics:
        metrics_display = pd.DataFrame(
            [
                {
                    "Metric": metric.upper() if metric != "r2" else "R²",
                    "Value": value,
                    "Interpretation": {
                        "mae": "Average absolute prediction error",
                        "rmse": "Error measure that penalises larger errors",
                        "r2": "Fit relative to a baseline; can be negative",
                    }[metric],
                }
                for metric, value in metrics.items()
            ]
        )
        st.dataframe(
            metrics_display,
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "Use metrics calculated on held-out evaluation data when assessing "
        "generalisation. Do not compare models trained or evaluated on "
        "different splits without checking the evaluation setup."
    )


# ============================================================
# 12. DATA QUALITY PAGE
# ============================================================

elif page == "Data Quality":
    show_banner(
        "Data Quality & Validation",
        "Inspect table completeness, missing values, duplicate records and "
        "key-field availability before using data for reporting or modelling.",
        "🔍",
    )

    tables = {
        "Customers": customers,
        "Policies": policies,
        "Premium payments": payments,
        "Claims": claims,
    }

    summary_rows = []

    for table_name, df in tables.items():
        missing_cells = int(df.isna().sum().sum()) if not df.empty else 0
        total_cells = int(df.shape[0] * df.shape[1]) if not df.empty else 0
        completeness = (
            (1 - missing_cells / total_cells) * 100
            if total_cells else 0.0
        )

        summary_rows.append(
            {
                "Table": table_name,
                "Rows": len(df),
                "Columns": len(df.columns),
                "Missing cells": missing_cells,
                "Completeness (%)": round(completeness, 2),
            }
        )

    summary = pd.DataFrame(summary_rows)

    st.markdown("### Dataset summary")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown("### Missing values by column")

    selected_table = st.selectbox(
        "Choose a table",
        options=list(tables.keys()),
        index=0,
        key="quality_table_selector",
    )

    selected_df = tables[selected_table]

    if selected_df.empty:
        show_empty(
            f"No data is available for {selected_table.lower()}. "
            "Check the database connection and table contents."
        )
    else:
        missing_count = selected_df.isna().sum()
        missing_count = missing_count[missing_count > 0].sort_values(
            ascending=True
        )

        if missing_count.empty:
            st.success(
                f"No missing cells were detected in the loaded "
                f"{selected_table.lower()} table."
            )
        else:
            missing_df = missing_count.rename(
                "Missing cells"
            ).reset_index()
            missing_df.columns = ["Column", "Missing cells"]
            missing_df["Missing (%)"] = (
                missing_df["Missing cells"] / len(selected_df) * 100
            ).round(2)
            missing_df = missing_df.sort_values(
                "Missing cells", ascending=True
            )

            fig = px.bar(
                missing_df,
                x="Missing cells",
                y="Column",
                orientation="h",
                text="Missing cells",
                hover_data=["Missing (%)"],
                labels={
                    "Missing cells": "Number of missing cells",
                    "Column": "Column name",
                },
                color="Missing (%)",
                color_continuous_scale=["#93C5FD", "#1D4ED8"],
            )
            fig.update_traces(
                textposition="outside",
                cliponaxis=False,
            )
            fig.update_layout(
                yaxis=dict(
                    tickfont=dict(color="#172033", size=11),
                    automargin=True,
                ),
                xaxis=dict(
                    tickfont=dict(color="#334155", size=11),
                    gridcolor="#E2E8F0",
                ),
                coloraxis_colorbar=dict(
                    title=dict(text="Missing (%)", font=dict(color="#334155")),
                    tickfont=dict(color="#334155"),
                ),
            )
            add_plot_style(
                fig,
                f"Missing values: {selected_table}",
                height=max(340, 45 * len(missing_df) + 100),
            )
            show_chart(fig, key="quality_missing_chart")

            st.dataframe(
                missing_df.sort_values(
                    "Missing cells", ascending=False
                ),
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("### Duplicate-record checks")

    duplicate_rows = []
    for table_name, df in tables.items():
        duplicate_count = int(df.duplicated().sum()) if not df.empty else 0
        duplicate_rows.append(
            {
                "Table": table_name,
                "Duplicate rows": duplicate_count,
                "Check status": (
                    "No exact duplicate rows"
                    if duplicate_count == 0
                    else "Review recommended"
                ),
            }
        )

    duplicate_summary = pd.DataFrame(duplicate_rows)
    st.dataframe(
        duplicate_summary,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "This check detects exact duplicate rows in the loaded data. "
        "It does not by itself establish whether business keys such as "
        "customer_id, policy_id or claim_id are unique."
    )

    st.markdown("### Key-field availability")

    key_checks = [
        ("Customers", customers, ["customer_id", "customerid", "client_id"]),
        ("Policies", policies, ["policy_id", "policyid", "policy_number"]),
        ("Premium payments", payments, ["payment_id", "paymentid", "receipt_id"]),
        ("Claims", claims, ["claim_id", "claimid", "claim_number"]),
    ]

    key_rows = []
    for table_name, df, candidates in key_checks:
        key_col = find_column(df, candidates)
        if key_col is None:
            status = "Key column not found"
            null_count = None
        else:
            null_count = int(df[key_col].isna().sum())
            status = "No null keys" if null_count == 0 else "Null keys found"

        key_rows.append(
            {
                "Table": table_name,
                "Key column": key_col or "—",
                "Null key values": null_count if null_count is not None else "—",
                "Status": status,
            }
        )

    st.dataframe(
        pd.DataFrame(key_rows),
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# 13. WAREHOUSE PAGE
# ============================================================

elif page == "Warehouse":
    show_banner(
        "Insurance Data Warehouse",
        "Browse the database tables, inspect loaded records and export synthetic "
        "data for controlled development, testing and analysis.",
        "🗄️",
    )

    table_options = {
        "Customers": customers,
        "Policies": policies,
        "Premium payments": payments,
        "Claims": claims,
    }

    inventory_rows = []
    for name, df in table_options.items():
        inventory_rows.append(
            {
                "Dataset": name,
                "Rows loaded": len(df),
                "Columns": len(df.columns),
                "Availability": "Loaded" if not df.empty else "Empty / unavailable",
            }
        )

    st.markdown("### Warehouse inventory")
    st.dataframe(
        pd.DataFrame(inventory_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Browse a dataset")

    selected_name = st.selectbox(
        "Select dataset",
        options=list(table_options.keys()),
        key="warehouse_table_selector",
    )

    selected_data = table_options[selected_name]

    if selected_data.empty:
        show_empty(
            f"The {selected_name.lower()} dataset is empty or unavailable."
        )
    else:
        filter_text = st.text_input(
            "Optional search across visible text",
            placeholder="For example, policy number or product name",
            key="warehouse_search",
        )

        display_data = selected_data.copy()

        if filter_text.strip():
            query = filter_text.strip().lower()
            text_columns = display_data.select_dtypes(
                include=["object", "string"]
            ).columns.tolist()

            if text_columns:
                matches = pd.Series(
                    False, index=display_data.index, dtype=bool
                )
                for column in text_columns:
                    matches |= (
                        display_data[column]
                        .astype("string")
                        .str.lower()
                        .str.contains(
                            re.escape(query),
                            na=False,
                            regex=True,
                        )
                    )
                display_data = display_data.loc[matches]
            else:
                display_data = display_data.iloc[0:0]

        st.caption(
            f"Showing {min(len(display_data), 500):,} of "
            f"{len(display_data):,} matching rows."
        )

        st.dataframe(
            display_data.head(500),
            use_container_width=True,
            hide_index=True,
        )

        safe_filename = re.sub(
            r"[^a-z0-9]+",
            "_",
            selected_name.lower(),
        ).strip("_")

        st.download_button(
            label=f"⬇ Download {selected_name} (CSV)",
            data=display_data.to_csv(index=False).encode("utf-8"),
            file_name=f"{safe_filename}.csv",
            mime="text/csv",
            use_container_width=True,
            key=f"download_{safe_filename}",
        )

    st.markdown("### Data access notes")
    st.markdown(
        """
        - The dashboard reads from the configured PostgreSQL database.
        - Table access is restricted to the four configured insurance tables.
        - Queries are intended for read-only reporting.
        - Downloaded CSVs contain the data currently loaded or filtered in the
          dashboard; handle exports according to your organisation's data policy.
        """
    )


# ============================================================
# 14. FOOTER
# ============================================================

show_footer()
