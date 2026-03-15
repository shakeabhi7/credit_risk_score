import streamlit as st
from datetime import datetime

from src.monitoring.logger import setup_logger
from src.database.prediction_logger import PredictionLogger
from src.database.db_queries import DatabaseQueries

logger = setup_logger("streamlit_search")

# Page Config
st.set_page_config(page_title="Search by ID",page_icon="🔍", layout="wide")

st.title("Search Prediction by Customer ID")
st.markdown("Find and view complete prediction details")

# Database Connection (cached)

@st.cache_resource
def get_queries():
    logger.info("initializing database connection")
    pred_logger= PredictionLogger()
    return DatabaseQueries(pred_logger)
queries = get_queries()

# connection check
if not queries.is_connected():
    st.error("Database not connected")
    logger.error("Database connection failed")
    st.stop()

st.success("Database Connected")

st.markdown("---")

#Search from (prevents unnecessary reruns)

with st.form("search_form"):
    col1, col2 = st.columns([4,1])

    with col1:
        customer_id = st.text_input(
            "Enter Customer ID",
            placeholder="CUST_1708017930_a7f3c9e1"
        )
    with col2:
        search_btn = st.form_submit_button("🔍 Search", width="stretch")

#Input Validation

if search_btn:
    if not customer_id:
        st.warning("Please enter a Customer ID")
        st.stop()
    if not customer_id.startswith("CUST_"):
        st.warning("Invalid Customer ID")
        st.stop()
    
    logger.info(f"Searching prediction for {customer_id}")

    prediction = queries.search_by_customer_id(customer_id)

    if not prediction:
        st.warning("No Prediction found for this customer ID")
        logger.warning(f"No prediction found for {customer_id}")
        st.stop()
    
    st.success("Prediction found!")

    # Metadata
    st.subheader("Metadata")

    metadata = prediction.get("metadata",{})

    ts = metadata.get("timestamp")

    timestamp = (
        ts.strftime("%Y-%m-%d %H:%M:%S")
        if isinstance(ts, datetime)
        else ts
    )
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(f"**Timestamp:** {timestamp}")

    with col2:
        st.write(f"**Model Version:** {metadata.get('model_version', 'N/A')}")

    with col3:
        st.write(f"**API Version:** {metadata.get('api_version', 'N/A')}")

    st.markdown("---")

    # Input Data

    st.subheader("Input Data")

    input_data = prediction.get("input_data", {})

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(f"**Age:** {input_data.get('age', 'N/A')}")
        st.write(f"**Income:** ${input_data.get('income', 0):,}")
        st.write(f"**Employment Type:** {input_data.get('employment_type', 'N/A')}")

    with col2:
        st.write(f"**Debt:** ${input_data.get('debt', 0):,}")
        st.write(f"**Credit Limit:** ${input_data.get('credit_limit', 0):,}")
        st.write(f"**Employment Years:** {input_data.get('employment_years', 'N/A')}")

    with col3:
        st.write(f"**Credit Used:** ${input_data.get('credit_used', 0):,}")

    st.markdown("---")

    # Engineered features

    st.subheader(" Engineered Features")

    engineered = prediction.get("engineered_features", {})

    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Debt-to-Income:** {engineered.get('debt_to_income', 0):.4f}")
        st.write(f"**Credit Utilization:** {engineered.get('credit_utilization', 0):.4f}")

    with col2:
        st.write(f"**Income per Age:** {engineered.get('income_per_age', 0):.2f}")
        st.write(f"**Employment Stability:** {engineered.get('employment_stability', 0):.4f}")

    st.markdown("---")
     
    # Prediction Result
    
    st.subheader(" Prediction Result")

    pred = prediction.get("prediction", {})

    col1, col2, col3, col4 = st.columns(4)

    status = "Good ✅" if pred.get("predicted_class") == 0 else "Bad ⚠️"

    with col1:
        st.metric("Credit Status", status)

    with col2:
        st.metric("Probability", f"{pred.get('probability', 0):.2%}")

    with col3:
        st.metric("Confidence", f"{pred.get('confidence', 0):.2%}")

    with col4:
        st.metric("Model Used", pred.get("model_name", "N/A"))

    st.markdown("---")
    
    # Preprocessing Info

    st.subheader(" Preprocessing Info")

    preprocess = prediction.get("preprocessing_info", {})

    st.write(f"**Scaler Type:** {preprocess.get('scaler_type', 'N/A')}")

    features_used = preprocess.get("features_used", [])

    st.write(f"**Features Used:** {len(features_used)} features")

    st.markdown("---")

    # Full JSON

    with st.expander("View Full JSON"):
        st.json(prediction)


#sidebar

st.sidebar.markdown("---")
st.sidebar.subheader("About")

st.sidebar.write(
"""
Search for specific predictions by Customer ID.

View complete details:
• Input data  
• Engineered features  
• Prediction result  
• Metadata & timestamps
"""
)