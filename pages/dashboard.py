import streamlit as st
import pandas as pd
import logging
from src.monitoring.logger import setup_logger
from src.database.prediction_logger import PredictionLogger
from src.database.db_queries import DatabaseQueries

logger = setup_logger('streamlit_dashboard')

st.set_page_config(page_title="Dashboard",layout="wide")

st.title("Dashboard")
st.markdown("View Statistic and recent prediction")

#Database Connection

@st.cache_resource
def get_queries():
    logger.info("Initializing database connection")
    pred_logger = PredictionLogger
    return DatabaseQueries(pred_logger)

queries = get_queries()

if not queries.is_connected():
    st.error("Database not connected")
    logger.error("Database connection failed")
    st.stop()

st.success("Database connected")

#Filters 
with st.form("dashboard_filters"):
    days = st.slider(
        "Last N Days",
        min_value=1,
        max_value=30,
        value=30
    )

    apply_btn = st.form_submit_button("Apply Filters")

st.markdown("---")
# Cached Data Loading
@st.cache_data(ttl=60)
def load_statistics(days):
    logger.info(f"Loading statistics for {days} days")
    return queries.get_statistics(days=days)

@st.cache_data(ttl=60)
def load_recent(days):
    logger.info(f"Loading recent predictions ({days} days)")
    return queries.get_recent_predictions(limit=10,days=days)

stats = load_statistics(days)

# Statsistics

if stats:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Predictions",stats["total_predictions"])

    with col2:
        st.metric(
            "Good Credit",
            f"{stats['good_credit']} ({stats['good_credit_percentage']}%)"
        )

    with col3:
        st.metric(
            "Bad Credit",
            f"{stats['bad_credit']} ({stats['bad_credit_percentage']}%)"
        )

    with col4:
        st.metric(
            "Avg Confidence",
            f"{stats['avg_confidence']:.2%}"
        )

    st.markdown("---")
    
    # Credit Distribution Chart

    with col1:
        st.write("** Credit Distribution**")
        chart_Data = pd.DataFrame({
            "Count": [
                    stats["good_credit"],
                    stats["bad_credit"]
                ]
        },
        index = ["Good Credit","Bad Credit"]
        )
        st.bar_chart(chart_Data,use_container_width=True)

    
    # Recent Predictions

    st.subheader("Recent Predictions")

    recent = load_recent(days)
    if recent:
        rows = []
        for pred in recent:
            input_data = pred.get("input_data",{})
            prediction = pred.get("prediction",{})
            metadata = pred.get("metadata",{})

            ts = metadata.get('timestamp')

            timestamp = (
                ts.strftime("%Y-%m-%d %H:%M:%S")
                if ts else "N/A"
            )

            rows.append({
                "Customer ID": input_data.get("customer_id", "N/A")[:20],
                "Status": "Good" if prediction.get("predicted_class") == 0 else "Bad",
                "Probability": f"{prediction.get('probability', 0):.2%}",
                "Confidence": f"{prediction.get('confidence', 0):.2%}",
                "Model": prediction.get("model_name", "N/A"),
                "Timestamp": timestamp
            })
        
        df = pd.DataFrame(rows)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            height=400
        )
    else:
        st.info("No Prediction found")

else:
    st.info("No statistics available")


#Sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.write(
"""
This dashboard shows prediction analytics from the credit risk model.

Features:
- Prediction statistics
- Credit distribution
- Recent predictions
- Model usage insights
"""
)