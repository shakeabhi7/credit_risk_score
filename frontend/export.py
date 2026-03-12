import streamlit as st
import pandas as pd
from datetime import datetime, timezone

from src.monitoring.logger import setup_logger
from src.database.prediction_logger import PredictionLogger
from src.database.db_queries import DatabaseQueries

logger = setup_logger("streamlit_export")

# Page config

st.set_page_config(page_title="Export",layout="wide")

st.title("Export Predictions")
st.markdown("Download predictions records as CSV")

# Database Connection(cached)
@st.cache_resource
def get_queries():
    logger.info("Intializing database connection")
    pred_logger = PredictionLogger()
    return DatabaseQueries(pred_logger)

queries = get_queries()

# Connection Checks

if not queries.is_connected():
    st.error("Database not connected")
    logger.error("Database connection failed")
    st.stop()

st.success("Database connected")
st.markdown("---")

# filter form (prevents multiple reruns)

with st.form("export_form"):
    limit = st.slider("Number of record to export",
                      min_value=5,
                      max_value=100,
                      value=50
                      )
    generate_btn = st.form_submit_button("📥 Generate CSV", use_container_width=True)

# CSV Conversion Cache
@st.cache_data
def convert_to_csv(df):
    return df.to_csv(index=False)

#Generate CSV

if generate_btn:
    logger.info(f"Generating CSV export for {limit} records")

    with st.spinner("Generating CSV..."):
        df = queries.get_



