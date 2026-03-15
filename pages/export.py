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
    generate_btn = st.form_submit_button("📥 Generate CSV", width="stretch")

# CSV Conversion Cache
@st.cache_data
def convert_to_csv(df):
    return df.to_csv(index=False)

#Generate CSV

if generate_btn:
    logger.info(f"Generating CSV export for {limit} records")

    with st.spinner("Generating CSV..."):
        df = queries.get_predictions_dataframe(limit=limit)
        if df.empty:
            st.info("No Records found in database")
            logger.warning("No records found for export")

        else:
            st.success(f"Generated {len(df)} records")

            st.markdown("---")

            #PReview
            st.subheader("Preview")

            st.dataframe(
                df,
                width="stretch",
                hide_index=True,
                height=400
            )
            st.markdown("---")

            #Download

            st.subheader("📥 Download")

            csv = convert_to_csv(df)
            filename = f"predictions_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.csv"

            st.download_button(
                label = "📥 Download CSV",
                data=csv,
                file_name = filename,
                mime="text/csv",
                width="stretch"
            )
            logger.info(f"CSV file prepared: {filename}")

            st.markdown("---")

            # Export Summary

            st.subheader("Export Summary")
            col1, col2, col3 = st.columns(3)

            total_records = len(df)

            good_count = (df["Predicted_Class"]=="Good").sum()
            bad_count = (df["Predicted_Class"]=="Bad").sum()
            with col1:
                st.metric("Total Records",total_records)
            with col2:
                st.metric("Good Credit",good_count)
            with col3:
                st.metric("Bad Credit",bad_count)
            logger.info(f"Export summary = Total: {total_records}, Good: {good_count}, Bad: {bad_count}")

# Sidebar

st.sidebar.markdown("---")
st.sidebar.subheader("About")

st.sidebar.write(
"""
Export prediction data as CSV.

Includes:
• Customer information  
• Input data  
• Engineered features  
• Prediction results  
• Timestamp  

Use cases:
• Data analysis  
• Reporting  
• Backup  
• Further ML processing
"""
)


