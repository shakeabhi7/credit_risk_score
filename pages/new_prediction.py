import streamlit as st
import requests
import logging
import os
from src.monitoring.logger import setup_logger

logger = setup_logger('streamlit_new_prediction')
st.set_page_config(
    page_title="New Prediction",
    layout="wide"
)

st.title("Credit Risk Prediction")
st.markdown("Enter Customer details")

# API CONFIG
API_URL = os.getenv("API_URL","http://localhost:8000")

@st.cache_data(ttl=10)
def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{API_URL}/health",timeout=5)
        return response.status_code==200
    except requests.exceptions.RequestException as e:
        logger.error(f"API health check failed: {e}")
        return False

api_running = check_api_health()

if not api_running:
    st.error("API Server is not running!")
    st.info("Start the API server with:")
    st.stop()

st.success("API server is running")

#Form layout
with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age",min_value=18,max_value=99,value=35)
        debt = st.number_input("Total Debt(USD)",min_value=0,value=15000,step=1000)
        employment_years = st.number_input("Employment Years",min_value=0,max_value=50,value=8)
    
    with col2:
        income = st.number_input(
            "Annual Income (USD)",
            min_value=1000,
            value=75000,
            step=1000
        )

        credit_limit = st.number_input(
            "Credit Limit (USD)",
            min_value=1000,
            value=50000,
            step=1000
        )

        employment_type = st.selectbox(
            "Employment Type",
            ["Salaried", "Self-employed", "Freelance", "Student"]
        )
    with col3:
        credit_used = st.number_input("Credit Used(USD)",min_value=0,value=20000,step=1000)

    st.markdown("---")

    submit_btn = st.form_submit_button("Predict")

#Prediction

if submit_btn:
    if credit_used > credit_limit:
        st.error("Credit used cannot exceed credit limit")
    
    elif employment_years>age:
        st.error("Employment years cannot exceed age")
    else:
        with st.spinner("Making prediction..."):
            try:
                payload = {
                    "age":age,
                    "income":income,
                    "debt": debt,
                    "credit_limit":credit_limit,
                    "credit_used":credit_used,
                    "employment_years":employment_years,
                    'employment_type': employment_type
                }

                response = requests.post(f"{API_URL}/predict",json=payload,timeout=20)

                if response.status_code != 200:
                    st.error(f"API Error: {response.status_code}")
                    st.write(response.text)
                    st.stop()
                result = response.json()

                st.success("Prediction Complete!")

                #Result

                st.markdown("---")
                st.subheader("Prediction Result")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    status = "Good" if result["predicted_class"] == 0 else "Bad"
                    st.metric("Credit Status",status)

                with col2:
                    probability = result["probability"]
                    st.metric("Probability", f"{probability:.2%}")

                with col3:
                    confidence = result["confidence"]
                    st.metric("Confidence",f"{confidence:.2f}")
                
                with col4:
                    st.metric("Model Used",result["model_name"])

                # progress bar
                st.progress(float(probability))

                # Details

                st.markdown("---")
                st.subheader("Detailed Information")

                info_col1, info_col2 = st.columns(2)

                with info_col1:
                    st.write("**Customer ID:**",result["customer_id"])
                    st.write("**Assessment:**",result["message"])

                    st.write("**Risk Indicators:")

                    debt_to_income = debt / (income + 1)
                    credit_utilization = credit_used/(credit_limit + 1)

                    st.write(f"• Debt-to-Income: {debt_to_income:.2%}")
                    st.write(f"• Credit Utilization: {credit_utilization:.2%}")
                
                with info_col2:
                    st.json({
                        "customer_id": result["customer_id"],
                        "prediction": result["predicted_class"],
                        "probability": round(probability, 4),
                        "confidence": round(confidence, 4),
                        "model": result["model_name"]
                    })
                
                st.success("Prediction stored in database")
            except requests.exceptions.Timeout:
                st.error("API request timed out")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API server")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                logger.error(f"Prediction error: {e}",exc_info=True)

# Sidebar

st.sidebar.markdown("---")
st.sidebar.subheader("About")

st.sidebar.write(
"""
This Credit Risk Scoring system uses machine learning to assess credit risk.

**Features**
- Real-time ML predictions
- Multiple model ensemble
- Automatic customer ID generation
- Database logging

**Models Used**
- Random Forest
- XGBoost
- Neural Network (MLP)
"""
)

st.sidebar.markdown("---")

st.sidebar.write(
    "**API Status:** Running"
    if api_running
    else "**API Status:** Offline"
)

