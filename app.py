import streamlit as st
import requests
from src.monitoring.logger import setup_logger
import os

logger = setup_logger(__name__)

#CONFIGURATION

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Credit Risk Scoring System",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

def check_api_status():
    """Check FastAPI health status"""

    try:
        response = requests.get(f"{API_URL}/health",timeout=3)
        if response.status_code==200:
            return "Running"
        return "Error"
    except requests.exceptions.RequestException as e:
        logger.warning(f"API health check failed:{e}")
        return "Offline"
    
@st.cache_resource
def get_db_status():
    """Check MongoDB connection"""
    try:
        from src.database.prediction_logger import PredictionLogger
        logger.info("Checking Database Connection")
        pred_logger = PredictionLogger()

        if pred_logger.connected:
            status = "Connected"
        else:
            status = "Not Connected"

        pred_logger.close()
        return status
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        return "Error"

#Custom CSS

st.markdown("""
<style>
.main-title {
    font-size: 2.5em;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
}

.subtitle {
    font-size: 1.2em;
    color: #555;
    text-align: center;
}

.status-card {
    padding: 10px;
    border-radius: 8px;
    background-color: #f5f5f5;
}
</style>
""", unsafe_allow_html=True)

#HOME PAGE

st.markdown(
    '<h1 class="main-title">💳 Credit Risk Scoring System</h1>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">ML-Powered Credit Risk Assessment Platform</p>',
    unsafe_allow_html=True
)

st.markdown("---")

#SYSTEM METRICS

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "📊 Model Accuracy",
        "85%+",
        "+3% improvement"
    )

with col2:
    st.metric(
        "⚡ Prediction Time",
        "~50ms",
        "Real-time"
    )

with col3:
    st.metric(
        "💾 Database",
        "Ready",
        "Operational"
    )

st.markdown("---")

#FEATURES

st.subheader("✨ Available Features")

col1, col2 = st.columns(2)

with col1:
    st.write("""
### 🔮 New Prediction
- Real-time credit risk assessment
- Input customer financial data
- Instant prediction with confidence score
- Automatic customer ID generation
- Database logging for audit trail
""")

with col2:
    st.write("""
### 📊 Dashboard
- View statistics and metrics
- Recent predictions history
- Visualization of credit distribution
- Performance analytics
- Time-based filtering
""")

col1, col2 = st.columns(2)

with col1:
    st.write("""
### 🔍 Search by Customer ID
- Find specific customer predictions
- Complete prediction details
- Input features + engineered features
- Probability & confidence score
""")

with col2:
    st.write("""
### 📥 Export Functionality
- Download predictions as CSV
- Customizable record limits
- Full prediction details included
- Ready for analysis
""")

st.markdown("---")

#TECHNOLOGY STACK

st.subheader("🛠️ Technology Stack")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("""
**Backend**
- FastAPI
- Python 3.10+
- Scikit-learn
- XGBoost
- TensorFlow
""")

with col2:
    st.write("""
**Frontend**
- Streamlit
- Interactive UI
- Real-time predictions
- Visualization support
""")

with col3:
    st.write("""
**Database**
- MongoDB
- Prediction logging
- History tracking
- Aggregated statistics
""")

st.markdown("---")

#ML MODELS

st.subheader("🤖 ML Models Used")

tab1, tab2, tab3 = st.tabs([
    "Random Forest",
    "XGBoost (Production)",
    "Neural Network"
])

with tab1:
    st.write("""
**Random Forest**

- Ensemble of decision trees
- Good feature importance
- Interpretable results

Accuracy: **84.5%**
AUC-ROC: **0.86**
""")

with tab2:
    st.write("""
**XGBoost (Production Model)** ⭐

- Gradient boosting algorithm
- Highly optimized
- Fast inference

Accuracy: **85.2%**
AUC-ROC: **0.87**
""")

with tab3:
    st.write("""
**Neural Network**

Architecture:
- 3 Hidden Layers
- ReLU activation
- Adam optimizer

Accuracy: **83.8%**
AUC-ROC: **0.84**
""")

st.markdown("---")

#RISK INTERPRETATION

st.subheader("🎯 Understanding Prediction Results")

col1, col2 = st.columns(2)

with col1:
    st.write("""
**Prediction Classes**

✅ **Good Credit**
- Low default risk

⚠️ **Bad Credit**
- Higher lending risk
""")

with col2:
    st.write("""
**Confidence Score**

0.85+ → High confidence  
0.70–0.85 → Medium  
<0.70 → Low confidence
""")

st.markdown("---")

# SIDEBAR

st.sidebar.markdown("System Status")

api_status = check_api_status()
db_status = get_db_status()

st.sidebar.write(f"**API Status:** {api_status}")
st.sidebar.write(f"**Database:** {db_status}")

st.sidebar.markdown("---")
st.sidebar.write("""
**Version:** 1.1.0  
**Built With:**  
Streamlit • FastAPI • MongoDB
""")

logger.info("Streamlit home page loaded successfully")