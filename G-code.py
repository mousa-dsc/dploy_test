import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import requests
import warnings
from io import BytesIO
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

warnings.filterwarnings('ignore')

# ── CONFIGURATION & SECRETS ────────────────────────────────────
st.set_page_config(page_title="Smart Analyzer Pro", page_icon="🧠", layout="wide")

# جلب المفتاح من Secrets لضمان الأمان في الـ Deployment
API_KEY = st.secrets.get("ANTHROPIC_API_KEY", "")
CLAUDE_URL = "https://api.anthropic.com/v1/messages"

# ── STYLING ──────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=Tajawal:wght@400;500;700&display=swap');
    .stApp { background:#040d18 !important; color: #caf0f8; }
    [data-testid="stSidebarContent"] { background:#060f1f !important; border-right:1px solid #0d2135; }
    h1, h2, h3, .hero-title { font-family: 'Syne', 'Tajawal', sans-serif; }
    .ai-box { background: rgba(0, 180, 216, 0.05); border: 1px solid #00b4d8; border-radius: 12px; padding: 20px; margin: 15px 0; }
</style>
""", unsafe_allow_html=True)

# ── DATA PROCESSING FUNCTIONS ────────────────────────────────
@st.cache_data
def load_and_clean_data(uploaded_file):
    name = uploaded_file.name.lower()
    try:
        if name.endswith('.csv'):
            df = pd.read_csv(uploaded_file, encoding='utf-8-sig')
        else:
            df = pd.read_excel(uploaded_file)
        
        # تنظيف أساسي
        df = df.dropna(how='all').dropna(axis=1, how='all')
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def get_ai_summary(df):
    """تجهيز ملخص بيانات مركز لـ Claude"""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    summary = {
        "dimensions": f"{df.shape[0]} rows x {df.shape[1]} columns",
        "numeric_features": num_cols[:15],
        "categorical_features": cat_cols[:15],
        "null_values": df.isnull().sum().to_dict(),
        "sample_data": df.head(5).to_dict(orient='records')
    }
    return json.dumps(summary, ensure_ascii=False)

# ── AI ENGINE ────────────────────────────────────────────────
def call_claude_api(prompt, system_prompt):
    if not API_KEY:
        return "⚠️ Anthropic API Key is missing. Please add it to Streamlit Secrets."
    
    headers = {
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    payload = {
        "model": "claude-3-sonnet-20240229",
        "max_tokens": 2000,
        "system": system_prompt,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    try:
        response = requests.post(CLAUDE_URL, headers=headers, json=payload, timeout=45)
        if response.status_code == 200:
            return response.json()['content'][0]['text']
        else:
            return f"API Error: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Connection error: {str(e)}"

# ── MAIN UI ──────────────────────────────────────────────────
st.title("🧠 Smart Data Analyzer Pro")
st.markdown("---")

with st.sidebar:
    st.header("📂 Data Input")
    uploaded = st.file_uploader("Upload Excel/CSV", type=['csv', 'xlsx'])
    lang = st.selectbox("Language", ["Arabic", "English"])
    st.info("This tool uses Claude AI to provide deep insights.")

if uploaded:
    df = load_and_clean_data(uploaded)
    
    if df is not None:
        tabs = st.tabs(["📈 Dashboard", "🤖 AI Insight", "🔬 ML Lab", "📄 Raw Data"])
        
        with tabs[0]: # Dashboard
            st.subheader("Data Overview")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", df.shape[0])
            c2.metric("Columns", df.shape[1])
            c3.metric("Numeric", len(df.select_dtypes(include=np.number).columns))
            c4.metric("Duplicates", df.duplicated().sum())
            
            # Simple Chart Builder
            st.markdown("### Quick Visualization")
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            if num_cols:
                target_col = st.selectbox("Select metric to visualize", num_cols)
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.histplot(df[target_col].dropna(), kde=True, color='#00b4d8')
                plt.title(f"Distribution of {target_col}")
                st.pyplot(fig)

        with tabs[1]: # AI Analysis
            st.subheader("AI-Powered Report")
            if st.button("Generate Smart Insights"):
                with st.spinner("Claude is analyzing your data structure..."):
                    data_json = get_ai_summary(df)
                    sys_p = "You are an expert Data Scientist. Provide deep, actionable business insights."
                    usr_p = f"Analyze this dataset: {data_json}. Language: {lang}. Provide: 1. Executive Summary 2. Top 3 Patterns found 3. Data quality recommendations."
                    
                    report = call_claude_api(usr_p, sys_p)
                    st.markdown(f'<div class="ai-box">{report}</div>', unsafe_allow_html=True)

        with tabs[2]: # ML Lab
            st.subheader("Automated Machine Learning")
            if len(num_cols) >= 2:
                target = st.selectbox("Select Target for Prediction", df.columns)
                if st.button("Run Auto-ML"):
                    # Basic Preprocessing
                    X = df.select_dtypes(include=np.number).drop(columns=[target], errors='ignore')
                    y = df[target]
                    
                    # Simple Label Encoding for target if object
                    if y.dtype == 'object':
                        le = LabelEncoder()
                        y = le.fit_transform(y.astype(str))
                        model = RandomForestClassifier()
                        task = "Classification"
                    else:
                        model = RandomForestRegressor()
                        task = "Regression"
                    
                    X_train, X_test, y_train, y_test = train_test_split(X.fillna(0), y, test_size=0.2)
                    model.fit(X_train, y_train)
                    score = model.score(X_test, y_test)
                    
                    st.success(f"Model Trained! {task} Score: {score:.2f}")
                    
                    # Feature Importance
                    fi = pd.Series(model.feature_importances_, index=X.columns).sort_values()
                    st.bar_chart(fi)
            else:
                st.warning("Need more numeric columns for ML.")

        with tabs[3]: # Raw Data
            st.dataframe(df)
else:
    st.info("Please upload a file to begin.")