import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import requests
import warnings
from io import BytesIO
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# ── الإعدادات العامة ──────────────────────────────────────────
st.set_page_config(page_title="Smart Analyzer Pro", page_icon="🧠", layout="wide")

# جلب مفتاح API من Secrets لضمان الأمان
# تأكد من إضافة ANTHROPIC_API_KEY في إعدادات Streamlit Cloud
API_KEY = st.secrets.get("ANTHROPIC_API_KEY", "")
CLAUDE_URL = "https://api.anthropic.com/v1/messages"

# ── التنسيق الجمالي (CSS) ──────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700&family=Tajawal:wght@400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Tajawal', sans-serif; }
    .main { background-color: #040d18; }
    .stMetric { background: #071828; padding: 15px; border-radius: 10px; border: 1px solid #0d3558; }
    .ai-box { 
        background: linear-gradient(135deg, #071828, #050f1c); 
        border-left: 5px solid #00b4d8; 
        padding: 20px; 
        border-radius: 10px; 
        margin: 15px 0;
        color: #caf0f8;
    }
</style>
""", unsafe_allow_html=True)

# ── دالات معالجة البيانات ──────────────────────────────────────
@st.cache_data
def load_data(file):
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file, encoding='utf-8-sig')
        return pd.read_excel(file)
    except Exception as e:
        st.error(f"خطأ في تحميل الملف: {e}")
        return None

def get_data_profile(df):
    """تجهيز ملخص تقني للموديل"""
    profile = {
        "rows": df.shape[0],
        "cols": df.columns.tolist(),
        "numeric_summary": df.describe().to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "sample": df.head(3).to_dict()
    }
    return json.dumps(profile, ensure_ascii=False)

# ── محرك الذكاء الاصطناعي ──────────────────────────────────────
def ask_claude(prompt):
    if not API_KEY:
        return "⚠️ مفتاح API غير موجود. فضلاً أضفه في Secrets."
    
    headers = {
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    payload = {
        "model": "claude-3-sonnet-20240229",
        "max_tokens": 1500,
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        resp = requests.post(CLAUDE_URL, headers=headers, json=payload, timeout=30)
        return resp.json()['content'][0]['text']
    except:
        return "⚠️ حدث خطأ في الاتصال بكلود."

# ── واجهة المستخدم ──────────────────────────────────────────
st.title("🧠 المحلل الذكي | Smart Analyzer")
st.write("ارفع ملف البيانات الخاص بك واحصل على تحليل فوري باستخدام الذكاء الاصطناعي")

uploaded_file = st.sidebar.file_uploader("اختر ملف Excel أو CSV", type=['csv', 'xlsx'])

if uploaded_file:
    df = load_data(uploaded_file)
    
    if df is not None:
        # إحصائيات سريعة
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("عدد الصفوف", df.shape[0])
        col2.metric("عدد الأعمدة", df.shape[1])
        col3.metric("القيم المفقودة", df.isnull().sum().sum())
        col4.metric("الأعمدة الرقمية", len(df.select_dtypes(include=np.number).columns))

        tabs = st.tabs(["📊 التحليل البصري", "🤖 تحليل كلود", "🔬 التنبؤ الآلي (ML)"])

        with tabs[0]:
            st.subheader("توزيع البيانات")
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            if num_cols:
                selected_col = st.selectbox("اختر عموداً للرسم:", num_cols)
                fig, ax = plt.subplots()
                sns.histplot(df[selected_col], kde=True, ax=ax, color="#00b4d8")
                st.pyplot(fig)
            else:
                st.warning("لا توجد أعمدة رقمية للرسم.")

        with tabs[1]:
            if st.button("🚀 اطلب من كلود تحليل البيانات"):
                with st.spinner("جاري التحليل..."):
                    summary = get_data_profile(df)
                    prompt = f"حلل هذه البيانات بالعربي وأعطني أهم 3 استنتاجات و3 توصيات تجارية: {summary}"
                    res = ask_claude(prompt)
                    st.markdown(f'<div class="ai-box">{res}</div>', unsafe_allow_html=True)

        with tabs[2]:
            st.subheader("نموذج تنبؤ سريع (Random Forest)")
            if len(num_cols) > 1:
                target = st.selectbox("اختر الهدف (Target):", num_cols)
                features = [c for c in num_cols if c != target]
                
                if st.button("تدريب الموديل"):
                    X = df[features].fillna(0)
                    y = df[target].fillna(df[target].mean())
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
                    
                    model = RandomForestRegressor().fit(X_train, y_train)
                    score = model.score(X_test, y_test)
                    st.success(f"دقة النموذج (R² Score): {score:.2f}")
                    
                    fi = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
                    st.bar_chart(fi.set_index('Feature'))
            else:
                st.info("تحتاج لأكثر من عمود رقمي لتشغيل التنبؤ.")
else:
    st.info("بانتظار رفع الملف...")