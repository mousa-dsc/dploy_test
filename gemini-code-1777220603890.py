import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# إعدادات الصفحة
st.set_page_config(page_title="Quick Data Analyzer", layout="wide")

st.title("📊 Quick Data Analyzer")
st.write("ارفع ملف Excel أو CSV لاستعراض الإحصائيات والرسوم البيانية فوراً")

# رفع الملف
uploaded_file = st.file_uploader("اختر ملفك", type=['csv', 'xlsx'])

if uploaded_file:
    # قراءة البيانات
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.success("✅ تم تحميل الملف بنجاح")
        
        # عرض البيانات
        with st.expander("👀 استعراض البيانات الخام"):
            st.dataframe(df)

        # إحصائيات سريعة
        st.subheader("📋 ملخص سريع")
        c1, c2, c3 = st.columns(3)
        c1.metric("عدد الصفوف", df.shape[0])
        c2.metric("عدد الأعمدة", df.shape[1])
        c3.metric("القيم المفقودة", df.isnull().sum().sum())

        # الرسوم البيانية
        st.subheader("📈 التحليل البصري")
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if num_cols:
            col_to_plot = st.selectbox("اختر عموداً لرسمه:", num_cols)
            
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.histplot(df[col_to_plot].dropna(), kde=True, ax=ax, color='#00b4d8')
            ax.set_title(f"Distribution of {col_to_plot}")
            st.pyplot(fig)
        else:
            st.warning("الملف لا يحتوي على أعمدة رقمية للرسم البياني.")

    except Exception as e:
        st.error(f"حدث خطأ أثناء قراءة الملف: {e}")
else:
    st.info("💡 ارفع ملفاً من جهازك لتجربة التطبيق.")