import streamlit as st
import pandas as pd
import plotly.express as px

# Setting the page configuration
st.set_page_config(page_title="Store Performance Dashboard", layout="wide")

# Title and Description
st.title("📊 Store Performance Analytics Dashboard")
st.markdown("""
هذا الداشبورد يستعرض مؤشرات الأداء الرئيسية (KPIs) للمتاجر بناءً على بيانات المبيعات، 
مما يساعد في فهم مستويات التحصيل والاحتفاظ بالعملاء.
""")

# Load the dataset
@st.cache_data
def load_data():
    # تأكد من وضع ملف الـ CSV في نفس مجلد الكود عند عمل الـ Deploy
    df = pd.read_csv('Store_Performance_KPIs.xlsx.xlsx - KPIs.csv')
    return df

df = load_data()

# --- Sidebar Filters ---
st.sidebar.header("Filter Options")
# اختيار المتجر بناءً على الـ Store ID
store_ids = df['Store ID'].unique()
selected_store = st.sidebar.selectbox("Select Store ID", options=["All Stores"] + list(store_ids))

if selected_store != "All Stores":
    filtered_df = df[df['Store ID'] == selected_store]
else:
    filtered_df = df

# --- Key Metrics (KPIs) ---
st.subheader("Key Performance Indicators (KPIs)")
col1, col2, col3, col4 = st.columns(4)

# حساب المتوسطات للعرض
avg_fulfillment = filtered_df['Order Fulfillment Rate (%)'].str.rstrip('%').astype(float).mean()
avg_cancellation = filtered_df['Order Cancellation Rate (%)'].str.rstrip('%').astype(float).mean()
avg_retention = filtered_df['Customer Retention Rate (%)'].str.rstrip('%').astype(float).mean()
total_orders = filtered_df['Total Orders'].sum()

with col1:
    st.metric("Total Orders", f"{total_orders:,}")
with col2:
    st.metric("Avg Fulfillment", f"{avg_fulfillment:.2f}%")
with col3:
    st.metric("Avg Cancellation", f"{avg_cancellation:.2f}%")
with col4:
    st.metric("Avg Retention", f"{avg_retention:.2f}%")

st.divider()

# --- Visualizations ---

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Top Stores by Average Order Value ($)")
    # ترتيب المتاجر حسب قيمة الطلب
    top_stores = df.nlargest(10, 'Average Order Value ($)')
    fig_bar = px.bar(top_stores, x='Store ID', y='Average Order Value ($)', 
                     title="Top 10 Stores by AOV", color='Average Order Value ($)')
    st.plotly_chart(fig_bar, use_container_width=True)

with col_right:
    st.subheader("Fulfillment vs Cancellation")
    # تحويل النسب إلى أرقام للرسم البياني
    plot_df = filtered_df.copy()
    plot_df['Fulfillment'] = plot_df['Order Fulfillment Rate (%)'].str.rstrip('%').astype(float)
    plot_df['Cancellation'] = plot_df['Order Cancellation Rate (%)'].str.rstrip('%').astype(float)
    
    fig_scatter = px.scatter(plot_df, x='Fulfillment', y='Cancellation', 
                             size='Total Orders', hover_name='Store ID',
                             title="Correlation: Fulfillment vs Cancellation")
    st.plotly_chart(fig_scatter, use_container_width=True)

# --- Raw Data Table ---
st.subheader("Raw Data Preview")
st.dataframe(filtered_df, use_container_width=True)