"""
Smart Excel Analyzer — Powered by Claude AI
Upload any Excel/CSV file and get instant AI analysis
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
import requests
warnings.filterwarnings('ignore')

from io import BytesIO
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

# ── Config ────────────────────────────────────────────────────
st.set_page_config(page_title="Smart Analyzer", page_icon="🧠", layout="wide",
                   initial_sidebar_state="expanded")

DARK='#040d18'; CARD='#071828'; A1='#00b4d8'; A2='#0077b6'
TXT='#caf0f8'; MUT='#4a8fa8'; GREEN='#2a9d8f'; RED='#e63946'; ORANGE='#f4a261'

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@400;500&family=Tajawal:wght@400;500;700&display=swap');
*, html, body { box-sizing:border-box; }
.stApp,.main { background:#040d18 !important; }
[class*="css"] { font-family:'Tajawal','Syne',sans-serif; }
[data-testid="stSidebarContent"] { background:linear-gradient(180deg,#060f1f,#040d18) !important; border-right:1px solid #0d2135; }

.hero { background:linear-gradient(135deg,#040d18,#071828,#040d18); border:1px solid #0d3558; border-radius:20px; padding:2.5rem 3rem; margin-bottom:2rem; position:relative; overflow:hidden; }
.hero::before { content:''; position:absolute; top:-50%;left:-50%;width:200%;height:200%; background:radial-gradient(ellipse at 60% 40%,rgba(0,180,216,.07),transparent 60%); }
.hero-label { font-family:'DM Mono',monospace; font-size:.7rem; letter-spacing:.25em; color:#00b4d8; text-transform:uppercase; margin-bottom:.6rem; }
.hero-title { font-family:'Syne',sans-serif; font-size:2.8rem; font-weight:800; background:linear-gradient(135deg,#caf0f8,#00b4d8,#0077b6); -webkit-background-clip:text; -webkit-text-fill-color:transparent; line-height:1.1; }
.hero-desc { color:#4a8fa8; font-size:.95rem; margin-top:.8rem; }

.kpi-grid { display:grid; grid-template-columns:repeat(5,1fr); gap:.8rem; margin:1.2rem 0; }
.kpi-card { background:linear-gradient(135deg,#071828,#050f1c); border:1px solid #0d3558; border-radius:14px; padding:1.2rem 1.4rem; position:relative; overflow:hidden; }
.kpi-card::after { content:''; position:absolute; top:0;left:0;right:0;height:2px; background:linear-gradient(90deg,#00b4d8,#0077b6); }
.kpi-val { font-family:'DM Mono',monospace; font-size:1.8rem; font-weight:500; color:#caf0f8; line-height:1; }
.kpi-lbl { font-size:.68rem; color:#4a8fa8; text-transform:uppercase; letter-spacing:.1em; margin-top:.4rem; }

.sec-hdr { display:flex; align-items:center; gap:.7rem; margin:2rem 0 1rem; padding-bottom:.6rem; border-bottom:1px solid #0d3558; }
.sec-hdr-text { font-family:'Syne',sans-serif; font-size:1.05rem; font-weight:700; color:#caf0f8; }
.sec-hdr-line { flex:1; height:1px; background:linear-gradient(90deg,#0d3558,transparent); }

.ai-box { background:linear-gradient(135deg,#071828,#060f1c); border:1px solid #0d3558; border-left:4px solid #00b4d8; border-radius:14px; padding:1.5rem 1.8rem; margin:1rem 0; }
.ai-box h4 { color:#00b4d8; font-family:'Syne',sans-serif; font-size:1rem; margin-bottom:.8rem; }
.ai-box p,.ai-box li { color:#8ecae6; font-size:.9rem; line-height:1.7; }
.ai-box strong { color:#caf0f8; }

.col-tag { display:inline-block; padding:3px 10px; border-radius:20px; font-family:'DM Mono',monospace; font-size:.72rem; margin:3px; }
.tag-num  { background:rgba(0,180,216,.12); border:1px solid #00b4d8; color:#00b4d8; }
.tag-cat  { background:rgba(42,157,143,.12); border:1px solid #2a9d8f; color:#2a9d8f; }
.tag-date { background:rgba(244,162,97,.12); border:1px solid #f4a261; color:#f4a261; }

.upload-area { background:#071828; border:2px dashed #0d3558; border-radius:20px; padding:3rem 2rem; text-align:center; }
.stTabs [data-baseweb="tab-list"] { background:#060f1f; border-radius:12px; padding:4px; gap:4px; border:1px solid #0d3558; }
.stTabs [data-baseweb="tab"] { border-radius:8px; color:#4a8fa8; font-family:'Syne',sans-serif; font-weight:600; font-size:.82rem; }
.stTabs [aria-selected="true"] { background:linear-gradient(135deg,#0d3558,#0a2540) !important; color:#00b4d8 !important; }
</style>
""", unsafe_allow_html=True)

def theme():
    plt.rcParams.update({'figure.facecolor':DARK,'axes.facecolor':CARD,'axes.edgecolor':'#0d3558',
        'text.color':TXT,'axes.labelcolor':MUT,'xtick.color':MUT,'ytick.color':MUT,
        'grid.color':'#0d3558','grid.alpha':.5,'axes.spines.top':False,'axes.spines.right':False})
theme()

# ══════════════════════════════════════════════════════════════
# CLAUDE API CALL
# ══════════════════════════════════════════════════════════════
def ask_claude(prompt: str, system: str = "") -> str:
    """Call Claude API for AI analysis"""
    try:
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1500,
            "system": system or "You are an expert data analyst. Respond in Arabic when the user writes in Arabic, English otherwise. Be concise, insightful, and actionable.",
            "messages": [{"role": "user", "content": prompt}]
        }
        resp = requests.post("https://api.anthropic.com/v1/messages",
                             headers=headers, json=payload, timeout=30)
        if resp.status_code == 200:
            return resp.json()['content'][0]['text']
        else:
            return f"⚠️ API Error {resp.status_code}"
    except Exception as e:
        return f"⚠️ Connection error: {str(e)[:80]}"

# ══════════════════════════════════════════════════════════════
# DATA HELPERS
# ══════════════════════════════════════════════════════════════
def detect_col_types(df):
    num_cols  = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols  = df.select_dtypes(include=['object','category']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    # Try to parse object cols as dates
    for col in cat_cols[:]:
        try:
            pd.to_datetime(df[col], infer_datetime_format=True)
            date_cols.append(col); cat_cols.remove(col)
        except: pass
    return num_cols, cat_cols, date_cols

def get_data_summary(df, num_cols, cat_cols):
    """Build a compact data summary for Claude"""
    summary = {
        "rows": len(df), "cols": len(df.columns),
        "numeric_cols": num_cols[:10],
        "categorical_cols": cat_cols[:10],
        "missing_pct": round(df.isnull().mean().mean() * 100, 1),
        "duplicates": int(df.duplicated().sum()),
    }
    if num_cols:
        desc = df[num_cols[:8]].describe().round(2)
        summary["numeric_stats"] = desc.to_dict()
    if cat_cols:
        summary["top_categories"] = {
            col: df[col].value_counts().head(3).to_dict()
            for col in cat_cols[:5]
        }
    return json.dumps(summary, ensure_ascii=False, default=str)

def load_file(f):
    name = f.name.lower()
    if name.endswith('.csv'):
        # Try different encodings
        for enc in ['utf-8','utf-8-sig','latin-1','cp1256']:
            try:
                f.seek(0)
                return pd.read_csv(f, encoding=enc)
            except: pass
    elif name.endswith(('.xlsx','.xls')):
        return pd.read_excel(BytesIO(f.read()), sheet_name=0)
    raise ValueError("صيغة غير مدعومة")

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<div style='padding:.8rem 0 .3rem'><div style='font-family:Syne,sans-serif;font-size:1.2rem;font-weight:800;color:#caf0f8'>🧠 Smart Analyzer</div><div style='font-family:DM Mono,monospace;font-size:.6rem;color:#4a8fa8;letter-spacing:.15em'>AI-POWERED · ANY DATA</div></div>", unsafe_allow_html=True)
    st.markdown("---")

    uploaded = st.file_uploader("📂 ارفع ملفك هنا", type=['xlsx','xls','csv'],
                                 help="Excel أو CSV — أي بيانات")

    if uploaded:
        st.markdown(f"""
        <div style='background:#071828;border:1px solid #0d3558;border-radius:10px;padding:.8rem;margin:.5rem 0;font-size:.8rem'>
            <div style='color:#caf0f8;font-weight:600'>📄 {uploaded.name}</div>
            <div style='color:#4a8fa8;font-family:DM Mono,monospace;font-size:.7rem'>{uploaded.size/1024:.1f} KB</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**⚙️ Analysis Options**")
    show_raw    = st.checkbox("عرض البيانات الخام", value=False)
    auto_clean  = st.checkbox("تنظيف تلقائي للبيانات", value=True)
    lang        = st.selectbox("🌐 لغة التحليل", ["العربية 🇸🇦", "English 🇺🇸"])
    st.markdown("---")
    st.markdown("<div style='color:#4a8fa8;font-size:.72rem'>يدعم: Excel · CSV<br>حد الحجم: 50MB<br>Powered by Claude AI</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-label">◆ AI-Powered Data Analysis · Any Excel or CSV</div>
    <div class="hero-title">🧠 Smart Data Analyzer</div>
    <div class="hero-desc">ارفع أي ملف Excel أو CSV وكلود يحلله لك فوراً — إحصاء · رسوم بيانية · ML · توصيات ذكية</div>
</div>
""", unsafe_allow_html=True)

if not uploaded:
    st.markdown("""
    <div class="upload-area">
        <div style="font-size:4rem;margin-bottom:1rem">📊</div>
        <div style="font-family:Syne,sans-serif;font-size:1.4rem;font-weight:800;color:#caf0f8;margin-bottom:.5rem">ارفع أي ملف بيانات</div>
        <div style="color:#4a8fa8;font-size:.9rem;margin-bottom:1.5rem">Excel (.xlsx / .xls) · CSV — أي نوع بيانات</div>
        <div style="display:flex;justify-content:center;gap:1rem;flex-wrap:wrap">
            <span style="background:rgba(0,180,216,.08);border:1px solid #00b4d8;color:#00b4d8;padding:6px 16px;border-radius:20px;font-size:.8rem">📈 مبيعات</span>
            <span style="background:rgba(42,157,143,.08);border:1px solid #2a9d8f;color:#2a9d8f;padding:6px 16px;border-radius:20px;font-size:.8rem">👥 عملاء</span>
            <span style="background:rgba(244,162,97,.08);border:1px solid #f4a261;color:#f4a261;padding:6px 16px;border-radius:20px;font-size:.8rem">🏭 عمليات</span>
            <span style="background:rgba(0,180,216,.08);border:1px solid #00b4d8;color:#00b4d8;padding:6px 16px;border-radius:20px;font-size:.8rem">💰 مالية</span>
            <span style="background:rgba(42,157,143,.08);border:1px solid #2a9d8f;color:#2a9d8f;padding:6px 16px;border-radius:20px;font-size:.8rem">🚚 لوجستيك</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════
# LOAD & CLEAN
# ══════════════════════════════════════════════════════════════
try:
    with st.spinner("📂 جاري تحميل الملف..."):
        df_raw = load_file(uploaded)
        df = df_raw.copy()
except Exception as e:
    st.error(f"❌ خطأ في تحميل الملف: {e}")
    st.stop()

if auto_clean:
    # Drop fully empty rows/cols
    df = df.dropna(how='all').dropna(axis=1, how='all')
    # Try to convert object cols to numeric where possible
    for col in df.select_dtypes(include='object').columns:
        converted = pd.to_numeric(df[col].astype(str).str.replace(',','').str.strip(), errors='coerce')
        if converted.notna().mean() > 0.7:
            df[col] = converted

num_cols, cat_cols, date_cols = detect_col_types(df)

# ══════════════════════════════════════════════════════════════
# QUICK STATS
# ══════════════════════════════════════════════════════════════
missing_pct = df.isnull().mean().mean() * 100
dups = df.duplicated().sum()

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card"><div class="kpi-val">{df.shape[0]:,}</div><div class="kpi-lbl">Rows</div></div>
    <div class="kpi-card"><div class="kpi-val">{df.shape[1]}</div><div class="kpi-lbl">Columns</div></div>
    <div class="kpi-card"><div class="kpi-val">{len(num_cols)}</div><div class="kpi-lbl">Numeric Cols</div></div>
    <div class="kpi-card"><div class="kpi-val">{missing_pct:.1f}%</div><div class="kpi-lbl">Missing Data</div></div>
    <div class="kpi-card"><div class="kpi-val">{dups:,}</div><div class="kpi-lbl">Duplicates</div></div>
</div>
""", unsafe_allow_html=True)

# Column type tags
st.markdown("**🏷️ Column Types:**", unsafe_allow_html=False)
tags_html = ""
for c in num_cols:  tags_html += f'<span class="col-tag tag-num">📊 {c}</span>'
for c in cat_cols:  tags_html += f'<span class="col-tag tag-cat">🏷️ {c}</span>'
for c in date_cols: tags_html += f'<span class="col-tag tag-date">📅 {c}</span>'
st.markdown(tags_html, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════
tabs = st.tabs(["🤖 AI Analysis", "📊 Charts", "📋 Statistics", "🔍 Deep Dive", "💬 Ask Anything"])

# ── TAB 1: AI ANALYSIS ────────────────────────────────────────
with tabs[0]:
    st.markdown('<div class="sec-hdr"><span style="font-size:1.2rem">🤖</span><span class="sec-hdr-text">Claude AI Analysis</span><div class="sec-hdr-line"></div></div>', unsafe_allow_html=True)

    if st.button("🚀 حلل البيانات بالذكاء الاصطناعي", use_container_width=True):
        summary = get_data_summary(df, num_cols, cat_cols)
        ar = "Arabic" if "العربية" in lang else "English"

        with st.spinner("🧠 كلود يحلل بياناتك..."):
            # Overall analysis
            prompt1 = f"""
You are analyzing a dataset with these stats:
{summary}

File name: {uploaded.name}
Language: Respond in {ar}

Provide:
1. **وصف البيانات** — وش هذا الملف على الأرجح؟
2. **أهم 3 ملاحظات** من الإحصاءات
3. **مشاكل البيانات** (missing, outliers, inconsistencies)
4. **3 أسئلة تحليلية مقترحة** يمكن الإجابة عليها

Be specific and data-driven. Use the actual column names and numbers.
"""
            analysis = ask_claude(prompt1)
            st.markdown(f'<div class="ai-box"><h4>📊 التحليل الأولي</h4><p>{analysis.replace(chr(10), "<br>")}</p></div>', unsafe_allow_html=True)

            # Recommendations
            prompt2 = f"""
Based on this dataset summary:
{summary}

File: {uploaded.name}
Language: {ar}

Give 4 specific, actionable business recommendations based on the data patterns you see.
Format as numbered list. Be concrete with the actual numbers from the data.
"""
            recs = ask_claude(prompt2)
            st.markdown(f'<div class="ai-box"><h4>💡 توصيات ذكية</h4><p>{recs.replace(chr(10), "<br>")}</p></div>', unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="background:#071828;border:1px solid #0d3558;border-radius:14px;padding:2rem;text-align:center;color:#4a8fa8">
            <div style="font-size:2.5rem;margin-bottom:.8rem">🤖</div>
            <div style="color:#caf0f8;font-weight:600;margin-bottom:.5rem">اضغط الزر لبدء التحليل</div>
            <div style="font-size:.85rem">كلود سيحلل بياناتك ويعطيك insights وتوصيات مخصصة</div>
        </div>
        """, unsafe_allow_html=True)

# ── TAB 2: CHARTS ─────────────────────────────────────────────
with tabs[1]:
    st.markdown('<div class="sec-hdr"><span style="font-size:1.2rem">📊</span><span class="sec-hdr-text">Auto-Generated Charts</span><div class="sec-hdr-line"></div></div>', unsafe_allow_html=True)

    if num_cols:
        # Distributions
        n = min(len(num_cols), 6)
        cols_to_plot = num_cols[:n]
        fig, axes = plt.subplots(2, 3, figsize=(14, 7)) if n > 3 else plt.subplots(1, n, figsize=(14, 4))
        fig.patch.set_facecolor(DARK)
        axes_flat = axes.flat if hasattr(axes, 'flat') else [axes] if n == 1 else axes

        for i, (ax, col) in enumerate(zip(axes_flat, cols_to_plot)):
            data = df[col].dropna().clip(upper=df[col].quantile(.99))
            ax.hist(data, bins=35, color=A1, alpha=.85, edgecolor=DARK)
            ax.axvline(df[col].median(), color=RED, ls='--', lw=1.5, label=f'Med:{df[col].median():.1f}')
            ax.set_title(col, fontsize=10, fontweight='bold', color=TXT)
            ax.legend(fontsize=7); ax.grid(True, alpha=.3)

        if n <= 3:
            for ax in list(axes_flat)[n:]: ax.set_visible(False)

        plt.tight_layout(pad=1.5)
        st.pyplot(fig); plt.close()

        # Correlation heatmap (if enough numeric cols)
        if len(num_cols) >= 3:
            st.markdown('<div class="sec-hdr"><span style="font-size:1.2rem">🔥</span><span class="sec-hdr-text">Correlation Matrix</span><div class="sec-hdr-line"></div></div>', unsafe_allow_html=True)
            corr_cols = num_cols[:10]
            fig2, ax2 = plt.subplots(figsize=(min(12, len(corr_cols)*1.2+2), 5))
            fig2.patch.set_facecolor(DARK)
            corr = df[corr_cols].corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='Blues',
                        linewidths=.5, ax=ax2, annot_kws={'size':8})
            ax2.set_facecolor(CARD)
            plt.tight_layout(); st.pyplot(fig2); plt.close()

    # Categorical bar charts
    if cat_cols:
        st.markdown('<div class="sec-hdr"><span style="font-size:1.2rem">🏷️</span><span class="sec-hdr-text">Categorical Columns</span><div class="sec-hdr-line"></div></div>', unsafe_allow_html=True)
        cat_to_show = [c for c in cat_cols if df[c].nunique() <= 30][:4]
        if cat_to_show:
            fig3, axes3 = plt.subplots(1, len(cat_to_show), figsize=(14, 5))
            fig3.patch.set_facecolor(DARK)
            if len(cat_to_show) == 1: axes3 = [axes3]
            for ax, col in zip(axes3, cat_to_show):
                vc = df[col].value_counts().head(10)
                ax.barh(vc.index.astype(str), vc.values, color=GREEN, alpha=.85, edgecolor=DARK)
                ax.set_title(col, fontsize=10, fontweight='bold', color=TXT)
                ax.invert_yaxis(); ax.grid(True, alpha=.3, axis='x')
            plt.tight_layout(); st.pyplot(fig3); plt.close()

    # Custom chart builder
    st.markdown('<div class="sec-hdr"><span style="font-size:1.2rem">🎨</span><span class="sec-hdr-text">Custom Chart Builder</span><div class="sec-hdr-line"></div></div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: chart_type = st.selectbox("Chart Type", ["Bar","Line","Scatter","Box","Pie"])
    with c2: x_col = st.selectbox("X Axis", df.columns.tolist())
    with c3: y_col = st.selectbox("Y Axis", num_cols if num_cols else df.columns.tolist())
    with c4: color_col = st.selectbox("Color by", ["None"] + cat_cols)

    if st.button("🎨 رسم الـ Chart", use_container_width=True):
        fig4, ax4 = plt.subplots(figsize=(12, 5))
        fig4.patch.set_facecolor(DARK); ax4.set_facecolor(CARD)
        try:
            if chart_type == "Bar":
                if df[x_col].nunique() <= 30:
                    grp = df.groupby(x_col)[y_col].mean().sort_values(ascending=False).head(20)
                    ax4.bar(range(len(grp)), grp.values, color=A1, edgecolor=DARK)
                    ax4.set_xticks(range(len(grp))); ax4.set_xticklabels(grp.index.astype(str), rotation=35, ha='right', fontsize=9)
            elif chart_type == "Line":
                ax4.plot(df[x_col].head(200), df[y_col].head(200), color=A1, lw=2)
            elif chart_type == "Scatter":
                if color_col != "None" and color_col in df.columns:
                    for cat, grp in df.groupby(color_col):
                        ax4.scatter(grp[x_col], grp[y_col], alpha=.5, s=15, label=str(cat)[:20])
                    ax4.legend(fontsize=8, loc='best')
                else:
                    ax4.scatter(df[x_col], df[y_col], alpha=.4, color=A1, s=12)
            elif chart_type == "Box":
                if df[x_col].nunique() <= 15:
                    data_box = [df[df[x_col]==cat][y_col].dropna() for cat in df[x_col].unique()[:10]]
                    ax4.boxplot(data_box, labels=[str(x)[:12] for x in df[x_col].unique()[:10]],
                                patch_artist=True, boxprops=dict(facecolor=A2, color=A1))
                    plt.xticks(rotation=30, ha='right', fontsize=9)
            elif chart_type == "Pie":
                if df[x_col].nunique() <= 10:
                    vc = df[x_col].value_counts().head(8)
                    ax4.pie(vc.values, labels=vc.index.astype(str), autopct='%1.1f%%',
                            colors=plt.cm.Blues(np.linspace(0.4, 0.9, len(vc))),
                            wedgeprops=dict(edgecolor=DARK, linewidth=1.5))
            ax4.set_title(f'{y_col} by {x_col}', color=TXT, fontsize=12, fontweight='bold')
            ax4.grid(True, alpha=.3)
            plt.tight_layout(); st.pyplot(fig4); plt.close()
        except Exception as e:
            st.warning(f"⚠️ تعذّر رسم هذا الـ chart: {e}")
            plt.close()

# ── TAB 3: STATISTICS ─────────────────────────────────────────
with tabs[2]:
    st.markdown('<div class="sec-hdr"><span style="font-size:1.2rem">📋</span><span class="sec-hdr-text">Detailed Statistics</span><div class="sec-hdr-line"></div></div>', unsafe_allow_html=True)

    if num_cols:
        desc = df[num_cols].describe().T.round(3)
        desc['missing'] = df[num_cols].isnull().sum()
        desc['missing%'] = (df[num_cols].isnull().mean() * 100).round(1)
        desc['skew'] = df[num_cols].skew().round(3)
        st.dataframe(desc.style.background_gradient(cmap='Blues', subset=['mean','std']), use_container_width=True)

    if cat_cols:
        st.markdown('<div class="sec-hdr"><span style="font-size:1.2rem">🏷️</span><span class="sec-hdr-text">Categorical Summary</span><div class="sec-hdr-line"></div></div>', unsafe_allow_html=True)
        cat_summary = pd.DataFrame({
            'Column': cat_cols,
            'Unique Values': [df[c].nunique() for c in cat_cols],
            'Most Common': [str(df[c].mode()[0]) if not df[c].mode().empty else 'N/A' for c in cat_cols],
            'Missing %': [f"{df[c].isnull().mean()*100:.1f}%" for c in cat_cols],
        })
        st.dataframe(cat_summary.set_index('Column'), use_container_width=True)

    # Missing data heatmap
    if df.isnull().any().any():
        st.markdown('<div class="sec-hdr"><span style="font-size:1.2rem">❓</span><span class="sec-hdr-text">Missing Data Map</span><div class="sec-hdr-line"></div></div>', unsafe_allow_html=True)
        miss_cols = df.columns[df.isnull().any()].tolist()[:15]
        fig5, ax5 = plt.subplots(figsize=(12, 4))
        fig5.patch.set_facecolor(DARK)
        miss_pct = df[miss_cols].isnull().mean() * 100
        ax5.bar(range(len(miss_pct)), miss_pct.values, color=[RED if v>20 else ORANGE if v>5 else GREEN for v in miss_pct.values])
        ax5.set_xticks(range(len(miss_pct))); ax5.set_xticklabels(miss_pct.index, rotation=35, ha='right', fontsize=9)
        ax5.set_title('Missing Data %', color=TXT, fontsize=12, fontweight='bold')
        ax5.axhline(20, color=RED, ls='--', lw=1, label='20% threshold')
        ax5.legend(fontsize=9); ax5.grid(True, alpha=.3, axis='y')
        plt.tight_layout(); st.pyplot(fig5); plt.close()

    if show_raw:
        st.markdown('<div class="sec-hdr"><span style="font-size:1.2rem">📄</span><span class="sec-hdr-text">Raw Data</span><div class="sec-hdr-line"></div></div>', unsafe_allow_html=True)
        st.dataframe(df.head(100), use_container_width=True)

# ── TAB 4: DEEP DIVE / ML ─────────────────────────────────────
with tabs[3]:
    st.markdown('<div class="sec-hdr"><span style="font-size:1.2rem">🔍</span><span class="sec-hdr-text">ML-Powered Deep Dive</span><div class="sec-hdr-line"></div></div>', unsafe_allow_html=True)

    if len(num_cols) >= 2:
        st.markdown("**🎯 اختر العمود الهدف (Target) للتنبؤ:**")
        c1, c2 = st.columns(2)
        with c1:
            target = st.selectbox("Target Column", num_cols + cat_cols)
        with c2:
            task = st.selectbox("Task Type", ["Auto Detect", "Regression", "Classification"])

        if st.button("🤖 تدريب نموذج ML", use_container_width=True):
            # Auto-detect task
            is_classification = (task == "Classification") or \
                                 (task == "Auto Detect" and (df[target].nunique() <= 10 or df[target].dtype == 'object'))

            try:
                # Prepare features
                feature_cols = [c for c in num_cols if c != target][:10]
                if not feature_cols:
                    st.warning("⚠️ لا يوجد columns عددية كافية")
                else:
                    df_ml = df[feature_cols + [target]].dropna()
                    X = df_ml[feature_cols]

                    if is_classification:
                        le_target = LabelEncoder()
                        y = le_target.fit_transform(df_ml[target].astype(str))
                        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, random_state=42)
                        model = RandomForestClassifier(100, random_state=42, n_jobs=-1)
                        model.fit(X_tr, y_tr)
                        score = accuracy_score(y_te, model.predict(X_te))
                        metric_name = "Accuracy"
                    else:
                        y = df_ml[target]
                        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.2, random_state=42)
                        model = RandomForestRegressor(100, random_state=42, n_jobs=-1)
                        model.fit(X_tr, y_tr)
                        score = r2_score(y_te, model.predict(X_te))
                        metric_name = "R² Score"

                    # Feature importance
                    fi = pd.DataFrame({'Feature': feature_cols, 'Importance': model.feature_importances_}).sort_values('Importance')

                    # Show results
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                        <div class="ai-box">
                            <h4>🎯 Model Results</h4>
                            <p><strong>Target:</strong> {target}<br>
                            <strong>Task:</strong> {"Classification" if is_classification else "Regression"}<br>
                            <strong>{metric_name}:</strong> {score:.4f} ({score*100:.1f}%)<br>
                            <strong>Training Samples:</strong> {len(X_tr):,}<br>
                            <strong>Features Used:</strong> {len(feature_cols)}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        fig6, ax6 = plt.subplots(figsize=(6, 4))
                        fig6.patch.set_facecolor(DARK); ax6.set_facecolor(CARD)
                        ax6.barh(fi['Feature'], fi['Importance'],
                                 color=[A1 if v == fi['Importance'].max() else A2 for v in fi['Importance']],
                                 edgecolor=DARK)
                        ax6.set_title('Feature Importance', color=TXT, fontsize=11, fontweight='bold')
                        ax6.grid(True, alpha=.3, axis='x')
                        plt.tight_layout(); st.pyplot(fig6); plt.close()

            except Exception as e:
                st.error(f"⚠️ خطأ في التدريب: {e}")
    else:
        st.info("⚠️ يحتاج على الأقل عمودين عدديين للـ ML analysis")

    # Outlier detection
    if num_cols:
        st.markdown('<div class="sec-hdr"><span style="font-size:1.2rem">🎯</span><span class="sec-hdr-text">Outlier Detection</span><div class="sec-hdr-line"></div></div>', unsafe_allow_html=True)
        outlier_col = st.selectbox("اختر عمود:", num_cols, key='outlier_col')
        col_data = df[outlier_col].dropna()
        q1, q3 = col_data.quantile(.25), col_data.quantile(.75)
        iqr = q3 - q1
        outliers = col_data[(col_data < q1-1.5*iqr) | (col_data > q3+1.5*iqr)]

        fig7, ax7 = plt.subplots(figsize=(12, 3))
        fig7.patch.set_facecolor(DARK); ax7.set_facecolor(CARD)
        ax7.scatter(range(len(col_data)), col_data, alpha=.3, color=A1, s=8, label='Normal')
        if len(outliers) > 0:
            ax7.scatter(outliers.index if hasattr(outliers,'index') else range(len(outliers)),
                        outliers.values, color=RED, s=25, zorder=5, label=f'Outliers ({len(outliers)})')
        ax7.set_title(f'Outliers in {outlier_col}', color=TXT, fontsize=11, fontweight='bold')
        ax7.legend(fontsize=9); ax7.grid(True, alpha=.3)
        plt.tight_layout(); st.pyplot(fig7); plt.close()
        st.markdown(f'<div class="ai-box"><h4>📊 Outlier Summary</h4><p>عدد الـ outliers: <strong>{len(outliers):,}</strong> ({len(outliers)/len(col_data)*100:.1f}% من البيانات)<br>النطاق الطبيعي: <strong>{q1-1.5*iqr:.2f}</strong> → <strong>{q3+1.5*iqr:.2f}</strong></p></div>', unsafe_allow_html=True)

# ── TAB 5: ASK ANYTHING ──────────────────────────────────────
with tabs[4]:
    st.markdown('<div class="sec-hdr"><span style="font-size:1.2rem">💬</span><span class="sec-hdr-text">Ask Claude About Your Data</span><div class="sec-hdr-line"></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="ai-box"><h4>💡 أمثلة على الأسئلة</h4><p>• وش العمود الأهم في البيانات؟<br>• وش الأنماط الغريبة اللي تشوفها؟<br>• كيف أحسن جودة البيانات؟<br>• وش العلاقة بين X وY؟<br>• اقترح خطوات تحليل إضافية</p></div>', unsafe_allow_html=True)

    user_q = st.text_area("❓ سؤالك:", placeholder="اكتب سؤالك هنا عن بياناتك...", height=100)

    if st.button("🚀 اسأل كلود", use_container_width=True) and user_q:
        summary = get_data_summary(df, num_cols, cat_cols)
        ar = "Arabic" if "العربية" in lang else "English"
        prompt = f"""
Data summary:
{summary}

File: {uploaded.name}
Columns: {list(df.columns)}
Sample data (first 3 rows):
{df.head(3).to_string()}

User question: {user_q}
Language: {ar}

Answer specifically based on the actual data provided. Reference column names and real numbers.
"""
        with st.spinner("🧠 كلود يفكر..."):
            answer = ask_claude(prompt)
        st.markdown(f'<div class="ai-box"><h4>🤖 إجابة كلود</h4><p>{answer.replace(chr(10), "<br>")}</p></div>', unsafe_allow_html=True)

    # Chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
