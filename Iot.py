import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="داشبورد IoT", layout="wide", page_icon="📊")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(BASE_DIR, 'dataset')
EDUCATION_DIR = os.path.join(DATASET_ROOT, 'education')
SENSING_DIR = os.path.join(DATASET_ROOT, 'sensing')

@st.cache_data
def load_and_process_data():
    if not os.path.exists(os.path.join(EDUCATION_DIR, 'grades.csv')):
        return None, f"پوشه دیتاست پیدا نشد. مسیر مورد جستجو:\n{DATASET_ROOT}"

    def get_student_list():
        try:
            df = pd.read_csv(os.path.join(EDUCATION_DIR, 'grades.csv'))
            df.columns = df.columns.str.strip()
            if 'uid' not in df.columns and 'u_id' in df.columns:
                 df.rename(columns={'u_id': 'uid'}, inplace=True)
            return df['uid'].unique()
        except: return []

    def extract_activity(uid):
        path = os.path.join(SENSING_DIR, 'activity', f'activity_{uid}.csv')
        if not os.path.exists(path): return 0.0
        try:
            df = pd.read_csv(path)
            col = 'activity inference' if 'activity inference' in df.columns else df.columns[1]
            total = len(df)
            active = len(df[df[col].isin([1, 2])])
            return active / total if total > 0 else 0.0
        except: return 0.0

    def extract_conversation(uid):
        path = os.path.join(SENSING_DIR, 'conversation', f'conversation_{uid}.csv')
        if not os.path.exists(path): return 0.0
        try:
            df = pd.read_csv(path)
            s_col = [c for c in df.columns if 'start' in c][0]
            e_col = [c for c in df.columns if 'end' in c][0]
            return (df[e_col] - df[s_col]).sum()
        except: return 0.0

    def extract_bluetooth(uid):
        path1 = os.path.join(SENSING_DIR, 'bluetooth', f'bt_{uid}.csv')
        path2 = os.path.join(SENSING_DIR, 'bluetooth', f'bluetooth_{uid}.csv')
        path = path1 if os.path.exists(path1) else path2
        if not os.path.exists(path): return 0
        try:
            df = pd.read_csv(path)
            if 'MAC' in df.columns: return df['MAC'].nunique()
            if len(df.columns) > 1: return df.iloc[:, 1].nunique()
            return 0
        except: return 0

    def extract_gps(uid):
        path = os.path.join(SENSING_DIR, 'gps', f'gps_{uid}.csv')
        if not os.path.exists(path): return 0.0
        try:
            df = pd.read_csv(path)
            if 'latitude' in df.columns and len(df) > 10:
                return (df['latitude'].var() + df['longitude'].var()) * 10000
            return 0.0
        except: return 0.0

    def get_piazza_score(uid, p_df):
        if p_df is None: return 0.0
        try:
            row = p_df[p_df['uid'] == uid]
            if row.empty: return 0.0
            d = row.iloc[0].get('days online', 0)
            v = row.iloc[0].get('views', 0)
            q = row.iloc[0].get('questions', 0)
            return d + (v * 0.05) + (q * 1.5)
        except: return 0.0

    uids = get_student_list()
    if len(uids) == 0: return None, "دانشجویی در فایل نمرات یافت نشد."

    try:
        p_df = pd.read_csv(os.path.join(EDUCATION_DIR, 'piazza.csv'))
        p_df.columns = p_df.columns.str.strip()
    except: p_df = None

    data = []
    for uid in uids:
        data.append({
            'uid': uid,
            'Activity (تحرک)': extract_activity(uid),
            'Conversation (مکالمه)': extract_conversation(uid),
            'Social (بلوتوث)': extract_bluetooth(uid),
            'Mobility (GPS)': extract_gps(uid),
            'Online (Piazza)': get_piazza_score(uid, p_df)
        })

    feat_df = pd.DataFrame(data)

    try:
        grade_df = pd.read_csv(os.path.join(EDUCATION_DIR, 'grades.csv'))
        grade_df.columns = grade_df.columns.str.strip()
        target_col = next((c for c in grade_df.columns if '13s' in c.lower() and 'gpa' in c.lower()), None)
        grade_df = grade_df.rename(columns={target_col: 'GPA', 'uid': 'uid'})
        grade_df['GPA'] = pd.to_numeric(grade_df['GPA'], errors='coerce')
        final_df = pd.merge(feat_df, grade_df[['uid', 'GPA']], on='uid')
        final_df = final_df[ (final_df['Conversation (مکالمه)'] > 0) | (final_df['Online (Piazza)'] > 0) ].dropna()
        return final_df, "Success"
    except Exception as e:
        return None, f"خطا در پردازش نمرات: {str(e)}"

df, status = load_and_process_data()

if df is None:
    st.error("❌ خطا:")
    st.text(status)
    st.stop()

FEATURES = ['Activity (تحرک)', 'Conversation (مکالمه)', 'Social (بلوتوث)', 'Mobility (GPS)', 'Online (Piazza)']
TARGET_THRESHOLD = 3.6

st.sidebar.title("کنترل پنل")
page = st.sidebar.radio("انتخاب صفحه:", 
    ["📊 نمای کلی و داده‌ها", 
     "📈 تحلیل رگرسیون (GPA)", 
     "⚖️ کلاسیفیکیشن (تشخیص خطر)", 
     "🧩 خوشه‌بندی رفتاری",
     "🔮 پیش‌بینی موردی (جدید)"]
)
st.sidebar.success(f"تعداد داده معتبر: {len(df)} دانشجو")

X = df[FEATURES]
y_cls = (df['GPA'] >= TARGET_THRESHOLD).astype(int)
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=FEATURES)
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_scaled, y_cls, test_size=0.3, random_state=42)

if page == "📊 نمای کلی و داده‌ها":
    st.title("داشبورد تحلیل داده‌های آموزشی (StudentLife)")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("میانگین نمرات (GPA)", f"{df['GPA'].mean():.2f}")
    k2.metric("میانگین فعالیت آنلاین", f"{df['Online (Piazza)'].mean():.0f}")
    k3.metric("میانگین مکالمات", f"{df['Conversation (مکالمه)'].mean()/60:.0f} min")
    k4.metric("تنوع مکانی (GPS)", f"{df['Mobility (GPS)'].mean():.2f}")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="GPA", nbins=10, title="توزیع نمرات دانشجویان", color_discrete_sequence=['#2ecc71'])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        corr = df[FEATURES + ['GPA']].corr()
        fig = px.imshow(corr, text_auto=True, title="همبستگی ویژگی‌ها با نمره", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)
    with st.expander("مشاهده جدول داده‌های خام"):
        st.dataframe(df)

elif page == "📈 تحلیل رگرسیون (GPA)":
    st.title("پیش‌بینی نمره با Random Forest")
    y = df['GPA']
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    m1, m2 = st.columns(2)
    m1.metric("خطای میانگین مربعات (MSE)", f"{mse:.4f}")
    m2.metric("دقت برازش (R2)", f"{r2:.4f}")
    st.subheader("چه عواملی بر نمره تاثیر دارند؟")
    imp_df = pd.DataFrame({'Feature': FEATURES, 'Importance': rf.feature_importances_}).sort_values('Importance')
    fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h', color='Importance', title="رتبه‌بندی اهمیت ویژگی‌ها")
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("نمره واقعی در مقابل پیش‌بینی")
    res_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    fig2 = px.scatter(res_df, x='Actual', y='Predicted', trendline='ols', title="خط برازش")
    st.plotly_chart(fig2, use_container_width=True)

elif page == "⚖️ کلاسیفیکیشن (تشخیص خطر)":
    st.title("تشخیص دانشجویان در معرض خطر")
    st.info(f"هدف: جداسازی دانشجویان ممتاز (GPA >= {TARGET_THRESHOLD}) از دانشجویان معمولی یا ضعیف.")
    if len(np.unique(yc_train)) > 1:
        svm = SVC(kernel='linear')
        svm.fit(Xc_train, yc_train)
        yc_pred = svm.predict(Xc_test)
        acc = accuracy_score(yc_test, yc_pred)
        st.metric("دقت مدل (Accuracy)", f"{acc*100:.1f}%")
        c1, c2 = st.columns(2)
        with c1:
            cm = confusion_matrix(yc_test, yc_pred)
            fig = px.imshow(cm, text_auto=True, title="ماتریس درهم‌ریختگی (Confusion Matrix)",
                            labels=dict(x="پیش‌بینی شده", y="واقعی"),
                            x=['در معرض خطر', 'ممتاز'], y=['در معرض خطر', 'ممتاز'],
                            color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("گزارش دقیق عملکرد")
            report = classification_report(yc_test, yc_pred, output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report).transpose().style.format("{:.2f}"))
    else:
        st.warning("داده‌های آموزش تنوع کافی برای تفکیک دو کلاس را ندارند.")

elif page == "🧩 خوشه‌بندی رفتاری":
    st.title("تحلیل الگوهای رفتاری (K-Means)")
    st.info("در این بخش دانشجویان بدون توجه به نمره، صرفاً بر اساس سبک زندگی گروه‌بندی می‌شوند.")
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    df['Cluster'] = clusters.astype(str)
    st.subheader("نمایش سه‌بعدی خوشه‌ها")
    fig = px.scatter_3d(df, x='Online (Piazza)', y='Conversation (مکالمه)', z='Mobility (GPS)',
                        color='Cluster', size='GPA', opacity=0.8,
                        title="خوشه‌بندی بر اساس (آنلاین، مکالمه، مکان)",
                        hover_data=['uid'])
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("ویژگی‌های هر خوشه (Cluster Profiles)")
    centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=FEATURES)
    centers.index.name = "Cluster ID"
    st.dataframe(centers.style.background_gradient(cmap='Greens'))

elif page == "🔮 پیش‌بینی موردی (جدید)":
    st.title("🔮 شبیه‌سازی و پیش‌بینی برای دانشجوی جدید")
    st.info("در این بخش می‌توانید داده‌های سنسوری فرضی وارد کنید و سیستم نمره و وضعیت خطر را پیش‌بینی کند.")
    y_reg = df['GPA']
    rf_full = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_full.fit(X_scaled, y_reg)
    col1, col2 = st.columns(2)
    with col1:
        in_activity = st.slider("🏃 میزان تحرک (0 تا 1)", 0.0, 1.0, float(df['Activity (تحرک)'].mean()))
        mean_conv = int(df['Conversation (مکالمه)'].mean())
        in_conversation = st.number_input("🗣 مجموع ثانیه‌های مکالمه (ترم)", 
                                          min_value=0, 
                                          max_value=200_000_000, 
                                          value=mean_conv, step=1000)
        in_bluetooth = st.number_input("👥 شاخص اجتماعی (تعداد دستگاه‌ها)", 
                                       min_value=0, 
                                       max_value=10_000, 
                                       value=int(df['Social (بلوتوث)'].mean()))
    with col2:
        mean_gps = float(df['Mobility (GPS)'].mean())
        in_gps = st.number_input("🌍 شاخص تنوع مکانی (GPS)", 
                                 min_value=0.0, 
                                 max_value=900_000_000.0, 
                                 value=mean_gps)
        in_online = st.number_input("💻 نمره فعالیت آنلاین (Piazza)", 
                                    min_value=0, 
                                    max_value=500_000, 
                                    value=int(df['Online (Piazza)'].mean()))
    if st.button("محاسبه وضعیت تحصیلی", type="primary"):
        input_data = pd.DataFrame([[in_activity, in_conversation, in_bluetooth, in_gps, in_online]], 
                                  columns=FEATURES)
        input_scaled = scaler.transform(input_data)
        pred_gpa = rf_full.predict(input_scaled)[0]
        st.markdown("---")
        st.subheader("نتیجه تحلیل هوشمند:")
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("معدل پیش‌بینی شده (GPA)", f"{pred_gpa:.2f}")
        if pred_gpa >= TARGET_THRESHOLD:
            res_col2.success(f"وضعیت: دانشجوی ممتاز (High Achiever)")
            st.balloons()
        else:
            res_col2.error(f"وضعیت: در معرض خطر افت (At-Risk)")
