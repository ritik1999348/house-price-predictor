import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)
st.markdown("""
<style>

/* Main App */
.stApp {
    background-image: url("https://images.unsplash.com/photo-1600585154526-990dced4db0d");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Dark Overlay */
.main::before {
    content: "";
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: rgba(0,0,0,0.35);
    z-index: -1;
}

/* Title */
h1 {
    color: white !important;
    font-size: 55px !important;
    font-weight: 800 !important;
}

/* Subheader */
h3 {
    color: white !important;
    font-size: 40px !important;
    font-weight: 700 !important;
}

/* Labels */
label {
    color: white !important;
    font-weight: 600 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-size: 18px;
    font-weight: 600;
    color: white;
    background: rgba(255,255,255,0.15);
    border-radius: 10px;
    padding: 10px 20px;
    margin-right: 10px;
}

/* Input Boxes */
.stSelectbox div[data-baseweb="select"],
.stNumberInput input {
    background-color: rgba(255,255,255,0.92) !important;
    border-radius: 15px !important;
    border: none !important;
    padding: 8px !important;
}

/* Slider */
.stSlider {
    color: #ff4b8b !important;
}

/* Predict Button */
.stButton>button {
    width: 100%;
    border-radius: 40px;
    height: 60px;
    border: none;
    font-size: 24px;
    font-weight: bold;
    color: white;
    background: linear-gradient(to right, #6a11cb, #ff4b8b);
    box-shadow: 0px 4px 15px rgba(0,0,0,0.4);
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.03);
    background: linear-gradient(to right, #ff4b8b, #6a11cb);
}

/* Glass Effect Container */
.block-container {
    background: rgba(255,255,255,0.12);
    padding: 2rem;
    border-radius: 25px;
    backdrop-filter: blur(10px);
}

/* Metrics */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.15);
    border-radius: 20px;
    padding: 15px;
    color: white;
}

/* Hide Streamlit Menu/Footer */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

import os

BASE_DIR = os.getcwd()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, 'house_price.csv.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'best_model.pkl')
FEAT_PATH = os.path.join(BASE_DIR, 'feature_cols.pkl')

print(DATA_PATH)
print(os.path.exists(DATA_PATH))

@st.cache_resource
def load_model():
    model       = joblib.load(MODEL_PATH)
    feat_cols   = joblib.load(FEAT_PATH)
    return model, feat_cols

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

model, feature_cols = load_model()
df = load_data()

st.title("🏠 House Price Prediction")

tab1, tab2, tab3 = st.tabs(["🔮 Predict Price", "📊 Data Analysis", "🤖 Model Info"])


with tab1:
    st.subheader("Property details bharo")
    col1, col2, col3 = st.columns(3)

    with col1:
        posted_by     = st.selectbox("Posted By", ['Owner','Dealer','Builder'])
        bhk_no        = st.slider("BHK Number", 1, 6, 2)
        bhk_type      = st.selectbox("Type", ['BHK','RK'])

    with col2:
        square_ft     = st.number_input("Square Feet", 300, 10000, 1000)
        under_const   = st.selectbox("Under Construction?", ['No','Yes'])
        ready_to_move = st.selectbox("Ready to Move?", ['Yes','No'])

    with col3:
        rera   = st.selectbox("RERA Approved?", ['Yes','No'])
        resale = st.selectbox("Resale Property?", ['No','Yes'])
        lat    = st.number_input("Latitude",  12.0, 28.0, 13.0)
        lon    = st.number_input("Longitude", 72.0, 88.0, 77.5)

    if st.button("💰 Price Predict ", type="primary"):
        posted_map = {'Owner':0, 'Dealer':1, 'Builder':2}
        bhk_map    = {'BHK':0, 'RK':1}

        input_data = {col: 0 for col in feature_cols}
        input_data['POSTED_BY_ENC']      = posted_map.get(posted_by, 0)
        input_data['BHK_NO.']            = bhk_no
        input_data['BHK_OR_RK_ENC']      = bhk_map.get(bhk_type, 0)
        input_data['SQUARE_FT']          = square_ft
        input_data['UNDER_CONSTRUCTION'] = 1 if under_const == 'Yes' else 0
        input_data['READY_TO_MOVE']      = 1 if ready_to_move == 'Yes' else 0
        input_data['RERA']               = 1 if rera == 'Yes' else 0
        input_data['RESALE']             = 1 if resale == 'Yes' else 0
        input_data['LATITUDE']           = lat
        input_data['LONGITUDE']          = lon

        X_input  = pd.DataFrame([input_data])[feature_cols]
        log_pred = model.predict(X_input)[0]
        price    = np.expm1(log_pred)

        st.success(f"## 🏠 Predicted Price: ₹ {price:.1f} Lakhs")
        low, high = price * 0.90, price * 1.10
        st.info(f"📊 Expected Range: ₹{low:.1f}L — ₹{high:.1f}L")

        ca, cb, cc = st.columns(3)
        ca.metric("Price/SqFt", f"₹{(price*100000/square_ft):.0f}")
        cb.metric("BHK Config",  f"{bhk_no} {bhk_type}")
        cc.metric("Area",        f"{square_ft} sq.ft")

with tab2:
    st.subheader("Dataset Analysis")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Median Price",  f"₹{df['TARGET(PRICE_IN_LACS)'].median():.0f}L")
    c3.metric("Avg Sq.Ft",    f"{df['SQUARE_FT'].mean():.0f}")
    c4.metric("Features",     "12")

    col_a, col_b = st.columns(2)
    with col_a:
        fig1 = px.histogram(
            df[df['TARGET(PRICE_IN_LACS)'] <= 500],
            x='TARGET(PRICE_IN_LACS)', nbins=50,
            title="Price Distribution",
            color_discrete_sequence=['#4ECDC4']
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        bhk_p = df.groupby('BHK_NO.')['TARGET(PRICE_IN_LACS)'].median().reset_index()
        fig2  = px.bar(bhk_p, x='BHK_NO.', y='TARGET(PRICE_IN_LACS)',
                       title="Median Price by BHK",
                       color_discrete_sequence=['#FF6B6B'])
        st.plotly_chart(fig2, use_container_width=True)


with tab3:
    st.subheader("Model Performance")
    st.markdown("""
| Model | R² Score |
|-------|----------|
| Linear Regression | ~0.65 |
| Decision Tree | ~0.70 |
| Random Forest | ~0.78 |
| Gradient Boosting | ~0.80 |
| **XGBoost** | **~0.82** |
    """)
    if hasattr(model, 'feature_importances_'):
        fi = pd.Series(
            model.feature_importances_[:10],
            index=feature_cols[:10]
        ).sort_values(ascending=True)
        fig_fi = px.bar(fi, orientation='h',
                        title="Top 10 Feature Importances",
                        color_discrete_sequence=['#45B7D1'])
        st.plotly_chart(fig_fi, use_container_width=True)


st.markdown(page_bg, unsafe_allow_html=True)
