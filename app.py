import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

model = joblib.load("best_model.pkl")
feature_cols = joblib.load("feature_cols.pkl")
df = pd.read_csv("house_price.csv.csv")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Poppins', sans-serif; }
[data-testid="stAppViewContainer"]{
    background-image:
    linear-gradient(rgba(255,255,255,0.28),rgba(255,255,255,0.28)),
    url("https://images.unsplash.com/photo-1600607687939-ce8a6c25118c?q=80&w=2070&auto=format&fit=crop");
    background-size: cover; background-position: center;
    background-repeat: no-repeat; background-attachment: fixed;
}
[data-testid="stHeader"]{ background: rgba(0,0,0,0); }
#MainMenu {visibility:hidden;} footer {visibility:hidden;}
.block-container{ padding-top: 1rem; padding-bottom: 2rem; }
.main-title{ font-size: 70px; font-weight: 800; color: #0f172a; margin-bottom: -15px; }
.sub-title{ font-size: 28px; color: #475569; margin-top: -10px; margin-bottom: 25px; }
.glass{ background: rgba(255,255,255,0.30); backdrop-filter: blur(16px);
    border-radius: 28px; padding: 30px; border: 1px solid rgba(255,255,255,0.4);
    box-shadow: 0 8px 32px rgba(0,0,0,0.12); }
.form-card{ background: rgba(255,255,255,0.82); padding: 18px; border-radius: 22px;
    margin-bottom: 18px; box-shadow: 0 5px 18px rgba(0,0,0,0.08); }
label{ font-size: 18px !important; font-weight: 600 !important; color: #111827 !important; }
.stSelectbox div[data-baseweb="select"], .stNumberInput input{
    border-radius: 14px !important; background: rgba(255,255,255,0.95) !important;
    border: 1px solid #e5e7eb !important; padding: 10px !important; }
.stTabs [data-baseweb="tab-list"]{ gap: 14px; }
.stTabs [data-baseweb="tab"]{ background: rgba(255,255,255,0.75); border-radius: 14px;
    padding: 14px 24px; font-size: 21px; font-weight: 700; color: #111827; }
.stTabs [aria-selected="true"]{ background: white !important; color: #5b5cff !important; }
.stButton>button{ width: 100%; height: 72px; border: none; border-radius: 18px;
    background: linear-gradient(90deg,#5b5cff,#ff4fa3); color: white; font-size: 28px;
    font-weight: 700; box-shadow: 0 10px 25px rgba(0,0,0,0.18); transition: 0.3s; }
.stButton>button:hover{ transform: scale(1.02); }
.prediction-box{ background: rgba(255,255,255,0.82); padding: 30px; border-radius: 24px;
    text-align: center; font-size: 40px; font-weight: 800; color: #111827; margin-top: 30px; }
</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1,8,2])
with col1: st.markdown("# 🏠")
with col2:
    st.markdown('<div class="main-title">House Price Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Smart Machine Learning Model</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div style="background:white;padding:12px 18px;border-radius:18px;text-align:center;font-weight:700;color:#ff4fa3;margin-top:15px;">Made with ❤️</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📈 Predict Price", "📊 Data Analysis", "ℹ️ Model Info"])

with tab1:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.markdown("<h1 style='color:#0f172a;font-size:38px;font-weight:800;'>🏘️ Property Details Bharo</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:22px;color:#475569;margin-top:-10px;'>Sahi jankari bharein, accurate price paayein ✨</p>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        posted_by = st.selectbox("👤 Dealer", ["Dealer","Owner","Builder"])
        bhk = st.slider("🛏️ BHK Number", 1,10,2)
        property_type = st.selectbox("🏡 Property Type", ["BHK","Villa","Apartment"])
    with c2:
        sqft = st.number_input("📐 Square Feet", value=1000)
        under = st.selectbox("🏗️ Under Construction?", ["No","Yes"])
        ready = st.selectbox("✅ Ready to Move?", ["Yes","No"])
    with c3:
        rera = st.selectbox("🛡️ RERA Approved?", ["Yes","No"])
        resale = st.selectbox("🔄 Resale Property?", ["No","Yes"])
        latitude = st.number_input("📍 Latitude", value=13.00)
    longitude = st.number_input("📌 Longitude", value=77.50)

    if st.button("📈 Price Prediction →"):
        try:
            input_data = pd.DataFrame([{
                "posted_by": posted_by,
                "under_construction": under,
                "rera": rera,
                "bhk_no": bhk,
                "square_ft": sqft,
                "ready_to_move": ready,
                "resale": resale,
                "longitude": longitude,
                "latitude": latitude
            }])
            input_data = pd.get_dummies(input_data)
            for col in feature_cols:
                if col not in input_data.columns:
                    input_data[col] = 0
            input_data = input_data[feature_cols]

            prediction = model.predict(input_data)[0]

            st.markdown(f"""
            <div class="prediction-box">
            🏷️ Estimated House Price <br><br>
            ₹ {prediction:,.2f} Lakhs
            </div>
            """, unsafe_allow_html=True)
        except Exception as e:
            st.error(e)
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="glass">', unsafe_allow_html=True)
    st.subheader("ℹ️ Model Information")
    st.write("""
    ✔️ Machine Learning Model  
    ✔️ Streamlit Dashboard  
    ✔️ Real Estate Prediction  
    ✔️ Interactive UI  
    ✔️ Live Deployment  
    """)
    st.markdown('</div>', unsafe_allow_html=True)
