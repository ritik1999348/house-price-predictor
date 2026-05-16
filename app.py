import streamlit as st
import pandas as pd
import numpy as np
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

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

/* BACKGROUND IMAGE */
[data-testid="stAppViewContainer"]{
    background-image:
    linear-gradient(rgba(255,255,255,0.45),
    rgba(255,255,255,0.45)),
    url("https://images.unsplash.com/photo-1600585154526-990dced4db0d?q=80&w=1920&auto=format&fit=crop");

    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

/* HEADER */
[data-testid="stHeader"]{
    background: rgba(0,0,0,0);
}

/* MAIN CONTAINER */
.block-container{
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* REMOVE STREAMLIT MENU */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* MAIN TITLE */
.main-title{
    font-size: 64px;
    font-weight: 800;
    color: #111827;
    margin-bottom: -10px;
}

/* SUBTITLE */
.sub-title{
    font-size: 26px;
    color: #4b5563;
    margin-bottom: 20px;
}

/* GLASS CARD */
.glass{
    background: rgba(255,255,255,0.30);
    border-radius: 25px;
    padding: 30px;
    backdrop-filter: blur(14px);
    border: 1px solid rgba(255,255,255,0.4);
    box-shadow: 0 8px 32px rgba(0,0,0,0.15);
}

/* INPUT BOX */
.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"]{
    border-radius: 14px !important;
    border: 1px solid #d1d5db !important;
    padding: 12px !important;
    background: rgba(255,255,255,0.9) !important;
}

/* LABELS */
label{
    font-size: 18px !important;
    font-weight: 600 !important;
    color: #111827 !important;
}

/* BUTTON */
.stButton>button{
    width: 100%;
    background: linear-gradient(90deg,#5B5CFF,#FF4FA3);
    color: white;
    border: none;
    border-radius: 18px;
    height: 65px;
    font-size: 26px;
    font-weight: 700;
    box-shadow: 0 8px 20px rgba(0,0,0,0.20);
    transition: 0.3s;
}

.stButton>button:hover{
    transform: scale(1.02);
}

/* TABS */
.stTabs [data-baseweb="tab-list"]{
    gap: 15px;
}

.stTabs [data-baseweb="tab"]{
    background: rgba(255,255,255,0.6);
    border-radius: 14px;
    padding: 12px 20px;
    font-size: 20px;
    font-weight: 700;
}

/* PREDICTION TEXT */
.prediction{
    background: rgba(255,255,255,0.75);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    font-size: 36px;
    font-weight: 800;
    color: #111827;
    margin-top: 25px;
}

</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1,10])

with col1:
    st.markdown("# 🏠")

with col2:
    st.markdown('<div class="main-title">House Price Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Smart Machine Learning Model</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "📈 Predict Price",
    "📊 Data Analysis",
    "ℹ️ Model Info"
])

with tab1:

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.markdown("## 🏘️ Property Details Bharo")
    st.markdown("### Sahi jankari bharein, accurate price paayein ✨")

    col1, col2, col3 = st.columns(3)

    with col1:
        posted_by = st.selectbox(
            "👤 Dealer",
            ["Dealer", "Owner", "Builder"]
        )

        bhk = st.slider(
            "🛏️ BHK Number",
            1, 10, 2
        )

        property_type = st.selectbox(
            "🏡 Property Type",
            ["BHK", "Villa", "Apartment"]
        )

    with col2:
        sqft = st.number_input(
            "📐 Square Feet",
            value=1000
        )

        under_const = st.selectbox(
            "🏗️ Under Construction?",
            ["No", "Yes"]
        )

        ready_move = st.selectbox(
            "✅ Ready to Move?",
            ["Yes", "No"]
        )

    with col3:
        rera = st.selectbox(
            "🛡️ RERA Approved?",
            ["Yes", "No"]
        )

        resale = st.selectbox(
            "🔄 Resale Property?",
            ["No", "Yes"]
        )

        latitude = st.number_input(
            "📍 Latitude",
            value=13.00
        )

    longitude = st.number_input(
        "📌 Longitude",
        value=77.50
    )

    # BUTTON
    if st.button("📈 Price Predict Karo →"):

        try:
            input_data = pd.DataFrame([{
                "posted_by": posted_by,
                "under_construction": under_const,
                "rera": rera,
                "bhk_no": bhk,
                "square_ft": sqft,
                "ready_to_move": ready_move,
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

            st.markdown(
                f'''
                <div class="prediction">
                🏷️ Estimated House Price <br><br>
                ₹ {prediction:,.2f} Lakhs
                </div>
                ''',
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

with tab2:

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.subheader("📊 Dataset Preview")

    st.dataframe(df.head())

    st.markdown("</div>", unsafe_allow_html=True)

with tab3:

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.subheader("ℹ️ Model Information")

    st.write("""
    ✔️ Machine Learning Model Used  
    ✔️ Real Estate Dataset  
    ✔️ Price Prediction System  
    ✔️ Streamlit Deployment  
    ✔️ Interactive Dashboard  
    """)

    st.markdown("</div>", unsafe_allow_html=True)
