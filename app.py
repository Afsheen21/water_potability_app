import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Water Potability Predictor", page_icon="💧", layout="wide")

# ---------- Styling (light pastel theme) ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f0fbff 0%, #f6f3ff 50%, #fff5f8 100%);
}
.hero {
    background: linear-gradient(120deg, #cdeffd, #dbe7ff, #ecdcff);
    padding: 30px 34px;
    border-radius: 20px;
    color: #1d3557;
    box-shadow: 0 4px 16px rgba(120, 160, 200, 0.25);
    margin-bottom: 24px;
}
.hero h1 { margin: 0; font-size: 2.3rem; color: #1d3557; }
.hero p  { margin: 8px 0 0 0; font-size: 1.05rem; color: #3d5a80; }

.card-title-chem, .card-title-phys {
    padding: 10px 18px; border-radius: 12px;
    font-weight: 700; font-size: 1.1rem; margin-bottom: 8px;
}
.card-title-chem { background: #ffe0ec; color: #8a2b50; }
.card-title-phys { background: #d8f0ff; color: #1f5f8b; }

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(255, 255, 255, 0.85);
    border-radius: 18px;
    border: 1px solid #e6edf5;
    box-shadow: 0 3px 12px rgba(150, 170, 200, 0.18);
}

div.stButton > button {
    background: linear-gradient(90deg, #ffd9a8, #ffc2d1);
    color: #5a3a2e; font-size: 1.15rem; font-weight: 700;
    padding: 0.7rem 2rem; border-radius: 14px; border: none; width: 100%;
    box-shadow: 0 3px 10px rgba(255, 180, 160, 0.4);
    transition: transform 0.15s;
}
div.stButton > button:hover { transform: scale(1.02); color: #5a3a2e; }

.result-good {
    background: #d9f7e3; border: 2px solid #9ce0b4;
    padding: 24px; border-radius: 18px; color: #1e6b3a;
    font-size: 1.5rem; text-align: center;
}
.result-bad {
    background: #ffe3e3; border: 2px solid #f5b5b5;
    padding: 24px; border-radius: 18px; color: #a12b2b;
    font-size: 1.5rem; text-align: center;
}
.bar-wrap {
    background: #ffd6d6; border-radius: 12px; height: 28px;
    overflow: hidden; margin-top: 8px;
}
.bar-fill {
    background: #a8e6bf; height: 100%; color: #1e6b3a;
    font-weight: 700; text-align: center; line-height: 28px;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #eaf8fc, #f3ecfb);
}
</style>
""", unsafe_allow_html=True)

# ---------- Model ----------
@st.cache_resource
def load_model():
    return joblib.load("Classification.pkl")

model = load_model()

# ---------- Header ----------
st.markdown("""
<div class="hero">
    <h1>💧 Water Potability Predictor</h1>
    <p>Adjust the sliders with your water measurements, then press Predict to see if the water is safe to drink.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("ℹ️ About")
    st.write("A Random Forest model trained on the Water Potability dataset.")
    st.write("**Inputs:** 9 chemical and physical water properties.")
    st.write("**Output:** potable (safe) or not potable, with a confidence score.")
    st.warning("Model accuracy is about 66%. This is a demo, not for real safety decisions.")

# ---------- Inputs ----------
left, right = st.columns(2)

with left:
    with st.container(border=True):
        st.markdown('<div class="card-title-chem">🧪 Chemical properties</div>', unsafe_allow_html=True)
        ph = st.slider("pH", 0.0, 14.0, 7.0, 0.1)
        chloramines = st.slider("Chloramines", 0.0, 15.0, 7.0, 0.1)
        sulfate = st.slider("Sulfate", 0.0, 500.0, 330.0, 1.0)
        organic_carbon = st.slider("Organic carbon", 0.0, 40.0, 14.0, 0.1)
        trihalomethanes = st.slider("Trihalomethanes", 0.0, 150.0, 66.0, 0.5)

with right:
    with st.container(border=True):
        st.markdown('<div class="card-title-phys">🌊 Physical properties</div>', unsafe_allow_html=True)
        hardness = st.slider("Hardness", 0.0, 500.0, 200.0, 1.0)
        solids = st.slider("Solids", 0.0, 70000.0, 20000.0, 100.0)
        conductivity = st.slider("Conductivity", 0.0, 1000.0, 420.0, 1.0)
        turbidity = st.slider("Turbidity", 0.0, 10.0, 4.0, 0.1)

st.write("")
predict = st.button("🔍 Predict Potability")

# ---------- Prediction ----------
if predict:
    row = pd.DataFrame([{
        "ph": ph,
        "Hardness": hardness,
        "Solids": solids,
        "Chloramines": chloramines,
        "Sulfate": sulfate,
        "Conductivity": conductivity,
        "Organic_carbon": organic_carbon,
        "Trihalomethanes": trihalomethanes,
        "Turbidity": turbidity,
    }])

    result = model.predict(row)[0]
    prob = float(model.predict_proba(row)[0][1])  # probability of "potable"

    st.write("")
    if result == 1:
        st.balloons()
        st.markdown('<div class="result-good">✅ <b>Water is potable</b><br>Safe to drink</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-bad">❌ <b>Water is NOT potable</b><br>Not safe to drink</div>',
                    unsafe_allow_html=True)

    st.write("")
    m1, m2 = st.columns(2)
    m1.metric("✅ Probability potable", f"{prob:.0%}")
    m2.metric("❌ Probability not potable", f"{1 - prob:.0%}")

    st.markdown(
        f'<div class="bar-wrap"><div class="bar-fill" style="width:{prob*100:.0f}%">{prob:.0%} potable</div></div>',
        unsafe_allow_html=True
    )