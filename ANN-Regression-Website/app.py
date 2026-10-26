import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="PowerPulse | ANN Predictor", page_icon="⚡", layout="wide")

st.markdown("""
<style>
.stApp {background: linear-gradient(135deg,#081225 0%,#101b35 55%,#0b1220 100%); color:#edf4ff;}
[data-testid="stHeader"] {background:rgba(0,0,0,0);}
.block-container {padding-top:2rem; max-width:1400px;}
h1,h2,h3,p,label,span {color:#edf4ff;}
/* High-contrast sidebar navigation for Streamlit's radio-based menu */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1930 0%, #101f38 100%);
    border-right: 1px solid #263c61;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: #142641;
    border: 1px solid #294568;
    border-radius: 10px;
    padding: 0.55rem 0.7rem;
    margin: 0.2rem 0;
    color: #f1f5ff !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label p,
section[data-testid="stSidebar"] [data-testid="stRadio"] label span,
section[data-testid="stSidebar"] [data-testid="stRadio"] label div {
    color: #f1f5ff !important;
    font-weight: 600;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: linear-gradient(90deg, #155e75, #164e63);
    border-color: #22d3ee;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] input {
    accent-color: #22d3ee;
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
section[data-testid="stSidebar"] label {
    color: #dbeafe !important;
}
.hero {padding:1.5rem 1.8rem;border:1px solid #263c61;border-radius:22px;background:linear-gradient(120deg,#132746,#10182b);margin-bottom:1.2rem;}
.hero small {color:#8ebcff;letter-spacing:2px;font-weight:700;}
.hero h1 {font-size:2.5rem;margin:.35rem 0;}
.hero p {color:#b9c9e5;font-size:1.05rem;}
.metric-card {padding:1rem;border-radius:16px;background:#111f36;border:1px solid #253c5d;}
div[data-testid="stMetric"] {background:#111f36;border:1px solid #253c5d;padding:15px;border-radius:14px;}
.stButton button {border-radius:12px;font-weight:700;min-height:3rem;}
footer {visibility:hidden;}
</style>
<div class="hero">
<small>POWERPULSE • MACHINE LEARNING</small>
<h1>Power Plant Energy Predictor</h1>
<p>Estimate electrical energy output using a trained Artificial Neural Network (PyTorch).</p>
</div>
""", unsafe_allow_html=True)

DATA_FILE = "powerplant_data.csv"
MODEL_FILE = "best_model.pt"
FEATURES = ["AT", "V", "AP", "RH"]
LABELS = {
    "AT": "Ambient Temperature (°C)",
    "V": "Exhaust Vacuum (cm Hg)",
    "AP": "Ambient Pressure (mbar)",
    "RH": "Relative Humidity (%)"
}

class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(4, 6), nn.ReLU(),
            nn.Linear(6, 6), nn.ReLU(),
            nn.Linear(6, 1)
        )
    def forward(self, x):
        return self.model(x)

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_FILE)
    required = FEATURES + ["PE"]
    if not set(required).issubset(data.columns):
        raise ValueError("Dataset is missing required columns: " + ", ".join(required))
    return data

@st.cache_resource
def load_model_and_metrics():
    data = load_data()
    X = data[FEATURES]
    y = data["PE"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # Re-create the exact training split/scaler used in the supplied notebook.
    scaler = StandardScaler()
    scaler.fit(X_train)
    model = ANN()
    try:
        weights = torch.load(MODEL_FILE, map_location="cpu", weights_only=True)
    except TypeError:
        weights = torch.load(MODEL_FILE, map_location="cpu")
    model.load_state_dict(weights)
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(scaler.transform(X_test), dtype=torch.float32)).numpy().ravel()
    metrics = {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "r2": float(r2_score(y_test, preds)),
        "actual": y_test.reset_index(drop=True),
        "predicted": pd.Series(preds)
    }
    return model, scaler, metrics, data

try:
    model, scaler, metrics, data = load_model_and_metrics()
except Exception as exc:
    st.error("Could not load the model or dataset. Keep app.py, best_model.pt and powerplant_data.csv in the same folder.")
    st.exception(exc)
    st.stop()

with st.sidebar:
    st.markdown("## ⚙️ Navigation")
    page = st.radio("Choose a section", ["Predict Energy", "Model Insights", "About Project"])
    st.divider()
    st.caption("Model: ANN • Framework: PyTorch")
    st.caption(f"Dataset: {len(data):,} records")

if page == "Predict Energy":
    st.subheader("Enter operating conditions")
    st.write("Adjust the four inputs below, then select **Predict energy output**.")
    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            at = st.slider("Ambient Temperature (AT, °C)", float(data.AT.min()), float(data.AT.max()), float(data.AT.median()), 0.1, help="Ambient air temperature.")
            v = st.slider("Exhaust Vacuum (V, cm Hg)", float(data.V.min()), float(data.V.max()), float(data.V.median()), 0.1, help="Steam turbine exhaust vacuum.")
        with c2:
            ap = st.slider("Ambient Pressure (AP, mbar)", float(data.AP.min()), float(data.AP.max()), float(data.AP.median()), 0.1, help="Ambient atmospheric pressure.")
            rh = st.slider("Relative Humidity (RH, %)", float(data.RH.min()), float(data.RH.max()), float(data.RH.median()), 0.1, help="Relative humidity.")
        submitted = st.form_submit_button("⚡ Predict energy output", use_container_width=True, type="primary")
    if submitted:
        row = pd.DataFrame([[at, v, ap, rh]], columns=FEATURES)
        scaled = scaler.transform(row)
        with torch.no_grad():
            prediction = float(model(torch.tensor(scaled, dtype=torch.float32)).item())
        st.markdown("### Prediction result")
        left, right = st.columns([1, 2])
        with left:
            st.metric("Estimated net energy output (PE)", f"{prediction:.2f} MW")
        with right:
            st.info("This is a model estimate based on the supplied dataset. It is not a guarantee of real plant output; use within the training data's ranges.")
        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.insert(0, {"AT": at, "V": v, "AP": ap, "RH": rh, "Predicted PE": round(prediction, 2)})
        st.session_state.history = st.session_state.history[:20]
    if st.session_state.get("history"):
        st.markdown("### Recent predictions")
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)
        if st.button("Clear prediction history"):
            st.session_state.history = []
            st.rerun()

elif page == "Model Insights":
    st.subheader("Model performance")
    st.caption("Evaluation uses a held-out 20% test split with random_state=42, matching the notebook's split.")
    a,b,c,d = st.columns(4)
    a.metric("R² score", f"{metrics['r2']:.4f}")
    b.metric("Mean absolute error", f"{metrics['mae']:.2f} MW")
    c.metric("Root mean squared error", f"{metrics['rmse']:.2f} MW")
    d.metric("Test samples", f"{len(metrics['actual']):,}")
    comp = pd.DataFrame({"Actual PE": metrics["actual"], "Predicted PE": metrics["predicted"]})
    fig = px.scatter(comp, x="Actual PE", y="Predicted PE", opacity=.65,
                     title="Actual vs Predicted energy output",
                     labels={"Actual PE":"Actual output (MW)", "Predicted PE":"Predicted output (MW)"})
    lo = float(min(comp.min()))
    hi = float(max(comp.max()))
    fig.add_trace(go.Scatter(x=[lo,hi], y=[lo,hi], mode="lines", name="Ideal prediction", line=dict(dash="dash")))
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(12,25,44,0)")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### Dataset overview")
    st.dataframe(data.describe().T.round(2), use_container_width=True)
    st.markdown("### Feature relationships")
    corr = data.corr(numeric_only=True)
    fig2 = px.imshow(corr, text_auto=".2f", aspect="auto", title="Correlation matrix", color_continuous_scale="RdBu_r")
    fig2.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig2, use_container_width=True)

else:
    st.subheader("About this project")
    st.write("This application uses a feed-forward Artificial Neural Network trained with PyTorch to estimate power plant electrical energy output.")
    st.markdown("**Inputs**")
    for f in FEATURES:
        st.write(f"• **{f}** — {LABELS[f]}")
    st.markdown("**Prediction target**")
    st.write("• **PE** — Net hourly electrical energy output (MW), as represented in the dataset.")
    st.markdown("**Model architecture**")
    st.code("4 input features → Dense(6) + ReLU → Dense(6) + ReLU → Dense(1)", language="text")
    st.markdown("**Important notes**")
    st.write("- Input sliders are limited to the observed dataset ranges to reduce out-of-range predictions.")
    st.write("- The scaler is reconstructed from the same 80/20 split and random seed as the notebook. Keep the dataset unchanged for consistent results.")
    st.write("- Metrics are estimates on a held-out test set, not proof of real-world performance.")
    st.write("- This demo is educational and should not be used as the sole basis for real plant operating decisions.")

st.divider()
st.caption("Built with Streamlit + PyTorch • ANN Regression Mini Project")
