import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.graph_objects as go
import sys
sys.path.append(".")
from preprocessing import preprocess_flux

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.keras")

@st.cache_data
def load_data():
    return pd.read_csv("demo_stars.csv")

model   = load_model()
demo_df = load_data()
flux_cols = [c for c in demo_df.columns if c != "LABEL"]

st.title("Planet Hunter")
st.caption("Powered by a 1D CNN trained on NASA Kepler telescope data.")

star_index = st.slider("A slider to browse 200 real stars from the Kepler dataset", 0, len(demo_df)-1, 0)
row   = demo_df.iloc[star_index]
flux  = row[flux_cols].values.astype(float)
label = int(row["LABEL"])

fig = go.Figure(go.Scatter(y=flux, mode="lines", line=dict(color="cyan", width=1)))
fig.update_layout(title="Light Curve", xaxis_title="Time", yaxis_title="Flux",
                  template="plotly_dark", height=300)
st.plotly_chart(fig, use_container_width=True)

if st.button("Analyze", type="primary"):
    prob    = float(model.predict(preprocess_flux(flux), verbose=0)[0][0])
    planet  = prob > 0.5
    verdict = "🌍 EXOPLANET DETECTED" if planet else "⭐ Not an exoplanet"
    correct = planet == (label == 1)
    st.metric("Result", verdict, f"{prob:.1%} confidence")
    if correct:
        st.success("✅ Model was correct")
    else:
        st.error("❌ Model was wrong")