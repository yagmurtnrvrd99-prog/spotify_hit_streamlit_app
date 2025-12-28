import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------
# Load artifacts
# --------------------
ART_DIR = "artifacts"
MODEL_PATH = f"{ART_DIR}/final_model.joblib"
META_PATH  = f"{ART_DIR}/meta.json"

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta

model, meta = load_artifacts()
TH = float(meta["threshold"])
FEATURES = meta["feature_columns"]

st.set_page_config(page_title="Spotify Hit Predictor", layout="wide")
st.title("🎵 Spotify Hit Predictor")

col1, col2 = st.columns(2)

with col1:
    duration_sec = st.slider("Duration (seconds)", 30, 900, 180)
    artist_followers_k = st.slider("Artist followers (K)", 0, 5000, 100, step=10)
    artist_popularity = st.slider("Artist popularity", 0, 100, 50)
    release_year = st.slider("Release year", 1950, 2025, 2020)
    track_genre_freq = st.slider("Track genre frequency", 0.0, 1.0, 0.05)

with col2:
    tempo = st.slider("Tempo", 40.0, 220.0, 120.0)
    loudness = st.slider("Loudness (dB)", -60.0, 0.0, -8.0)
    danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
    energy = st.slider("Energy", 0.0, 1.0, 0.5)
    valence = st.slider("Valence", 0.0, 1.0, 0.5)
    speechiness = st.slider("Speechiness", 0.0, 1.0, 0.05)
    acousticness = st.slider("Acousticness", 0.0, 1.0, 0.20)
    instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.00)
    liveness = st.slider("Liveness", 0.0, 1.0, 0.15)

row = {
    "duration_ms": duration_sec * 1000,
    "danceability": danceability,
    "energy": energy,
    "loudness": loudness,
    "speechiness": speechiness,
    "acousticness": acousticness,
    "instrumentalness": instrumentalness,
    "liveness": liveness,
    "valence": valence,
    "tempo": tempo,
    "release_year": float(release_year),
    "artist_followers": artist_followers_k * 1000,
    "artist_popularity": float(artist_popularity),
    "track_genre_freq": track_genre_freq,
}

X = pd.DataFrame([row])

for c in FEATURES:
    if c not in X.columns:
        X[c] = 0
X = X[FEATURES].fillna(0)

if st.button("Predict"):
    if hasattr(model, "predict_proba"):
        score = float(model.predict_proba(X)[:, 1][0])
    else:
        score = float(model.decision_function(X)[0])

    st.subheader("Result")
    st.write(f"**Hit probability:** {score:.4f}")
    st.write(f"**Threshold:** {TH:.2f}")

    if score >= TH:
        st.success("🎉 This song would be a HIT!")
    else:
        st.warning("❌ This song would NOT be a hit.")
