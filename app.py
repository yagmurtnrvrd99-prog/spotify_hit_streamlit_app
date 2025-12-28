import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------
# Paths
# -----------------------------
ART_DIR = "artifacts"
MODEL_PATH = f"{ART_DIR}/final_model.joblib"
META_PATH  = f"{ART_DIR}/meta.json"

# -----------------------------
# Load artifacts
# -----------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta

model, meta = load_artifacts()

TH = float(meta.get("threshold", 0.67))
FEATURES = meta.get("feature_columns", [])

# Optional: genre frequency mapping saved in meta (if you add it later)
GENRE_FREQ_MAP = meta.get("genre_freq_map", {})

# -----------------------------
# UI Config + Style
# -----------------------------
st.set_page_config(page_title="Spotify Hit Predictor", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.2rem; padding-bottom: 2.5rem;}
      h1 {margin-bottom: 0.2rem;}
      h2 {margin-top: 1.2rem;}
      .stButton>button {border-radius: 10px; padding: 0.55rem 1.1rem;}
      .stExpander {border-radius: 14px;}
      [data-baseweb="slider"] {padding-top: 0.35rem;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Spotify Hit Predictor")
st.caption("Enter track & artist information and choose a genre.")

# -----------------------------
# Reset button
# -----------------------------
if st.button("Reset"):
    for k in list(st.session_state.keys()):
        if not k.startswith("_"):
            del st.session_state[k]
    st.rerun()

# -----------------------------
# Genre grouping (super_map style)
# -----------------------------
super_map = {
    # Acoustic / Folk / Country
    "acoustic": "Acoustic/Folk/Country", "folk": "Acoustic/Folk/Country",
    "country": "Acoustic/Folk/Country", "bluegrass": "Acoustic/Folk/Country",
    "singer-songwriter": "Acoustic/Folk/Country", "songwriter": "Acoustic/Folk/Country",

    # Pop
    "pop": "Pop", "indie-pop": "Pop", "synth-pop": "Pop", "k-pop": "Pop",
    "j-pop": "Pop", "mandopop": "Pop", "cantopop": "Pop", "pop-film": "Pop",
    "british": "Pop",

    # Hip-Hop / R&B
    "hip-hop": "Hip-Hop/R&B", "rap": "Hip-Hop/R&B", "r-n-b": "Hip-Hop/R&B",
    "soul": "Hip-Hop/R&B", "funk": "Hip-Hop/R&B",

    # Electronic / Dance
    "edm": "Electronic/Dance", "electronic": "Electronic/Dance", "electro": "Electronic/Dance",
    "house": "Electronic/Dance", "deep-house": "Electronic/Dance", "techno": "Electronic/Dance",
    "detroit-techno": "Electronic/Dance", "chicago-house": "Electronic/Dance",
    "drum-and-bass": "Electronic/Dance", "dubstep": "Electronic/Dance",
    "dance": "Electronic/Dance", "club": "Electronic/Dance", "disco": "Electronic/Dance",

    # Rock / Metal
    "rock": "Rock/Metal", "alt-rock": "Rock/Metal", "alternative": "Rock/Metal",
    "punk": "Rock/Metal", "punk-rock": "Rock/Metal", "hard-rock": "Rock/Metal",
    "metal": "Rock/Metal", "black-metal": "Rock/Metal", "death-metal": "Rock/Metal",
    "metalcore": "Rock/Metal", "grunge": "Rock/Metal", "industrial": "Rock/Metal",
    "rock-n-roll": "Rock/Metal", "rockabilly": "Rock/Metal", "hardcore": "Rock/Metal",
    "psych-rock": "Rock/Metal", "emo": "Rock/Metal", "garage": "Rock/Metal",

    # Classical / Jazz
    "classical": "Classical/Jazz", "piano": "Classical/Jazz", "jazz": "Classical/Jazz",
    "ambient": "Classical/Jazz", "new-age": "Classical/Jazz",

    # Latin / Reggae
    "latin": "Latin/Reggae", "latino": "Latin/Reggae", "reggaeton": "Latin/Reggae",
    "reggae": "Latin/Reggae", "dancehall": "Latin/Reggae", "brazil": "Latin/Reggae",

    # Other
    "anime": "Other", "disney": "Other", "children": "Other", "comedy": "Other",
}

subgenres = sorted(list(super_map.keys()))
supergenres = sorted(list(set(super_map.values())))

st.header("Genre Selection")

col_g1, col_g2 = st.columns(2)
with col_g1:
    default_super = "Acoustic/Folk/Country" if "Acoustic/Folk/Country" in supergenres else supergenres[0]
    chosen_super = st.selectbox("Genre", supergenres, index=supergenres.index(default_super))

with col_g2:
    sub_list = [g for g in subgenres if super_map.get(g) == chosen_super]
    chosen_sub = st.selectbox("Sub-genre", sub_list, index=0 if len(sub_list) else None)

track_genre_freq = float(GENRE_FREQ_MAP.get(chosen_sub, 0.05))

# -----------------------------
# Basic Features
# -----------------------------
st.header("Basic Features")

c1, c2 = st.columns(2)

with c1:
    duration_sec = st.slider("duration", min_value=30, max_value=900, value=180)
    st.caption(f"Selected: {duration_sec//60}:{duration_sec%60:02d}")

    # UPDATED: followers slider up to 150,000K
    artist_followers_k = st.slider(
        "artist_followers (K)",
        min_value=0,
        max_value=150_000,
        value=100,
        step=100
    )
    st.caption(f"{artist_followers_k:,}K = {artist_followers_k*1000:,} followers")

    danceability = st.slider("danceability", 0.0, 1.0, 0.50)
    energy = st.slider("energy", 0.0, 1.0, 0.50)
    loudness = st.slider("loudness", -60.0, 0.0, -8.0)

with c2:
    tempo = st.slider("tempo", 40.0, 220.0, 120.0)
    artist_popularity = st.slider("artist_popularity", 0, 100, 50)
    valence = st.slider("valence", 0.0, 1.0, 0.50)
    release_year = st.slider("release_year", 1950, 2025, 2020)

# -----------------------------
# Advanced (optional)
# -----------------------------
with st.expander("Advanced (optional)", expanded=False):
    use_exact_duration = st.checkbox("Enter exact duration (seconds)", value=False)
    use_exact_followers = st.checkbox("Enter exact followers", value=False)

    if use_exact_duration:
        duration_sec_exact = st.number_input(
            "Exact duration (seconds)",
            min_value=1,
            max_value=36000,
            value=int(duration_sec),
            step=1
        )
        duration_sec = int(duration_sec_exact)

    if use_exact_followers:
        followers_exact = st.number_input(
            "Exact followers",
            min_value=0,
            max_value=2_000_000_000,
            value=int(artist_followers_k * 1000),
            step=1000
        )
        artist_followers_k = int(followers_exact // 1000)

    speechiness = st.slider("speechiness", 0.0, 1.0, 0.05)
    acousticness = st.slider("acousticness", 0.0, 1.0, 0.20)
    instrumentalness = st.slider("instrumentalness", 0.0, 1.0, 0.00)
    liveness = st.slider("liveness", 0.0, 1.0, 0.15)

# If expander not opened, keep defaults to avoid NameError
if "speechiness" not in locals():
    speechiness, acousticness, instrumentalness, liveness = 0.05, 0.20, 0.00, 0.15

# -----------------------------
# Build model row (MODEL expects duration_ms, artist_followers, ...)
# -----------------------------
row = {
    "duration_ms": float(duration_sec) * 1000.0,
    "danceability": float(danceability),
    "energy": float(energy),
    "loudness": float(loudness),
    "speechiness": float(speechiness),
    "acousticness": float(acousticness),
    "instrumentalness": float(instrumentalness),
    "liveness": float(liveness),
    "valence": float(valence),
    "tempo": float(tempo),
    "release_year": float(release_year),
    "artist_followers": float(artist_followers_k) * 1000.0,
    "artist_popularity": float(artist_popularity),
    "track_genre_freq": float(track_genre_freq),
}

X = pd.DataFrame([row])

# Ensure all expected columns exist + correct order
for c in FEATURES:
    if c not in X.columns:
        X[c] = 0.0

X = X[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)

# -----------------------------
# Predict
# -----------------------------
st.divider()

if st.button("Predict"):
    if hasattr(model, "predict_proba"):
        score = float(model.predict_proba(X)[:, 1][0])
    else:
        score = float(model.decision_function(X)[0])

    pred = int(score >= TH)

    # UPDATED: no (score | th) text
    if pred == 1:
        st.success("This song would be a HIT!")
    else:
        st.warning("This song would NOT be a hit.")

    # Optional "non-hit chance" label (uses hidden score but doesn't display it)
    if pred == 0:
        if score >= TH * 0.9:
            st.info("Non-hit chance: **Low**")
        elif score >= TH * 0.75:
            st.info("Non-hit chance: **Medium**")
        else:
            st.info("Non-hit chance: **High**")
