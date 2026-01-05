import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import os

ART_DIR = "artifacts"
MODEL_PATH = os.path.join(ART_DIR, "final_model.joblib")
META_PATH = os.path.join(ART_DIR, "meta.json")
GENRE_MAP_PATH = os.path.join(ART_DIR, "genre_freq_map.json")

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(GENRE_MAP_PATH, "r", encoding="utf-8") as f:
        genre_freq_map = json.load(f)
    return model, meta, genre_freq_map

model, meta, GENRE_FREQ_MAP = load_artifacts()

TH = float(meta.get("threshold", 0.67))
FEATURES = meta.get("feature_columns", [])
if not FEATURES:
    st.error("feature_columns not found in artifacts/meta.json")
    st.stop()
if "track_genre_freq" not in FEATURES:
    st.error("Model does not include track_genre_freq.")
    st.stop()

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

if "reset_nonce" not in st.session_state:
    st.session_state["reset_nonce"] = 0

def do_reset():
    keep = {"reset_nonce"}
    for k in list(st.session_state.keys()):
        if k not in keep:
            del st.session_state[k]
    st.session_state["reset_nonce"] += 1

st.title("Spotify Hit Predictor")
st.caption("Adjust the inputs and press Predict. Use the ? icons next to each variable for a short explanation.")

if st.button("Reset", on_click=do_reset):
    st.rerun()

super_map = {
    "rock": "Rock/Metal", "alt-rock": "Rock/Metal", "alternative": "Rock/Metal",
    "hard-rock": "Rock/Metal", "punk": "Rock/Metal", "punk-rock": "Rock/Metal",
    "metal": "Rock/Metal", "black-metal": "Rock/Metal", "death-metal": "Rock/Metal",
    "metalcore": "Rock/Metal", "grunge": "Rock/Metal", "industrial": "Rock/Metal",
    "rock-n-roll": "Rock/Metal", "rockabilly": "Rock/Metal", "hardcore": "Rock/Metal",
    "psych-rock": "Rock/Metal", "emo": "Rock/Metal", "garage": "Rock/Metal",
    "pop": "Pop", "indie-pop": "Pop", "synth-pop": "Pop", "pop-film": "Pop",
    "k-pop": "Pop", "j-pop": "Pop", "mandopop": "Pop", "cantopop": "Pop",
    "british": "Pop",
    "hip-hop": "Hip-Hop/R&B", "rap": "Hip-Hop/R&B", "r-n-b": "Hip-Hop/R&B",
    "soul": "Hip-Hop/R&B", "funk": "Hip-Hop/R&B",
    "edm": "Electronic/Dance", "electronic": "Electronic/Dance", "electro": "Electronic/Dance",
    "house": "Electronic/Dance", "deep-house": "Electronic/Dance",
    "techno": "Electronic/Dance", "detroit-techno": "Electronic/Dance", "chicago-house": "Electronic/Dance",
    "drum-and-bass": "Electronic/Dance", "dubstep": "Electronic/Dance",
    "dance": "Electronic/Dance", "club": "Electronic/Dance", "disco": "Electronic/Dance",
    "acoustic": "Acoustic/Folk/Country", "folk": "Acoustic/Folk/Country",
    "country": "Acoustic/Folk/Country", "bluegrass": "Acoustic/Folk/Country",
    "singer-songwriter": "Acoustic/Folk/Country", "songwriter": "Acoustic/Folk/Country",
    "latin": "Latin/Reggae", "latino": "Latin/Reggae", "reggaeton": "Latin/Reggae",
    "reggae": "Latin/Reggae", "dancehall": "Latin/Reggae", "brazil": "Latin/Reggae",
    "classical": "Classical/Jazz", "piano": "Classical/Jazz", "jazz": "Classical/Jazz",
    "ambient": "Classical/Jazz", "new-age": "Classical/Jazz",
    "anime": "Other", "disney": "Other", "children": "Other", "comedy": "Other",
}

genres = sorted(GENRE_FREQ_MAP.keys())
supergenres = sorted(set(super_map.get(g, "Other") for g in genres))

st.header("Genre Selection")
g1, g2 = st.columns(2)

with g1:
    default_super = "Pop" if "Pop" in supergenres else supergenres[0]
    chosen_super = st.selectbox(
        "Super genre",
        supergenres,
        index=supergenres.index(default_super),
        key=f"super_{st.session_state['reset_nonce']}",
        help="A broad category (e.g., Pop, Rock/Metal). This only filters the list of track genres below."
    )

with g2:
    sub_list = [g for g in genres if super_map.get(g, "Other") == chosen_super]
    sub_list = sub_list if sub_list else genres
    chosen_genre = st.selectbox(
        "track_genre",
        sub_list,
        index=0,
        key=f"genre_{st.session_state['reset_nonce']}",
        help="The track’s genre label used in the dataset. This affects track_genre_freq (how common the genre is in the dataset)."
    )

track_genre_freq = float(GENRE_FREQ_MAP.get(chosen_genre, 0.0))

st.header("Basic Features")
c1, c2 = st.columns(2)

with c1:
    duration_sec = st.slider(
        "duration (sec)",
        30,
        900,
        180,
        key=f"duration_{st.session_state['reset_nonce']}",
        help="Track length in seconds. Longer or shorter tracks can behave differently across genres, but the effect is usually moderate."
    )
    st.caption(f"Selected: {duration_sec//60}:{duration_sec%60:02d}")

    artist_followers_k = st.slider(
        "artist_followers (K)",
        0,
        150_000,
        100,
        step=100,
        key=f"followers_{st.session_state['reset_nonce']}",
        help="Approx. artist followers on Spotify in thousands (K). Higher values usually mean a larger audience and can increase hit probability."
    )
    st.caption(f"{artist_followers_k:,}K = {artist_followers_k*1000:,} followers")

    danceability = st.slider(
        "danceability",
        0.0,
        1.0,
        0.50,
        key=f"dance_{st.session_state['reset_nonce']}",
        help="How suitable the track is for dancing (0–1). Higher generally means more rhythmic and dance-friendly."
    )

    energy = st.slider(
        "energy",
        0.0,
        1.0,
        0.50,
        key=f"energy_{st.session_state['reset_nonce']}",
        help="Perceived intensity and activity (0–1). Higher often means louder, faster, and more intense."
    )

    loudness = st.slider(
        "loudness (dB)",
        -30.0,
        0.0,
        -8.0,
        key=f"loud_{st.session_state['reset_nonce']}",
        help="Average loudness in dB (Spotify uses a negative dBFS-like scale). Closer to 0 means louder/stronger master. Typical modern music is around -12 to -5."
    )

with c2:
    tempo = st.slider(
        "tempo (BPM)",
        40.0,
        220.0,
        120.0,
        key=f"tempo_{st.session_state['reset_nonce']}",
        help="Estimated tempo in beats per minute (BPM). Higher means faster."
    )

    artist_popularity = st.slider(
        "artist_popularity",
        0,
        100,
        50,
        key=f"apop_{st.session_state['reset_nonce']}",
        help="Spotify popularity score for the artist (0–100). Higher values usually indicate more current visibility and may strongly increase hit probability."
    )

    valence = st.slider(
        "valence",
        0.0,
        1.0,
        0.50,
        key=f"val_{st.session_state['reset_nonce']}",
        help="Musical positivity (0–1). Higher is happier/cheerful; lower is sad/angry."
    )

    release_year = st.slider(
        "release_year",
        1950,
        2025,
        2020,
        key=f"year_{st.session_state['reset_nonce']}",
        help="Release year of the track. Some eras/years can correlate with hit likelihood depending on the dataset."
    )

with st.expander("Advanced (optional)", expanded=False):
    use_exact_duration = st.checkbox(
        "Enter exact duration (seconds)",
        value=False,
        key=f"exdur_{st.session_state['reset_nonce']}",
        help="If enabled, you can type an exact duration value instead of using the slider."
    )
    use_exact_followers = st.checkbox(
        "Enter exact followers",
        value=False,
        key=f"exfol_{st.session_state['reset_nonce']}",
        help="If enabled, you can type an exact follower count instead of using the slider."
    )

    if use_exact_duration:
        duration_sec = int(
            st.number_input(
                "Exact duration (seconds)",
                min_value=1,
                max_value=36000,
                value=int(duration_sec),
                step=1,
                key=f"dur_in_{st.session_state['reset_nonce']}",
                help="Exact track length in seconds."
            )
        )

    if use_exact_followers:
        followers_exact = int(
            st.number_input(
                "Exact followers",
                min_value=0,
                max_value=2_000_000_000,
                value=int(artist_followers_k * 1000),
                step=1000,
                key=f"fol_in_{st.session_state['reset_nonce']}",
                help="Exact follower count (not in K)."
            )
        )
        artist_followers_k = followers_exact // 1000

    speechiness = st.slider(
        "speechiness",
        0.0,
        1.0,
        0.05,
        key=f"sp_{st.session_state['reset_nonce']}",
        help="Presence of spoken words (0–1). Higher values often mean rap/spoken content."
    )

    acousticness = st.slider(
        "acousticness",
        0.0,
        1.0,
        0.20,
        key=f"ac_{st.session_state['reset_nonce']}",
        help="Confidence measure of whether the track is acoustic (0–1). Higher means more acoustic."
    )

    instrumentalness = st.slider(
        "instrumentalness",
        0.0,
        1.0,
        0.00,
        key=f"ins_{st.session_state['reset_nonce']}",
        help="Predicts whether a track contains no vocals (0–1). Higher means more likely instrumental."
    )

    liveness = st.slider(
        "liveness",
        0.0,
        1.0,
        0.15,
        key=f"liv_{st.session_state['reset_nonce']}",
        help="Detects presence of an audience (0–1). Higher means more likely recorded live."
    )

if "speechiness" not in locals():
    speechiness, acousticness, instrumentalness, liveness = 0.05, 0.20, 0.00, 0.15

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
for c in FEATURES:
    if c not in X.columns:
        X[c] = 0.0
X = X[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0.0)

st.divider()

if st.button("Predict", key=f"pred_{st.session_state['reset_nonce']}"):
    if hasattr(model, "predict_proba"):
        hit_prob = float(model.predict_proba(X)[:, 1][0])
    else:
        hit_prob = float(model.decision_function(X)[0])

    non_hit_prob = float(1.0 - hit_prob)

    if hit_prob >= TH:
        st.success(f"This song would be a HIT!  (Hit probability: {hit_prob:.3f})")
    else:
        st.warning(f"This song would NOT be a hit.  (Non-hit probability: {non_hit_prob:.3f})")
