from pathlib import Path
import re
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------------
# APP CONFIGURATION
# ------------------------------------------------------------
st.set_page_config(
    page_title="Lyric Song Prediction by Lyrics",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed",
)

APP_DIR = Path(__file__).resolve().parent
DATA_PATH = APP_DIR / "data" / "songs.csv"
REQUIRED_COLUMNS = {"artist", "song", "text"}
TOP_K = 10


# ------------------------------------------------------------
# TEXT PREPROCESSING
# ------------------------------------------------------------
def clean_text(value: str) -> str:
    """Normalize lyrics for more consistent text matching."""
    value = str(value).lower()
    value = re.sub(r"[^a-z0-9\s']", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


# ------------------------------------------------------------
# LOAD DATASET AND BUILD SEARCH INDEX ONCE
# Streamlit caches this resource so users do not upload the CSV.
# ------------------------------------------------------------
@st.cache_resource(show_spinner="Loading songs and preparing the search engine…")
def load_search_engine(csv_path: str, modified_time: float):
    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            "The CSV is missing required columns: " + ", ".join(sorted(missing))
        )

    df = df[["artist", "song", "text"]].copy()
    df = df.dropna(subset=["text"])
    df["artist"] = df["artist"].fillna("Unknown artist").astype(str)
    df["song"] = df["song"].fillna("Unknown song").astype(str)
    df["text"] = df["text"].astype(str)
    df["clean_lyrics"] = df["text"].map(clean_text)
    df = df[df["clean_lyrics"].str.len() > 0].reset_index(drop=True)

    if df.empty:
        raise ValueError("No usable lyrics were found in the CSV.")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=30000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        dtype=np.float32,
    )
    lyrics_matrix = vectorizer.fit_transform(df["clean_lyrics"])

    return df, vectorizer, lyrics_matrix


# ------------------------------------------------------------
# SEARCH
# ------------------------------------------------------------
def find_songs(query: str, engine, top_k: int = TOP_K):
    df, vectorizer, lyrics_matrix = engine
    cleaned_query = clean_text(query)

    if len(cleaned_query) < 12 or len(cleaned_query.split()) < 3:
        return None, "Please enter at least 3 words and 12 characters. Longer, distinctive lyrics usually work better."

    query_vector = vectorizer.transform([cleaned_query])
    scores = cosine_similarity(query_vector, lyrics_matrix).ravel()

    if scores.size == 0 or float(scores.max()) <= 0:
        return pd.DataFrame(), "No matching words were found in this dataset."

    # Sort from highest similarity to lowest similarity.
    ranked_indices = np.argsort(scores)[::-1]

    results = []
    seen_songs = set()

    for index in ranked_indices:
        score = float(scores[index])
        if score <= 0:
            break

        row = df.iloc[int(index)]
        # Avoid showing the same song/artist combination repeatedly.
        unique_key = (
            row["song"].strip().casefold(),
            row["artist"].strip().casefold(),
        )
        if unique_key in seen_songs:
            continue

        seen_songs.add(unique_key)
        results.append(
            {
                "Song": row["song"],
                "Artist": row["artist"],
                "Similarity": round(score * 100, 1),
            }
        )

        if len(results) >= top_k:
            break

    result_df = pd.DataFrame(results)
    return result_df, f"Found {len(result_df)} ranked result(s)."


# ------------------------------------------------------------
# PAGE DESIGN
# ------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 12% 5%, rgba(124, 58, 237, .20), transparent 28%),
            radial-gradient(circle at 90% 28%, rgba(6, 182, 212, .12), transparent 25%),
            #090910;
        color: #f7f5ff;
    }
    .block-container { max-width: 1050px; padding-top: 2rem; }
    .brand { color: #67e8f9; font-weight: 800; letter-spacing: 3px; font-size: .85rem; }
    .hero-title { font-size: clamp(2.5rem, 6vw, 4.7rem); line-height: 1.04; font-weight: 800; letter-spacing: -2px; }
    .hero-gradient {
        background: linear-gradient(90deg, #a78bfa, #67e8f9);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .muted { color: #aaa7bd; }
    div.stButton > button {
        background: linear-gradient(100deg, #a78bfa, #67e8f9);
        color: #090910; border: 0; border-radius: 10px; font-weight: 800;
        min-height: 46px;
    }
    div.stButton > button:hover { border: 0; color: #090910; filter: brightness(1.08); }
    div[data-testid="stTextArea"] textarea {
        background: #151522; color: #fff; border: 1px solid #39364f; border-radius: 12px;
    }
    div[data-testid="stDataFrame"] { border: 1px solid #333149; border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="brand">♪ &nbsp; LYRICFIND · LYRICS SEARCH ENGINE</div>', unsafe_allow_html=True)
st.write("")
st.markdown(
    '<div class="hero-title">Find the song<br><span class="hero-gradient">behind the lyrics.</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="muted">Paste a few lines from a song. The search engine compares your words '
    'with the built-in lyrics dataset and ranks the closest matches.</p>',
    unsafe_allow_html=True,
)

# The CSV is part of the deployed project: no visitor upload is needed.
if not DATA_PATH.exists():
    st.error(
        "Dataset file not found. The project owner must place the CSV at "
        "`data/songs.csv` before deploying."
    )
    st.stop()

try:
    engine = load_search_engine(str(DATA_PATH), DATA_PATH.stat().st_mtime)
    songs_df = engine[0]
except Exception as error:
    st.error(f"Could not prepare the song dataset: {error}")
    st.stop()

st.caption(f"🟢 Dataset ready · {len(songs_df):,} songs indexed")

with st.form("lyrics_search_form"):
    query = st.text_area(
        "YOUR LYRICS",
        placeholder="Type or paste at least 3 words from the song…",
        height=145,
        help="Use a longer and more distinctive lyric passage for better matching.",
    )
    submitted = st.form_submit_button("Find my song  →", use_container_width=True)

if submitted:
    results, message = find_songs(query, engine)
    if results is None:
        st.warning(message)
    elif results.empty:
        st.info(message)
    else:
        st.success(message)
        st.subheader("Closest matches")
        display = results.copy()
        display["Similarity"] = display["Similarity"].map(lambda value: f"{value:.1f}%")
        st.dataframe(display, use_container_width=True, hide_index=True)
        st.caption(
            "Similarity is a text-matching score, not a probability or a guarantee "
            "that the song is the correct one. Results are limited to this dataset."
        )

with st.expander("About this project"):
    st.markdown(
        """
        **Lyric Song Prediction by Lyrics** is a lyrics-based song search application.

        - **Python:** application programming language
        - **Pandas:** loads and manages the CSV dataset
        - **Scikit-learn / TF-IDF:** converts lyrics into numerical vectors
        - **Cosine Similarity:** compares the user's lyrics with dataset lyrics
        - **Streamlit:** creates this interactive web application

        This application searches the included CSV dataset. It does not identify songs
        from audio and does not search every song on the internet.
        """
    )

st.markdown("---")
st.caption("LYRIC SONG PREDICTION BY LYRICS  ·  Python · TF-IDF · Cosine Similarity")
