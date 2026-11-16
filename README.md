# Lyric Song Prediction by Lyrics

A Python-only (Streamlit UI) lyrics search website. The CSV dataset is bundled in
`data/songs.csv`, so website visitors do **not** upload the dataset.

## Run on your computer

1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the website:

   ```bash
   streamlit run app.py
   ```

5. Open the local URL shown in the terminal (usually http://localhost:8501).

## Publish it online (Streamlit Community Cloud)

1. Create a GitHub repository.
2. Upload `app.py`, `requirements.txt`, `README.md`, and the `data` folder containing `songs.csv`.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Choose **Create app**, select your repository and branch, and set the main file to `app.py`.
5. Deploy. Streamlit gives you a public URL.

The dataset is about 73 MB, below GitHub's 100 MB per-file limit, but check your
repository/account limits and the dataset's license/redistribution permissions.
Do not use Git LFS unless you have confirmed that your hosting setup supports it.

## Dataset handling

The app automatically reads `data/songs.csv` and caches the search index with
`st.cache_resource`. Visitors never upload the CSV. The index is built when the
app first starts and reused while the app process remains alive.

## Accuracy and limitations

- Cleans punctuation and repeated whitespace.
- Uses TF-IDF with single words and two-word phrases.
- Ranks matches with cosine similarity.
- Suppresses repeated song/artist entries in the displayed results.
- Requests at least three words and twelve characters to reduce weak searches.
- Similarity scores are not probabilities.
- It searches only the supplied dataset, not the whole internet or audio.
