from pathlib import Path

import pandas as pd
import pickle
import streamlit as st

st.set_page_config(page_title="Podcast Listening Time")

BASE = Path(__file__).parent


@st.cache_resource
def load_model():
    return pickle.load(open(BASE / "podcast_model.pkl", "rb"))


model = load_model()
features = list(model.feature_names_in_)


def options(prefix):
    return sorted(c[len(prefix):] for c in features if c.startswith(prefix))


DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
TIMES = ["Morning", "Afternoon", "Evening", "Night"]
days = [d for d in DAYS if d in options("Publication_Day_")] or options("Publication_Day_")
times = [t for t in TIMES if t in options("Publication_Time_")] or options("Publication_Time_")

st.title("Podcast Listening Time Predictor")
st.write(
    "A Random Forest model (trained on the Kaggle Podcast Listening Time dataset, Playground Series S5E4) "
    "predicts how many minutes a listener will spend on an episode, from the episode's metadata."
)

col1, col2 = st.columns(2)
with col1:
    podcast = st.selectbox("Podcast", options("Podcast_Name_"))
    genre = st.selectbox("Genre", options("Genre_"))
    day = st.selectbox("Publication day", days)
    time = st.selectbox("Publication time", times)
    sentiment = st.selectbox("Episode sentiment", options("Episode_Sentiment_"))
with col2:
    length = st.slider("Episode length (minutes)", 1, 120, 60)
    host_pop = st.slider("Host popularity (%)", 0, 120, 60)
    has_guest = st.checkbox("Episode has a guest", value=True)
    guest_pop = st.slider("Guest popularity (%)", 0, 120, 50, disabled=not has_guest)
    ads = st.slider("Number of ads", 0, 5, 1)

if st.button("Predict"):
    row = {c: 0 for c in features}
    row.update({
        "Episode_Length_minutes": length,
        "Host_Popularity_percentage": host_pop,
        "Guest_Popularity_percentage": guest_pop if has_guest else 0,
        "Number_of_Ads": ads,
        f"Podcast_Name_{podcast}": 1,
        f"Genre_{genre}": 1,
        f"Publication_Day_{day}": 1,
        f"Publication_Time_{time}": 1,
        f"Episode_Sentiment_{sentiment}": 1,
    })
    X = pd.DataFrame([row])[features]

    minutes = max(0.0, float(model.predict(X)[0]))
    st.success(f"Predicted listening time: **{minutes:.1f} minutes** ({minutes / length:.0%} of the episode)")
    st.progress(min(1.0, minutes / length))

st.caption(
    "Model: Random Forest (validation RMSE ≈ 13 minutes on the Kaggle data). Episode length is by far the "
    "strongest predictor: people listen to roughly 70% of an episode on average."
)
