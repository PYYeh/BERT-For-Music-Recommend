import streamlit as st
import music_serch
import numpy as np
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

CLIENT_ID = "c53de52bc5854f37a14b15f999a5d9a0"
CLIENT_SECRET = "e136aba84789469f9ce42e3207aca56a"

# Initialize the Spotify client
client_credentials_manager = SpotifyClientCredentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET)
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

def get_song_album_cover_url(song_name, artist_name):
    search_query = f"track:{song_name} artist:{artist_name}"
    results = sp.search(q=search_query, type="track")

    if results and results["tracks"]["items"]:
        track = results["tracks"]["items"][0]
        album_cover_url = track["album"]["images"][0]["url"]
        print(album_cover_url)
        return album_cover_url
    else:
        return "https://i.postimg.cc/0QNxYz4V/social.png"

def recommend(result_df):
    recommended_music_names = []
    recommended_music_posters = []
    for i in range(10):
        # fetch the movie poster
        artist = result_df.iloc[i].artist
        print(artist)
        print(result_df.iloc[i].song)
        recommended_music_posters.append(get_song_album_cover_url(result_df.iloc[i].song, artist))
        recommended_music_names.append(result_df.iloc[i].song)

    return recommended_music_names,recommended_music_posters


# Function to load and setup classifiers
@st.cache_resource
def setup_classifiers():
    return music_serch.setup_classifiers()

emotion_classifier, sentiment_classifier = setup_classifiers()

st.header('BERT Music Recommender System')

# Text input
user_input = st.text_area("Enter your text", "I love using transformers.")

# Emotion ratio sliders
sadness = st.slider('Sadness', 0.0, 1.0, 0.5)
joy = st.slider('Joy', 0.0, 1.0, 0.5)
love = st.slider('Love', 0.0, 1.0, 0.5)
anger = st.slider('Anger', 0.0, 1.0, 0.5)
fear = st.slider('Fear', 0.0, 1.0, 0.5)
surprise = st.slider('Surprise', 0.0, 1.0, 0.5)

emotion_ratios = np.array([sadness, joy, love, anger, fear, surprise])


# Function for music recommendation
def music_recomend(text, ratios, classifier1, classifier2):
    search_df, song_category = music_serch.music_recomend(text, ratios, classifier1, classifier2)
    return search_df, song_category

# Process input and display recommendations
if st.button('Recommend Songs'):
    search_df, song_category = music_recomend(user_input, emotion_ratios, emotion_classifier, sentiment_classifier)
    st.write(f"You might like songs from the '{song_category}' genre.")
    st.write("Recommended Songs:")
    st.write(search_df)

    recommended_music_names, recommended_music_posters = recommend(search_df)
    col1, col2, col3, col4, col5= st.columns(5)
    with col1:
        st.text(recommended_music_names[0])
        st.image(recommended_music_posters[0])
    with col2:
        st.text(recommended_music_names[1])
        st.image(recommended_music_posters[1])
    with col3:
        st.text(recommended_music_names[2])
        st.image(recommended_music_posters[2])
    with col4:
        st.text(recommended_music_names[3])
        st.image(recommended_music_posters[3])
    with col5:
        st.text(recommended_music_names[4])
        st.image(recommended_music_posters[4])
