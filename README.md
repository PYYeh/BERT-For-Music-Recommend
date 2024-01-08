# BERT-For-Music-Recommend


**Overview:**
This project focuses on creating a music recommendation system that leverages Google's BERT (Bidirectional Encoder Representations from Transformers) language model for analyzing user chat content. By interpreting the nuances of human conversation through advanced NLP techniques, the system aims to recommend music tracks that resonate with the mood, themes, or specific keywords identified in the chat.

**Key Components:**

1. **BERT-Based Text Analysis:**
    - Implementing Google's BERT model to understand and analyze the chat content.
    - BERT's bidirectional training allows for a deeper comprehension of context and subtleties in user conversations.
2. **Music Database Integration:**
    - Creating a comprehensive database of music tracks categorized by genre, mood, lyrics, and other relevant attributes.
    - Each track is treated as a 'document' for indexing and retrieval purposes.
3. **Chat-to-Music Matching Algorithm:**
    - Developing an algorithm that matches the insights gained from BERT analysis with the music database.
    - This involves determining the relevance of each track to the conversation's context and emotional tone.
4. **Relevance Ranking and Personalization:**
    - Ranking the retrieved music tracks based on their relevance to the analyzed chat content.
    - Incorporating user-specific data for personalizing recommendations, thereby enhancing user experience.
5. **User Interface Design:**
    - Designing an intuitive and user-friendly interface for displaying recommended music tracks.
    - Providing options for users to give feedback, which helps in refining future recommendations.


## Demo Video
Watch a demo of the project in action: [BERT Music Recommendation Demo](https://youtu.be/54a0q762CI0).

<iframe width="560" height="315" src="https://www.youtube.com/embed/54a0q762CI0?si=DmwI05pGTOcHvxw4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>


### Primary Python Scripts
1. **BERT_music_recomend_web.py**: A Streamlit web application script that integrates with Spotify's Web API for music recommendations.
2. **music_serch.py**: A script for data processing, possibly involving machine learning techniques using BERT models and other data handling tasks.


## To Run The Demo
1. Install the required Python libraries:
   ```bash
   pip install streamlit spotipy pandas numpy scikit-learn transformers jupyter
   ```

2. Set Spotify API credentials (Client ID and Client Secret) in `BERT_music_recomend_web.py`.

3. To run the Streamlit app:
   ```bash
   streamlit run BERT_music_recomend_web.py
   ```

4. To explore the Jupyter Notebooks:
   - Launch Jupyter Notebook or JupyterLab.
   - Navigate to the notebook files and open them.

## Usage
- For the Streamlit app: Launch the web app and follow the on-screen instructions for music recommendations.
- For the Jupyter Notebooks: Execute the cells in the notebooks to perform data analyses and explore music recommendation models.


### Jupyter Notebooks
The repository also includes three Jupyter Notebooks, which likely contain explorations and the data analyses of the project:
1. **music_kmens.ipynb**
2. **music_mood.ipynb**
3. **music_serch.ipynb**
4. **emotion_recon.ipynb**
5. **fine_tune_bert_for_text_classification.ipynb**


## Dependencies
- Streamlit
- Spotipy
- Pandas
- NumPy
- Scikit-learn
- Transformers (Hugging Face)
- Other common Python libraries for data science and machine learning (for Jupyter Notebooks)


## References
- BERT model for emotion recognition: [Hugging Face Model](https://huggingface.co/bhadresh-savani/bert-base-uncased-emotion?text=I+like+you.+I+love+you)
- Sentiment Analysis with Python: [Hugging Face Blog](https://huggingface.co/blog/sentiment-analysis-python)
- YouTube Tutorials and Datasets:
  - Content-Based Filtering - [YouTube Video](https://www.youtube.com/watch?v=uDzLxos0lNU&t=555s)
  - User-Related Recommendations - [YouTube Video](https://www.youtube.com/watch?v=gaZKjAKfe0s)
  - Content-Related Recommendations - [YouTube Video](https://www.youtube.com/watch?v=jm9JamrbSv8), [Spotify Million Song Dataset on Kaggle](https://www.kaggle.com/datasets/notshrirang/spotify-million-song-dataset)
  - Music Genre Classification - [YouTube Video](https://www.youtube.com/watch?v=doUTqWUAuDw), [GTZAN Dataset on Kaggle](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification/data)
  - Chat Data Set for Emotion Detection - [Hugging Face Dataset](https://huggingface.co/datasets/daily_dialog)
- Music Mood Classification: [Tufts University Research](https://sites.tufts.edu/eeseniordesignhandbook/2015/music-mood-classification/?source=post_page-----b2dda2bf455--------------------------------)
- Neutral Emotion BERT Model: [Hugging Face Model](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)
