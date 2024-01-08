# BERT-For-Music-Recommend


## Overview
This repository hosts a collection of Python scripts and Jupyter Notebooks aimed at providing music recommendations using BERT (Bidirectional Encoder Representations from Transformers) and Spotify's Web API.

### Primary Python Scripts
1. **BERT_music_recomend_web.py**: A Streamlit web application script that integrates with Spotify's Web API for music recommendations.
2. **music_serch.py**: A script for data processing, possibly involving machine learning techniques using BERT models and other data handling tasks.


## Demo Video
Watch a demo of the project in action: [BERT Music Recommendation Demo](https://youtu.be/54a0q762CI0).

<iframe width="560" height="315" src="https://www.youtube.com/embed/54a0q762CI0?si=DmwI05pGTOcHvxw4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>


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
