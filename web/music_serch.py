import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from transformers import pipeline

loaded_data = None

# 數據載入函數
def load_data():
    global loaded_data
    if loaded_data is None:
        o_data = pd.read_csv('../data/spotify_millsongdata.csv')
        c_data = pd.read_csv("../data/spotify_millsongdata_clustered_data.csv")
        loaded_data = (o_data, c_data)
    return loaded_data

# 設置情感和情緒分類的模型
def setup_classifiers():
    emotion_classifier = pipeline("text-classification", model="bhadresh-savani/bert-base-uncased-emotion")
    sentiment_classifier = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment")
    return emotion_classifier, sentiment_classifier

# 映射情感和情緒分類到歌曲類別
def map_emotion_sentiment_to_song_category(emotion_prediction, sentiment_prediction):
    # 情感到歌曲類別的映射
    sentiment_mapping = {'LABEL_0': 'Anxious/Sad', 'LABEL_1': 'Neutral', 'LABEL_2': 'Happy'}
    emotion_mapping = {'sadness': 'Anxious/Sad', 'joy': 'Happy', 'love': 'Happy', 'anger': 'Anxious/Sad', 'fear': 'Anxious/Sad', 'surprise': 'Energetic'}
    
    top_emotion = max(emotion_prediction, key=lambda x: x['score'])['label']
    top_sentiment = max(sentiment_prediction, key=lambda x: x['score'])['label']

    if sentiment_mapping[top_sentiment] == 'Neutral':
        return 'Calm'
    return emotion_mapping.get(top_emotion, 'Unknown')

def extract_songs_by_cluster(data, cluster_name):
    return data[data['Cluster Name'] == cluster_name]

# 推薦歌曲
def recommend_songs(text, emotion_classifier, sentiment_classifier, c_data):
    emotion_prediction = emotion_classifier(text)
    sentiment_prediction = sentiment_classifier(text)
    song_category = map_emotion_sentiment_to_song_category(emotion_prediction, sentiment_prediction)
    return extract_songs_by_cluster(c_data, song_category), song_category

# 使用 KNN 找到最接近的情感比例歌曲
def find_closest_songs(emotion_ratios, knn_data):
    query_point = emotion_ratios.reshape(1, -1)
    nn = NearestNeighbors(n_neighbors=10)
    nn.fit(knn_data)
    distances, indices = nn.kneighbors(query_point)
    return indices, distances[0]

# 主程式
def music_recomend(text, emotion_ratios, emotion_classifier, sentiment_classifier):
    # 數據載入
    o_data, c_data = load_data()

    # 使用已初始化的分類器
    recommended_songs_list, song_category = recommend_songs(text, emotion_classifier, sentiment_classifier, c_data)

    knn_data = recommended_songs_list.drop(columns=['Cluster', 'Cluster Name'])
    closest_songs, distances = find_closest_songs(emotion_ratios, knn_data)
    closest_emotion_song = knn_data.iloc[closest_songs[0]]
    song_indices = closest_emotion_song.index
    #print(song_indices)
    search_df = o_data.iloc[song_indices]
    search_df.rename(columns={'text': 'lyrics'}, inplace=True)
    search_df.drop('link', axis=1, inplace=True)

    return search_df, song_category


if __name__ == "__main__":
    # 初始化模型
    emotion_classifier, sentiment_classifier = setup_classifiers()
    emotion_ratios = np.array([0.97132064, 0.92075195, 0.93364823, 0.94880388, 0.0850701, 0.22479665])
    text = "I love using transformers. The best part is wide range of support and its easy to use"
    search_df, song_category = music_recomend(text, emotion_ratios, emotion_classifier, sentiment_classifier)
    print(song_category)
    print(search_df)
