import streamlit as st
import pandas as pd
import numpy as np
from surprise import Reader, Dataset, SVD, accuracy
from surprise.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MultiLabelBinarizer

# Load datasets
movies_data = pd.read_csv('movies.csv')
ratings_data = pd.read_csv('ratings.csv')

# Preprocessing
movies_data.dropna(inplace=True)
movies_data["genres"] = movies_data["genres"].apply(lambda x: x.split("|"))

# Merge datasets
movies_data2 = movies_data.merge(ratings_data, on='movieId', how='inner')

# Train SVD model
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings_data[["userId", "movieId", "rating"]], reader)
trainset, testset = train_test_split(data, test_size=0.2)
svd_model = SVD()
svd_model.fit(trainset)

# KNN for genre-based recommendations
mlb = MultiLabelBinarizer()
genre_features = mlb.fit_transform(movies_data["genres"])
knn_genre = NearestNeighbors(metric="cosine", algorithm="brute")
knn_genre.fit(genre_features)

# Functions
def predict_rating(user_id, movie_id):
    return svd_model.predict(user_id, movie_id).est

def recommend_movies_by_genre(movie_title, n_recommendations=5):
    if movie_title not in movies_data["title"].values:
        return ["Movie not found!"]
    movie_idx = movies_data[movies_data["title"] == movie_title].index[0]
    _, indices = knn_genre.kneighbors([genre_features[movie_idx]], n_neighbors=n_recommendations+1)
    return [movies_data.iloc[idx]["title"] for idx in indices.flatten()[1:]]

def hybrid_recommendation(user_id, movie_title, n_recommendations=5):
    if movie_title not in movies_data["title"].values:
        return ["Movie not found!"]
    movie_idx = movies_data[movies_data["title"] == movie_title].index[0]
    _, indices = knn_genre.kneighbors([genre_features[movie_idx]], n_neighbors=n_recommendations+1)
    knn_recommendations = [movies_data.iloc[idx]["movieId"] for idx in indices.flatten()[1:]]
    svd_predictions = [(movie_id, svd_model.predict(user_id, movie_id).est) for movie_id in movies_data['movieId'].unique()]
    svd_predictions.sort(key=lambda x: x[1], reverse=True)
    hybrid_recommendations = [movies_data[movies_data["movieId"] == movie_id]["title"].values[0] for movie_id, _ in svd_predictions if movie_id in knn_recommendations][:n_recommendations]
    return hybrid_recommendations

# Streamlit App
st.title('Movie Recommender System')
movie_input = selected_movie = st.selectbox("Select a Movie:", movies_data["title"].values)
user_input = st.number_input('Enter User ID:', min_value=1, max_value=1000, step=1)
if st.button('Get Recommendations'):
    st.write('Content-Based Recommendations:', recommend_movies_by_genre(movie_input))
    st.write('Collaborative Filtering Recommendation:', predict_rating(user_input, 1))  # Example movie_id = 1
    st.write('Hybrid Recommendations:', hybrid_recommendation(user_input, movie_input))
