import json
from Movie_Recommendation_System.dbfunction import *
from Movie_Recommendation_System.objects import *
import numpy as np
import pandas as pd
import pickle
import requests
from scipy.sparse.linalg import svds
from sklearn import linear_model
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from surprise import Reader, Dataset, SVD, evaluate




#modelAdaBoostClassifier = pickle.load(open('Movie_Recommendation_System/static/data/model_AdaBoostClassifier.pkl', 'rb'))
modelSVD = pickle.load(open('Movie_Recommendation_System/static/data/SVDModel.pkl', 'rb'))




df = loadTmdb5000()




# Predict user rating of movie
#==================================================================
def apiPredictionRating(userId, movieId):
	pre = modelSVD.predict(userId, movieId)
	
	return str(pre.est)




# Recommend movies for user
#==================================================================
# Get and sort the user's predictions
def apiRecommendationByMovie(movieId):
    reader = Reader()
    ratings = loadRatingsSmall()
    movies_list = loadMoviesMetadata()

    ratings_df = pd.DataFrame(ratings, columns = ['userId', 'movieId', 'rating', 'timestamp'], dtype = int)
    movies_df = pd.DataFrame(movies_list, columns = ['id', 'title', 'genres'])
    movies_df['id'] = movies_df['id']
    movies_df['id'] = movies_df['id'].str.replace('-','')
    movies_df.dropna(subset=["id"], axis = 0 , inplace= True)
    movies_df["id"] = movies_df["id"].astype(str).astype(int)

    R_df = ratings_df.pivot(index = 'userId', columns ='movieId', values = 'rating')
    R_df = R_df.fillna(0)
    R = R_df.as_matrix()
    user_ratings_mean = np.mean(R, axis = 1)
    R_demeaned = R - user_ratings_mean.reshape(-1, 1)    
    U, sigma, Vt = svds(R_demeaned, k = 50)
    sigma = np.diag(sigma)

    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    preds_df = pd.DataFrame(all_user_predicted_ratings, columns = R_df.columns)

    already_rated, predictions = recommend_movies(preds_df, userId, movies_df, ratings_df, 10)
    items = predictions.values.tolist()

    result = []
    for item in items:
        temp = jsonRecommendationObj()

        #temp.genres = item.get('genres')
        temp.id = item[0]
        #temp.overview = item.get('overview')
        #temp.score = item.get('score')
        temp.title = item[1]
        #temp.vote_count = item.get('vote_count')
        #temp.vote_average = item.get('vote_average')

        result.append(temp)

    return result




# Recommend movies for user
#==================================================================
def apiRecommendationByUser(userId):
    reader = Reader()
    ratings = loadRatingsSmall()
    movies_list = loadMoviesMetadata()

    ratings_df = pd.DataFrame(ratings, columns = ['userId', 'movieId', 'rating', 'timestamp'], dtype = int)
    movies_df = pd.DataFrame(movies_list, columns = ['id', 'title', 'genres'])
    movies_df['id'] = movies_df['id']
    movies_df['id'] = movies_df['id'].str.replace('-','')
    movies_df.dropna(subset=["id"], axis = 0 , inplace= True)
    movies_df["id"] = movies_df["id"].astype(str).astype(int)

    R_df = ratings_df.pivot(index = 'userId', columns ='movieId', values = 'rating')
    R_df = R_df.fillna(0)
    R = R_df.as_matrix()
    user_ratings_mean = np.mean(R, axis = 1)
    R_demeaned = R - user_ratings_mean.reshape(-1, 1)    
    U, sigma, Vt = svds(R_demeaned, k = 50)
    sigma = np.diag(sigma)

    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    preds_df = pd.DataFrame(all_user_predicted_ratings, columns = R_df.columns)

    already_rated, predictions = recommend_movies(preds_df, userId, movies_df, ratings_df, 10)
    items = predictions.values.tolist()

    result = []
    for item in items:
        temp = jsonRecommendationObj()

        #temp.genres = item.get('genres')
        temp.id = item[0]
        #temp.overview = item.get('overview')
        #temp.score = item.get('score')
        temp.title = item[1]
        #temp.vote_count = item.get('vote_count')
        #temp.vote_average = item.get('vote_average')

        result.append(temp)

    return result


# Get and sort the user's predictions
def recommend_movies(predictions_df, userID, movies_df, original_ratings_df, num_recommendations = 5):
    user_row_number = userID # UserID starts at 1, not 0
    sorted_user_predictions = predictions_df.iloc[userID].sort_values(ascending=False)
    
    # Get the user's data and merge in the movie information.
    user_data = original_ratings_df[original_ratings_df.userId == (userID)]
    user_full = (user_data.merge(movies_df, how = 'left', left_on = 'movieId', right_on = 'id')
                 .sort_values(['rating'], ascending=False))
    
    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = (movies_df[~movies_df['id'].isin(user_full['movieId'])]
                       .merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left', 
                              left_on = 'id', right_on = 'movieId')
                       .rename(columns = {user_row_number: 'Predictions'})
                       .sort_values('Predictions', ascending = False)
                       .iloc[:num_recommendations, :-1])
    
    return user_full, recommendations




# Retrieve top trending movies
#==================================================================
def apiTrending():
    C = df['vote_average'].mean()
    m = df['vote_count'].quantile(0.9)
    q_movies = df.copy().loc[df['vote_count'] >= m]
    q_movies.shape
    
    # Calculate based on the IMDB formula
    def weighted_rating(x, m=m, C=C):
        v = x['vote_count']
        R = x['vote_average']
        return (v/(v+m) * R) + (m/(m+v) * C)
    
    # Define a new feature 'score' and calculate its value with `weighted_rating()`
    q_movies['score'] = q_movies.apply(weighted_rating, axis = 1)
	# Sort movies based on score calculated above
    q_movies = q_movies.sort_values('score', ascending = False)
	# Retrieve the top 10 movies
    topten = q_movies[['id','title', 'vote_count', 'vote_average', 'score']].head(10)
    items = topten.values.tolist()

    result = []
    for item in items:
        temp = jsonTrendingObj()

        temp.id = item[0]
        temp.title = item[1]
        temp.vote_count = item[2]
        temp.vote_average = item[3]
        temp.score = item[4]

        result.append(temp)

    return result