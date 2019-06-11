import ast
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




cols = ["genres", "id", "overview", "tagline", "title", "vote_count", "vote_average", 'imdb_id']

#modelAdaBoostClassifier = pickle.load(open('Movie_Recommendation_System/static/data/model_AdaBoostClassifier.pkl', 'rb'))
modelSVD = pickle.load(open('Movie_Recommendation_System/static/data/SVDModel.pkl', 'rb'))




# Predict user rating of movie
#==================================================================
def apiPredictionRating(userId, movieId):
	pre = modelSVD.predict(userId, movieId)
	
	return str(pre.est)




# Recommend similar movies
#==================================================================
def apiRecommendationByMovie(movieId):
    similarMovieList_Keyword_genre, movie_genres_keyword_score = getSimilarMovieKeywords(movieId)
    similarMovieList_Keyword_genre = similarMovieList_Keyword_genre[similarMovieList_Keyword_genre.score.notnull()]
    similarMovieList_Keyword_genre = similarMovieList_Keyword_genre.sort_values(by='score').tail(10)
    print(similarMovieList_Keyword_genre.columns)
    
    items = similarMovieList_Keyword_genre.values.tolist()
    print(items[0])
    result = []
    for item in items:
        temp = movieObj()

        temp.genres = item[0]
        temp.id = item[1]
        temp.imdbId = item[7]
        temp.overview = item[2]
        #temp.score = item[3]
        temp.tags = item[3]
        temp.title = item[4]
        temp.vote_count = item[5]
        temp.vote_average = item[6]

        result.append(temp)

    return result


# Find similar movies by keywords
def getSimilarMovieKeywords(movie_id):    
    # Data Impulation: keywords
    df_keyword = loadKeywords()
    df_keyword = df_keyword.head(2000)

    # Data Impulation: movie_metadata
    df_movie_meta = loadMoviesMetadata()
    df_movie_meta = df_movie_meta[cols].head(2000)    
    df_movie_meta['id'] = df_movie_meta['id'].str.replace('-','')
    df_movie_meta.dropna(subset=["id"], axis = 0 , inplace= True)
    df_movie_meta["id"] = df_movie_meta["id"].astype(str).astype(int)
    df_movie_meta= df_movie_meta.merge(df_keyword,on='id')
    df_movie_meta.set_index('id')

    # Parse the stringified features into their corresponding python objects
    df_movie_meta['keywords'] = df_movie_meta['keywords'].apply(ast.literal_eval)
    df_movie_meta['keywords'] = df_movie_meta['keywords'].apply(get_list)

    movie_genres_keyword_score = endcode_keywords(df_movie_meta)
    movie_genres_keyword_score = movie_genres_keyword_score.drop(['keywords'], axis=1)
    movie_genres_keyword_score = endcode_genre(movie_genres_keyword_score)
    movie_genres_keyword_score = movie_genres_keyword_score.drop(['genres'], axis=1)
    movie_genres_keyword_score["id"] = movie_genres_keyword_score["id"].astype(str).astype(int)
  
    movie_item = np.array(movie_genres_keyword_score.loc[movie_genres_keyword_score.id == movie_id])
    if len(movie_item) > 0:
        movie_item = np.delete(movie_item, 0)
    else:
        movie_item = np.array(movie_genres_keyword_score.loc[movie_genres_keyword_score.id == 0])
        movie_item = np.delete(movie_item, 0)


    pearsonObj = [pearson(np.array(movie_genres_keyword_score.iloc[i, 1:]), movie_item) 
                  for i in range(movie_genres_keyword_score.shape[0])]
    similarity_value = np.array(pearsonObj)
   

    df_movie_meta["score"] = similarity_value
    return df_movie_meta, movie_genres_keyword_score


# Return the list of top 3 elements or all; whichever is more.
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    #Return empty list in case of missing/malformed data
    return []


# Parse json and separate columns with keywords of movie
def endcode_keywords(df):
    df_movies_keyword = df[['id','genres','keywords']]
    count = df_movies_keyword["keywords"]
    
    for index in range(len(count)):
       for item in count[index]:
        name = "kw_" + item
        if {name}.issubset(df_movies_keyword.columns):
        	df_movies_keyword.at[index,name] = 1
        else:
            df_movies_keyword[name] = 0
            df_movies_keyword.at[index,name] = 1

    return df_movies_keyword


def endcode_genre(df_movies_genre):
    genres = df_movies_genre["genres"]
    for index in range(len(genres)):
        item = ast.literal_eval(genres[index])
        for j in item:
            j = str(j).replace("'", '"')
            json_data = json.loads(j)
            name = "genres_" + str(json_data["id"]) + "_" + str(json_data["name"])
            if {name}.issubset(df_movies_genre.columns):
                df_movies_genre.at[index,name] = 1
            else:
                df_movies_genre[name] = 0
                df_movies_genre.at[index,name] = 1

    return df_movies_genre


def pearson(u, v):
    mean_u = mean(u)
    mean_v = mean(v)
    
    specified_rating_indices_u = set(specified_rating_indices(u)[0])
    specified_rating_indices_v = set(specified_rating_indices(v)[0])
    
    mutually_specified_ratings_indices = specified_rating_indices_u.intersection(specified_rating_indices_v)
    mutually_specified_ratings_indices = list(mutually_specified_ratings_indices)
    
    u_mutually = u[mutually_specified_ratings_indices]
    v_mutually = v[mutually_specified_ratings_indices]
    
    centralized_mutually_u = u_mutually - mean_u
    centralized_mutually_v = v_mutually - mean_v

    result = np.sum(np.multiply(centralized_mutually_u, centralized_mutually_v))
    sum_u = np.sqrt(np.sum(np.square(centralized_mutually_u)))
    sum_v = np.sqrt(np.sum(np.square(centralized_mutually_v)))
    result = result / (sum_u * sum_v)

    return result


def mean(u):
    # may use specified_rating_indices but use more time
    specified_ratings = u[specified_rating_indices(u)]
    m = sum(specified_ratings)/np.shape(specified_ratings)[0]

    return m


# Indice for vector
def specified_rating_indices(u):
    return list(map(tuple, np.where(np.isfinite(u))))




# Recommend movies for user
#==================================================================
def apiRecommendationByUser(userId):
    reader = Reader()
    ratings = loadRatingsSmall()
    movies_list = loadMoviesMetadata()

    ratings_df = pd.DataFrame(ratings, columns = ['userId', 'movieId', 'rating', 'timestamp'], dtype = int)
    movies_df = pd.DataFrame(movies_list, columns = cols)
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
    print(items[0])

    result = []
    for item in items:
        temp = movieObj()

        temp.genres = item[0]
        temp.id = item[1]
        temp.imdbId = item[7]
        temp.overview = item[2]
        #temp.score = item[3]
        temp.tags = item[3]
        temp.title = item[4]
        temp.vote_count = item[5]
        temp.vote_average = item[6]

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
    df = loadTmdb5000()
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
    topten = q_movies[["genres", "id", "overview", "score", "tagline", "title", "vote_count", "vote_average"]].head(10)

    items = topten.values.tolist()
    result = []
    for item in items:
        temp = movieObj()

        temp.genres = item[0]
        temp.id = item[1]
        temp.overview = item[2]
        temp.score = item[3]
        temp.tags = item[4]
        temp.title = item[5]
        temp.vote_count = item[6]
        temp.vote_average = item[7]

        result.append(temp)

    return result