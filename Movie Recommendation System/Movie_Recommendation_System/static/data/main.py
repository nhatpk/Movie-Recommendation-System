#!flask/bin/python
import os
from flask import Flask
from flask import request
import pandas as pd
from sklearn import linear_model
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


import pickle

def loadInitData():
	import numpy as np
	df1=pd.read_csv('data_movies/tmdb-5000-movie-dataset/tmdb_5000_credits.csv')
	df2=pd.read_csv('data_movies/tmdb-5000-movie-dataset/tmdb_5000_movies.csv')
	df1.columns = ['id','tittle','cast','crew']
	df2= df2.merge(df1,on='id')
	return df2

app = Flask(__name__)

@app.route('/isAlive')
def index():
    return "true"


@app.route('/prediction/api/v1.0/prediction', methods=['GET'])
def get_loanprediction():
    feature1 = float(request.args.get('f1'))
    feature2 = float(request.args.get('f2'))
    feature3 = float(request.args.get('f3'))
    feature4 = float(request.args.get('f4'))
    loaded_model = pickle.load(open('model_AdaBoostClassifier.pkl', 'rb'))
    data =[0.49565217391304345,0.49565217391304345,0.5017171494285714,36]
    #prediction = loaded_model.predict([data])
    prediction = loaded_model.predict([[feature1, feature2, feature3, feature4]])
    return str(prediction)
    
#/prediction/api/v1.0/topTenMovies
@app.route('/prediction/api/v1.0/topTenMovies', methods=['GET'])
def Demographic11():
	df2 = loadInitData()
	C= df2['vote_average'].mean()
	m= df2['vote_count'].quantile(0.9)
	q_movies = df2.copy().loc[df2['vote_count'] >= m]
	q_movies.shape
	def weighted_rating(x, m=m, C=C):
		v = x['vote_count']
		R = x['vote_average']
		# Calculation based on the IMDB formula
		return (v/(v+m) * R) + (m/(m+v) * C)
	
	# Define a new feature 'score' and calculate its value with `weighted_rating()`
	q_movies['score'] = q_movies.apply(weighted_rating, axis=1)
	#Sort movies based on score calculated above
	q_movies = q_movies.sort_values('score', ascending=False)
	#Print the top 15 movies
	topten = q_movies[['id','title', 'vote_count', 'vote_average', 'score']].head(10)
	data = topten.to_json(orient='records')
	return data
	
	
def endcode_genre(df_movies_genre):
    import json
    import ast
    countGener = df_movies_genre["genres"]
    print(len(countGener))
    for index in range(len(countGener)):
        item = ast.literal_eval(countGener[index])
        for j in item:
            j = str(j).replace("'", '"')
            json_data = json.loads(j)
            name = "genres_" + str(json_data["id"]) + "_" + str(json_data["name"])
            #print(name)
            if {name}.issubset(df_movies_genre.columns):
                df_movies_genre.at[index,name] = 1
            else:
                df_movies_genre[name] = 0
                df_movies_genre.at[index,name] = 1
    return df_movies_genre
def mean(u):
    # may use specified_rating_indices but use more time
    #print(specified_rating_indices(u))
    specified_ratings = u[specified_rating_indices(u)]
    #u[np.isfinite(u)]
    m = sum(specified_ratings)/np.shape(specified_ratings)[0]
    return m

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
    result = result / (np.sqrt(np.sum(np.square(centralized_mutually_u))) * np.sqrt(np.sum(np.square(centralized_mutually_v))))

    return result


# indices for vector
def specified_rating_indices(u):
    return list(map(tuple, np.where(np.isfinite(u))))

#get similar movies
def get_movie_similarity_value_for(movie_index, movie_matrix):
   # print(movies_matrix.loc[movies_matrix.id == 862])
   
    movie_item = np.array(movie_matrix.loc[movies_matrix.id == 862])
    movie_item = np.delete(movie_item, 0)
    #print(movie_item)
    #print(np.array(movie_matrix.iloc[0, 1:]))
    #print(pearson(movie_matrix.iloc[0, 1:], movie_item))
    similarity_value = np.array([pearson(np.array(movie_matrix.iloc[i, 1:]), movie_item) for i in range(movie_matrix.shape[0])])
    return similarity_value
    
#parse json and separate columns with keywords of movie

def endcode_keywords(df2):
    import json
    import ast
    df_movies_keyword = df2[['id','genres','keywords']]
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

# Returns the list top 3 elements or entire list; whichever is more.
def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > 3:
            names = names[:3]
        return names

    #Return empty list in case of missing/malformed data
    return []
    
def getSimilarMovieKeywords(movie_id):
 
    import ast
    df_movie_meta=pd.read_csv('data_movies/the-movies-dataset/movies_metadata.csv')
    df_keyword =pd.read_csv('data_movies/the-movies-dataset/keywords.csv')
    cols = ["id","genres", "title", "overview", "vote_average", "vote_count"]
    df_movie_meta = df_movie_meta[cols].head(2000)
    df_keyword = df_keyword.head(2000)
    
    df_movie_meta['id'] = df_movie_meta['id'].str.replace('-','')
    df_movie_meta.dropna(subset=["id"], axis = 0 , inplace= True)
    df_movie_meta["id"] = df_movie_meta["id"].astype(str).astype(int)
    df_movie_meta= df_movie_meta.merge(df_keyword,on='id')
    df_movie_meta.set_index('id')

    # Parse the stringified features into their corresponding python objects
    #from ast import literal_eval
    df_movie_meta['keywords'] = df_movie_meta['keywords'].apply(ast.literal_eval)
    df_movie_meta['keywords'] = df_movie_meta['keywords'].apply(get_list)
    #print(df_movie_meta.shape())
    movie_genres_keyword_score = endcode_keywords(df_movie_meta)
    movie_genres_keyword_score = movie_genres_keyword_score.drop(['keywords'], axis=1)
    movie_genres_keyword_score = endcode_genre(movie_genres_keyword_score)
    movie_genres_keyword_score = movie_genres_keyword_score.drop(['genres'], axis=1)
    movie_genres_keyword_score["id"] = movie_genres_keyword_score["id"].astype(str).astype(int)
  
    movie_item = np.array(movie_genres_keyword_score.loc[movie_genres_keyword_score.id == movie_id])
    movie_item = np.delete(movie_item, 0)
    similarity_value = np.array([pearson(np.array(movie_genres_keyword_score.iloc[i, 1:]), movie_item) for i in range(movie_genres_keyword_score.shape[0])])
   

    df_movie_meta["score"] = similarity_value
    return df_movie_meta, movie_genres_keyword_score
    
    
#/prediction/api/v1.0/similarMovies?movieId=862               	
@app.route('/prediction/api/v1.0/similarMovies', methods=['GET'])
def SimilarMovie():
	movieId = float(request.args.get('movieId'))
	similarMovieList_Keyword_genre, movie_genres_keyword_score =  getSimilarMovieKeywords(movieId)
	similarMovieList_Keyword_genre = similarMovieList_Keyword_genre[similarMovieList_Keyword_genre.score.notnull()]
	similarMovieList_Keyword_genre = similarMovieList_Keyword_genre.sort_values(by='score').tail(10)
	resultjson = similarMovieList_Keyword_genre.to_json(orient='records')
	return resultjson

 #/prediction/api/v1.0/predictUserMovieRating?userId=4&movieId=302
@app.route('/prediction/api/v1.0/predictUserMovieRating', methods=['GET'])
def predictUserMovie():
	movieId = request.args.get('movieId')
	userId = request.args.get('userId')
	svdModel = pickle.load(open('SVDModel.pkl', 'rb'))
	pre = svdModel.predict(userId, movieId)
	
	return str(pre.est)

#/prediction/api/v1.0/recommendMoviesList?userId=2

@app.route('/prediction/api/v1.0/recommendMoviesList', methods=['GET'])
def getMovieListRecommendation():
	
    import pandas as pd
    import numpy as np
    from surprise import Reader, Dataset, SVD, evaluate
	
    userId = int(request.args.get('userId'))
    reader = Reader()
    ratings=pd.read_csv('data_movies/the-movies-dataset/ratings_small.csv')
    movies_list=pd.read_csv('data_movies/the-movies-dataset/movies_metadata.csv')

    ratings_df = pd.DataFrame(ratings, columns = ['userId', 'movieId', 'rating', 'timestamp'], dtype = int)
    #data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
    movies_df = pd.DataFrame(movies_list, columns = ['id', 'title', 'genres'])
    movies_df['id'] = movies_df['id']
    movies_df['id'] = movies_df['id'].str.replace('-','')
    movies_df.dropna(subset=["id"], axis = 0 , inplace= True)
    movies_df["id"] = movies_df["id"].astype(str).astype(int)
    #movies_df['id'] = movies_df['id'].apply(pd.to_numeric)

    R_df = ratings_df.pivot(index = 'userId', columns ='movieId', values = 'rating')
    R_df=R_df.fillna(0) 

    #R_df = R_df.fillna(R_df.mean()) # Replace the na with column mean (Movie mean)

    R = R_df.as_matrix()
    user_ratings_mean = np.mean(R, axis = 1)
    R_demeaned = R - user_ratings_mean.reshape(-1, 1)

    
    from scipy.sparse.linalg import svds
    U, sigma, Vt = svds(R_demeaned, k = 50)

    sigma = np.diag(sigma)

    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    preds_df = pd.DataFrame(all_user_predicted_ratings, columns = R_df.columns)



    def recommend_movies(predictions_df, userID, movies_df, original_ratings_df, num_recommendations=5):

       # Get and sort the user's predictions
        user_row_number = userID # UserID starts at 1, not 0
        sorted_user_predictions = predictions_df.iloc[userID].sort_values(ascending=False)
        #print (sorted_user_predictions.head(10))
        print (list(pd.DataFrame(sorted_user_predictions).columns))

        # Get the user's data and merge in the movie information.
        user_data = original_ratings_df[original_ratings_df.userId == (userID)]
        user_full = (user_data.merge(movies_df, how = 'left', left_on = 'movieId', right_on = 'id').
                         sort_values(['rating'], ascending=False))

        #print ('User {0} has already rated {1} movies.'.format(userID, user_full.shape[0]))
        #print ('Recommending the highest {0} predicted ratings movies not already rated.'.format(num_recommendations))

        # Recommend the highest predicted rating movies that the user hasn't seen yet.
        recommendations = (movies_df[~movies_df['id'].isin(user_full['movieId'])].
             merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left',
                   left_on = 'id',
                   right_on = 'movieId').
             rename(columns = {user_row_number: 'Predictions'}).
             sort_values('Predictions', ascending = False).
                           iloc[:num_recommendations, :-1]
                          )

        return user_full, recommendations


    already_rated, predictions = recommend_movies(preds_df, userId, movies_df, ratings_df, 10)
    result = predictions.to_json(orient='records')
    return result
   
if __name__ == '__main__':
	app.run(port=7000,host='0.0.0.0')


