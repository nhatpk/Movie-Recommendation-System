"""
Routes and views for the flask application.
"""


from datetime import datetime
from flask import *
import json
from Movie_Recommendation_System import app
from Movie_Recommendation_System.api import *




@app.route('/')
@app.route('/home')
def home():
    return render_template(
        'home.html',
        title = 'Home Page',
        year = datetime.now().year
    )




# Predict user rating of movie
#==================================================================
@app.route('/api/predictionRating/', methods=['GET'])
def predictionRatingAPI():
    userId = int(request.args.get('userId', 0))
    movieId = int(request.args.get('movieId', 0))
    rating = apiPredictionRating(userId, userId)

    jsonStr = json.dumps(rating)
    result = Response(jsonStr, status=200, mimetype='application/json')
    return result


@app.route('/predictionRating', methods=['GET', 'POST'])
def predictionRating():
    userId = int(request.form.get('userId', 0)) | int(request.args.get('userId', 0))
    movieId = int(request.form.get('movieId', 0)) | int(request.args.get('movieId', 0))
    rating = apiPredictionRating(userId, userId)

    return render_template(
        'predictionRating.html',
        title = 'Pediction Rating',
        year = datetime.now().year,
        rating = rating,
        userId = userId,
        movieId = movieId
    )




# Recommend movies by movie
#==================================================================
@app.route('/api/recommendationByMovie', methods=['GET'])
def recommendationByMovieAPI():
    movieId = int(request.args.get('id', 0))
    listRecommendation, movieTitle = apiRecommendationByMovie(movieId)
    list = [obj.toJSON() for obj in listRecommendation]

    jsonStr = json.dumps(list)
    result = Response(jsonStr, status=200, mimetype='application/json')
    return result


@app.route('/recommendationByMovie', methods=['GET', 'POST'])
def recommendationByMovie():
    movieId = int(request.form.get('id', 0)) | int(request.args.get('id', 0))
    listRecommendation, movieTitle = apiRecommendationByMovie(movieId)

    return render_template(
        'recommendationByMovie.html',
        title = 'Recommendations',
        year = datetime.now().year,
        list = listRecommendation,
        movieTitle = movieTitle
    )




# Recommend movies for user
#==================================================================
@app.route('/api/recommendationByUser/', methods=['GET'])
def recommendationByUserAPI():
    userId = int(request.args.get('id', 0))
    listRecommendation = apiRecommendationByUser(userId)
    list = [obj.toJSON() for obj in listRecommendation]

    jsonStr = json.dumps(list)
    result = Response(jsonStr, status=200, mimetype='application/json')
    return result


@app.route('/recommendationByUser', methods=['GET', 'POST'])
def recommendationByUser():
    userId = int(request.form.get('id', 0)) | int(request.args.get('id', 0))
    listRecommedation = apiRecommendationByUser(userId)

    return render_template(
        'recommendationByUser.html',
        title = 'Recommendations',
        year = datetime.now().year,
        list = listRecommedation,
        userId = userId
    )




# Retrieve top trending movies
#==================================================================
@app.route('/api/trending', methods=['GET'])
def trendingAPI():
    listTrending = apiTrending()
    list = [obj.toJSON() for obj in listTrending]

    jsonStr = json.dumps(list)
    result = Response(jsonStr, status=200, mimetype='application/json')
    return result


@app.route('/trending')
def trending():
    listTrending = apiTrending()

    return render_template(
        'trending.html',
        title = 'Top Ten',
        year = datetime.now().year,
        list = listTrending
    )