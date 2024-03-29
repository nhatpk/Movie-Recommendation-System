import pandas as pd


def loadKeywords():
    return pd.read_csv('Movie_Recommendation_System/static/data/keywords.csv')


def loadLinks():
    return pd.read_csv('Movie_Recommendation_System/static/data/links_small.csv')


def loadMoviesMetadata():
    return pd.read_csv('Movie_Recommendation_System/static/data/movies_metadata.csv')


def loadRatingsSmall():
    return pd.read_csv('Movie_Recommendation_System/static/data/ratings_small.csv')


def loadTmdb5000():
	df1 = pd.read_csv('Movie_Recommendation_System/static/data/tmdb_5000_credits.csv')
	df2 = pd.read_csv('Movie_Recommendation_System/static/data/tmdb_5000_movies.csv')

	df1.columns = ['id', 'tittle', 'cast', 'crew']
	df= df2.merge(df1, on = 'id')

	return df