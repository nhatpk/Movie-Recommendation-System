import json
from Movie_Recommendation_System.dbfunction import *
import requests


class objGenre(object):
    id = 0
    name = ""

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


class movieObj(object):
    genres = ""
    genreObjs = [objGenre]
    id = 0
    imdbId = ""
    imageUrl = ""
    overview = ""
    score = 0.0
    tags = ""
    title = ""
    vote_average = 0.0
    vote_count = 0

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


def mappingMovieObj(list):
    #df_links = loadLinks()

    result = []
    for item in list:
        temp = movieObj()

        #cols = ["genres", "id", "overview", "tagline", "title", "vote_count", "vote_average", 'imdb_id']

        temp.genres = convert(item[0])
        temp.genreObjs = item[0]
        temp.id = item[1]
        temp.imdbId = item[7] #str((df_links[df_links['movieId'] == temp.id])['imdbId'].values)
        temp.imageUrl = getPoster(item[4])
        temp.overview = item[2]
        temp.tags = item[3]
        temp.title = item[4]
        temp.vote_count = item[5]
        temp.vote_average = item[6]

        result.append(temp)

    return result


def mappingMovieObjForTrending(list):
    df_links = loadLinks()

    result = []
    for item in list:
        temp = movieObj()

        temp.genres = convert(item[0])
        temp.genreObjs = item[0]
        temp.id = item[1]
        temp.imdbId = str((df_links[df_links['movieId'] == temp.id])['imdbId'].values)
        temp.imageUrl = getPoster(item[5])
        temp.overview = item[2]
        temp.score = item[3]
        temp.tags = item[4]
        temp.title = item[5]
        temp.vote_count = item[6]
        temp.vote_average = item[7]

        result.append(temp)

    return result


# Convert list of obj to string 
def convert(jsonStr):
    list = json.loads(jsonStr.replace("\'", "\""))
    result = ""
    for s in list:
        result += s['name'] + ", "

    return result[:-2]


#
def getPoster(movieTitle):
    # Data Impulation: imdbId start with tt and has next 7-8 numbers: tt0114709
    #imdbId = imdbId[1:-1] # Remove [] at first and last
    #imdbId = imdbId.rjust(8, '0') # Right justified of length width with 0
    #imdbId = "tt" + imdbId
    
    #print('--------------------------------')
    #print(movieTitle)

    url = "http://www.omdbapi.com/?apikey=6e417307&" + "t=" + str(movieTitle)

    #{"Title":"Toy Story","Year":"1995","Rated":"G","Released":"22 Nov 1995","Runtime":"81 min","Genre":"Animation, Adventure, Comedy, Family, Fantasy","Director":"John Lasseter","Writer":"John Lasseter (original story by), Pete Docter (original story by), Andrew Stanton (original story by), Joe Ranft (original story by), Joss Whedon (screenplay by), Andrew Stanton (screenplay by), Joel Cohen (screenplay by), Alec Sokolow (screenplay by)","Actors":"Tom Hanks, Tim Allen, Don Rickles, Jim Varney","Plot":"A cowboy doll is profoundly threatened and jealous when a new spaceman figure supplants him as top toy in a boy's room.","Language":"English","Country":"USA","Awards":"Nominated for 3 Oscars. Another 23 wins & 17 nominations.","Poster":"https://m.media-amazon.com/images/M/MV5BMDU2ZWJlMjktMTRhMy00ZTA5LWEzNDgtYmNmZTEwZTViZWJkXkEyXkFqcGdeQXVyNDQ2OTk4MzI@._V1_SX300.jpg","Ratings":[{"Source":"Internet Movie Database","Value":"8.3/10"},{"Source":"Rotten Tomatoes","Value":"100%"},{"Source":"Metacritic","Value":"95/100"}],"Metascore":"95","imdbRating":"8.3","imdbVotes":"788,709","imdbID":"tt0114709","Type":"movie","DVD":"20 Mar 2001","BoxOffice":"N/A","Production":"Buena Vista","Website":"http://www.disney.com/ToyStory","Response":"True"}

    requestResult = requests.get(url)
    jsonObj = requestResult.json()
    url = jsonObj.get('Poster')

    #print(jsonObj)

    return url