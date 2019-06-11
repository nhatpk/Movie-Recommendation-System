import json


class objGenre(object):
    id = 0
    name = ""

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


class jsonPredictionObj(object):
    genres = [objGenre]
    id = 0
    movieId = 0
    title = ""

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


class jsonRecommendationObj(object):
    genres = [objGenre]
    id = 0
    overview = ""
    score = 0.0
    tags = ""
    title = ""
    vote_average = 0.0
    vote_count = 0

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


class jsonTrendingObj(object):
    id = 0
    score = 0.0
    title = ""
    vote_average = 0.0
    vote_count = 0

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)