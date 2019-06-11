import json


class objGenre(object):
    id = 0
    name = ""

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)


class movieObj(object):
    genres = ""
    genreObjs = [objGenre]
    id = 0
    imdbId = 0
    imageUrl = ""
    overview = ""
    score = 0.0
    tags = ""
    title = ""
    vote_average = 0.0
    vote_count = 0

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)