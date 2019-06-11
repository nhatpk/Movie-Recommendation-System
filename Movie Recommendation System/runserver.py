"""
This script runs the Movie_Recommendation_System application using a development server.
"""

from os import environ
from Movie_Recommendation_System import app

if __name__ == '__main__':
    HOST = environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(environ.get('SERVER_PORT', '5555'))
    except ValueError:
        PORT = 5555
    app.run(HOST, PORT)
