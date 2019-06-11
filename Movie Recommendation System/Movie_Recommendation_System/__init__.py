"""
The flask application package.
"""

from flask import Flask
app = Flask(__name__)

import Movie_Recommendation_System.views
