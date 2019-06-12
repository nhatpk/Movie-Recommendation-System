# Movie-Recommendation-System
Machine Learning Model (SVD) to predict ratings / recommendation movies <br/>
Using Python + Flask

1. Enviroment requirements: Please see "Movie-Recommendation-System/Movie Recommendation System/requirements.txt"
** Hint: Can load this project in Visual Studio and build python environment

2. Docker setup: Please see "Movie-Recommendation-System/Dockerfile"

3. Html template is placed in "Movie-Recommendation-System/Movie Recommendation System/Movie_Recommendation_System/templates/" with base page is layout.html

4. js/css and data files are stored in "Movie-Recommendation-System/Movie Recommendation System/Movie_Recommendation_System/static/"

5. py files are in "Movie-Recommendation-System/Movie Recommendation System/Movie_Recommendation_System/" <br/>
   a. api.py: all main functions are here <br/>
   b. dbfunction.py: manage db-related progress like read/write <br/>
   c. objects.py: manage object problems like obj self-calling functions, mapping... <br/>
   d. views.py: mainly routing progress <br/>
   e. runserver.py: set host & port for your application. For Docker: if __name__ == '__main__': app.run(host='0.0.0.0', port='5000') <br/>
