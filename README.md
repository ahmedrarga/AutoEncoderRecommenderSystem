# AutoEncoderRecommenderSystem
Deep auto encoder for movie recommendation system

1. This is a flask api app, to run it you need to install the frameworks in requirements, on ubuntu.txt: 
  pip3 install -r requirements.txt
  
2. cd into project directory and run main.py:
  python3 main.py

3. to train the model you need to pass a http request:
  example: 127.0.0.1:5000/admin/<password>
  
4. to get recommendations for a specific user you need to pass user mail and user history:
    example: 127.0.0.1:5000/api/recommendations?user=ahmad@ahmad.com&hist={4:5, 242134:3.5, 3453:1, ....}

attention: the recommendation operation returns a json response with movie id's that might the user like, 
to see the complete results with the movie name and description you need to install the android app in my repositories:Tv-Guide, 
or to search this id's in tmdb site (www.themoviedb.com).
