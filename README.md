# AutoEncoderRecommenderSystem
Deep auto encoder for movie recommendation system

1. This is a flask rest api app, to run it you need to install the frameworks in requirements.txt, on ubuntu: 
  pip3 install -r requirements.txt
  
2. cd into project directory and run main.py:
  python3 main.py

3. to train the model you need to pass a http request:
  example: 127.0.0.1:5000/admin/train/<password>
  change the temporary password in the code from finalproject1212ahmad to the password you want
  the trainig done in the data in /data directory, if you need to train the model, you need to upload your own data to the directory and run: python3 preprocessing.py
  and then train the model.
  
4. to get recommendations for a specific user you need to pass user mail and user history:
    example: 127.0.0.1:5000/api/recommendations?user=ahmad@ahmad.com&hist={4:5, 242134:3.5, 3453:1, ....}
    the user parameter is to save specific user history to learn him more with the time and predict more relevant recommendations.

attention: the recommendation operation returns a json response with movie id's that might the user like, 
to see the complete results with the movie name and description you need to install the android app in my repositories:Tv-Guide, 
or to search this id's in tmdb site (www.themoviedb.com).
