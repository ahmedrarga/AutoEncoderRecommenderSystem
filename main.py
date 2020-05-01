from flask import Flask, jsonify, request
from recommender.src.System import get_popular_movies, popular_movies, train
app = Flask(__name__)


@app.route('/')
def index():
    return 'Movie recommender system using deep auto-encoders with collaborative filtering approach'


@app.route('/api/popular_movies/<int:topn>')
def get_popular(topn=20):
    return jsonify({'popular_movies': get_popular_movies(topn)})


@app.route('/admin/create_popular_movies/<string:pwd>')
def create_popular(pwd='123456'):
    if pwd == '123456':
        return jsonify(popular_movies())
    else:
        return jsonify({'Error': 'Authentication failed'})


@app.route('/admin/train/<string:pwd>')
def train_model(pwd='123456'):
    if pwd == '123456':
        args = request.args.to_dict()
        try:
            return jsonify({"train": train(int(args['batch_size']), float(args['lr']), int(args['epochs']))})
        except (ValueError, KeyError) as e:
            return jsonify({'Error': str(e)})
    else:
        return jsonify({'Error': 'Authentication failed'})







if __name__ == '__main__':
    app.run()

