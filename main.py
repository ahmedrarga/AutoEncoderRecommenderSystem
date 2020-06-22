from flask import Flask, jsonify, request
from recommender.src.System import *
app = Flask(__name__)


@app.route('/')
def index():
    return 'Movie recommender system using deep auto-encoders with collaborative filtering approach'


@app.route('/api/popular_movies')
def get_popular():
    args = request.args.to_dict()
    try:
        topn = int(args['topn'])
    except (KeyError, ValueError) as e:
        return jsonify({'Error': str(e)})

    return jsonify({'popular_movies': get_popular_movies(topn)})


@app.route('/admin/create_popular_movies/<string:pwd>')
def create_popular(pwd='1'):
    if pwd == '123456':
        return jsonify(popular_movies())
    else:
        return jsonify({'Error': 'Authentication failed'})


@app.route('/admin/train/<string:pwd>')
def train_model(pwd='1'):
    if pwd == 'finalproject1212ahmad':
        args = request.args.to_dict()
        try:
            return jsonify({"train": train(int(args['batch_size']), float(args['lr']), int(args['epochs']))})
        except (ValueError, KeyError) as e:
            return jsonify({'Error': str(e)})
    else:
        return jsonify({'Error': 'Authentication failed'})


@app.route('/api/recommendations')
def get_recs():
    try:
        import ast
        args = request.args.to_dict()
        email, vector = args['user'], ast.literal_eval(args['hist'])
        if not (type(vector) is dict):
            return jsonify({'Error': 'Wrong user history representation'})
        else:
            return jsonify(recommendations((email, vector)))
    except (KeyError, ValueError, Exception) as e:
        return jsonify({'Error': str(e)})


if __name__ == '__main__':
    app.run()

