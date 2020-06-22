import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

version = 1
with open('recommender/data/v.txt', 'r') as f:
    try:
        version = int(f.read())
    except:
        version = 1

path = 'recommender/models/model-v' + str(version - 1)
matrix = pd.read_csv('recommender/src/reconstructed.csv')
try:
    matrix = matrix.drop(['Unnamed: 0', 'userId'], axis=1)
except:

    #matrix = matrix.drop(['userId'], axis=1)
    print('a')
print(matrix.shape)

np_matrix = matrix.to_numpy(dtype=np.float)
print(np_matrix.shape)

def predict(user):
    if len(np_matrix[0]) != len(user):
        print(type(user))
        while len(np_matrix[0]) != len(user):
            user = np.delete(user, len(user) - 1)

    print("matrix: ", len(np_matrix[0]))
    print("user: ", len(user))
    sim_array = cosine_similarity(np_matrix, user.reshape(1, -1))
    j=0
    max_score, index, i = sim_array[3][j], 0, 0
    lst = {}
    for i in range(5):
        for sim in sim_array:
            if max_score < sim[0] and not i in list(lst.keys()):
                max_score = sim[0]
                index = i
            i += 1
        lst[index] = max_score
        index, max_score = 0, sim_array[3][0]

    for i in lst.keys():
        print("index: ", i, ' score: ', lst[i])

    import random
    r = random.randint(0, 4)
    tmp = list(lst.keys())
    index = tmp[r]

    return index, lst[index]


def normalize(v):
    v_max, vmin = max(v), min(v)
    lst = []
    for i, val in enumerate(v):
        lst.append(((val-vmin)/(v_max-vmin))*4.5 + 0.5)
    return lst


path = "recommender/data/"
df1 = pd.read_csv(path + 'links.csv')
df1.set_index('tmdbId', inplace=True)


def tmdb_to_id(id):
    try:
        return df1.loc[id]['movieId']
    except KeyError:
        return None


df2 = pd.read_csv(path + 'links.csv')
df2.set_index('movieId', inplace=True)


def id_to_tmdb(lst=[]):
    to_ret = []
    for id in lst:
        try:
            to_ret.append(int(df2.loc[id]['tmdbId']))
        except KeyError:
            continue
    return to_ret


def recommend(user, topn=20, as_tmdb=False):
    user = create_vector(user)
    array = np.asarray(replace(list(user.values()), to_replace=0))
    index, max_score = predict(array)
    print('index: ', index, ' with score: ', max_score)
    similar_user = matrix.loc[index].sort_values(ascending=False)
    to_recommend = [int(i) for i in similar_user.index[:topn].to_list()]
    movies = pd.read_csv('./recommender/data/movies.csv')
    movies.set_index('movieId', inplace=True)
    print("Recommended: ")
    for i in to_recommend:
        print(movies.loc[i]['title'])
    if as_tmdb:
        return id_to_tmdb(to_recommend)
    else:
        return to_recommend


def create_vector(user):
    movies = pd.read_csv('recommender/data/movies.csv')
    movies.set_index('movieId', inplace=True)
    print("user movies: ")
    indices = dict().fromkeys([int(i) for i in matrix.columns.to_list()])
    count = 0
    for id in user.keys():
        try:
            i = tmdb_to_id(id)

            indices.update({i : user[id]})
            print(indices[i])
            count += 1
        except KeyError:
            continue

    return indices


def replace(lst, to_replace=0):
    new = []
    for val in lst:
        if not (val is None):
            new.append(val)
        else:
            new.append(to_replace)
    print(list(filter(lambda x : x != 0, new)))
    return new
