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
matrix = pd.read_csv(path + '/reconstructed.csv')
matrix = matrix.drop(['Unnamed: 0', 'userId'], axis=1)
print(matrix.shape)

np_matrix = matrix.to_numpy(dtype=np.float)
print(np_matrix.shape)

def predict(user):
    sim_array = cosine_similarity(np_matrix, user.reshape(1, -1))
    max_score, index, i = sim_array[3][0], 0, 0
    for sim in sim_array:
        if max_score < sim[0]:
            max_score = sim[0]
            index = i
        i += 1
    return index, max_score


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


def recommend(user, topn=10, as_tmdb=False):
    user = create_vector(user)
    print(user)
    array = np.asarray(replace(list(user.values()), to_replace=0))
    print(len(array))
    index, max_score = predict(array)
    print('index: ', index, ' with score: ', max_score)
    similar_user = matrix.loc[index].sort_values(ascending=False)
    to_recommend = [int(i) for i in similar_user.index[:topn].to_list()]
    if as_tmdb:
        return id_to_tmdb(to_recommend)
    else:
        return to_recommend


def create_vector(user):
    indices = dict().fromkeys([int(i) for i in matrix.columns.to_list()])
    print(len(indices))
    for id in user.keys():
        try:
            i = tmdb_to_id(id)
            indices.update({i : user[id]})
            indices[i] = user[i]
        except KeyError:
            continue
    print(len(indices))
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
