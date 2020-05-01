import pandas as pd
from recommender.src.Model import AutoEncoder
from recommender.src.predict import predict
path = 'recommender/data/'


def train(batch_size=32, lr=0.0001, epochs=50):
    matrix = pd.read_csv(path + '/matrices/matrix.csv')
    print(matrix.head())
    num_movies = len(matrix.iloc[0])

    model = AutoEncoder(batch_size=batch_size, lr=lr, epochs=epochs)
    model.build_model(num_movies)
    print(model)
    hist = model.train(matrix.values,matrix.values)
    AutoEncoder.plot(hist)
    new = model.reconstruct(matrix)
    print(new.head())
    pd.DataFrame(new).to_csv('recommender/models/model-v' + str(AutoEncoder.version - 1) + '/reconstructed.csv')




def normalize(v):
    v_max, vmin = max(v), min(v)
    lst = []
    for i, val in enumerate(v):
        lst.append(((val-vmin)/(v_max-vmin))*4.5 + 0.5)

    return lst
def get_indices(df):
    cols = df.columns
    dic = {}
    for c in cols:
        if(str(c).isnumeric()):
            dic[int(c)] = 0
    return dic

path = 'recommender/data/'

movies = pd.read_csv(path + 'movies.csv')
movies.set_index('movieId', inplace=True)


def get_movies_for_user(u_vector=[]):
    to_ret = []
    for id in u_vector:
        to_ret.append(movies.loc[id])

    return to_ret






def popular_movies():
    df = pd.read_csv(path + 'ratings.csv')
    df = df.sort_values('movieId', ascending=True)
    print(df)
    current = 1
    count = 0
    lst = pd.DataFrame({'movieId': [], 'count': []}, dtype=int)
    lst.append({'movieId': -1, 'rated': -1}, ignore_index=True)
    for index, row in df.iterrows():
        if row['movieId'] == current:
            count += 1
        else:
            lst = lst.append({'movieId': int(current), 'count': count}, ignore_index=True)
            print('movie ' + str(current) + ' count ' + str(count) + ' users')
            current = row['movieId']
            count = 1
    lst = lst.append({'movieId': int(current), 'count': count}, ignore_index=True)
    lst.to_csv(path + 'popular_movies.csv')
    return {'Status': 'Successfuly finished'}


def get_popular_movies(topn=20):
    df = pd.read_csv(path + 'popular_movies.csv')
    df = df.sort_values('count', ascending=False)
    lst = []
    for index, row in df.head(topn).iterrows():
        lst.append(row['movieId'])

    return id_to_tmdb(lst)

