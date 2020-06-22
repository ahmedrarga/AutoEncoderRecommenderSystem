import pandas as pd
from recommender.src.Model import AutoEncoder
from recommender.src.predict import *
path = 'recommender/data/'


def train(batch_size=32, lr=0.0001, epochs=50):
    matrix = pd.read_csv(path + 'train_data.csv').pivot(index='userId', columns='movieId', values='rating').fillna(0)
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


def recommendations(user):
    mail, flag = user[0], False
    vector = user[1]
    users = pd.read_csv(path + 'tv-base/users.csv')
    user_id = 0
    for index, row in users.iterrows():
        if row['email'] == mail:
            flag = True
            user_id = row['userId']
    if not flag:
        user_id = len(users['userId']) + 1
        users = users.append({'userId': user_id, 'email': mail}, ignore_index=True)
        users.to_csv(path + 'tv-base/users.csv', index=False)
        ratings = pd.read_csv(path + 'tv-base/ratings.csv')
        for key in vector.keys():
            ratings = ratings.append({'userId': user_id, 'movieId': key, 'rating': vector[key]}, ignore_index=True)
            ratings = ratings.sort_values(by=['userId'])
            ratings.to_csv(path + 'tv-base/ratings.csv', index=False)

    return recommend(vector, as_tmdb=True)


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


if __name__ == '__main__':
    inp = 0
    print("pick a number to choose:")
    print("1. train model")
    print("2. get predictions")
    print("3. get popular movies")
    print("4. exit")

    inp = int(input("choose number: "))
    if inp == 1:
        train()
    elif inp == 2:
        print("Enter the history: ", "such: user=1&hist={3:5, 100:4.5 ...}")
        user = input("Data: ")
        recommendations(user)
    elif inp == 3:
        get_popular_movies()
    elif inp == 4:
        exit(0)



