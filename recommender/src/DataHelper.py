import pandas as pd

path = 'recommender/data/'

movies = pd.read_csv(path + 'movies.csv')
movies.set_index('movieId', inplace=True)


def get_movies_for_user(u_vector=[]):
    to_ret = []
    for id in u_vector:
        to_ret.append(movies.loc[id])

    return to_ret


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







'''
mat = pd.read_csv(path + 'chunks/chunk1.csv').pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
user = mat.loc[2]
user = user.sort_values(ascending=False)
indices = user.index[:10].tolist()
vector = [int(i) for i in indices]
get_movies_for_user(vector)
print('-------------------------')
new_mat = pd.read_csv('reconstructed.csv')
new_mat.set_index('userId', inplace=True)
new_user = new_mat.loc[2]
new_user = new_user.sort_values(ascending=False)
indices = new_user.index[:10].tolist()
vector = [int(i) for i in indices]

get_movies_for_user(vector)
'''



