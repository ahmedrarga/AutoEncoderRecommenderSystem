from recommender.src.Model import AutoEncoder
from recommender.src.DataHelper import get_movies_for_user, get_popular_movies, popular_movies
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

path = 'recommender/data/'


def train(batch_size=32, lr=0.001, epochs=50):
    matrix = pd.read_csv(path + 'train_data.csv')
    matrix = matrix.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
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

'''
index=500
new_mat = pd.read_csv('reconstructed.csv')
new_mat = new_mat.drop(['Unnamed: 0'],axis=1)
array = new_mat.loc[index].to_numpy()
print(array)
tmp_user = get_indices(new_mat)
lst = []

m = new_mat
new_mat = new_mat.to_numpy()
l = []

for a in new_mat:
    l.append(cosine_similarity(array.reshape(1,-1), a.reshape(1,-1)).reshape(1)[0])
sim = l[0]
for s in l:
    if s > sim and l.index(s) != index:
        sim = s
print(sim)
print(l.index(sim))

for key in list(tmp_user.keys()):
    if(tmp_user[key] != 0):
        lst.append(key)
user = m.loc[index]
user = user.sort_values(ascending=False)
indices = user.index[:10].tolist()
vector = [int(i) for i in indices]
for i in get_movies_for_user(vector):
    print(i['title'], '---', i['genres'])
print('----------------------------------------')
user = m.loc[l.index(sim)]
user = user.sort_values(ascending=False)
indices = user.index[:10].tolist()
vector = [int(i) for i in indices]
k=0
v = normalize(user.tolist()[:10])
for i in get_movies_for_user(vector):
    print(i['title'],'---',i['genres'],'score: ', v[k])
    k+=1
'''