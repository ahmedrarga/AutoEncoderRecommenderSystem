import pandas as pd
import numpy as np

path = '../data/'

class Data:
    user_id = 'userId'
    movie_id = 'movieId'
    rating = 'rating'

    def __init__(self, path, create=False):
        self.path = path
        self.ratings = None
        self.interactions = None
        self.chunk_size = 24
        if create:
            self.save_new_movies(1970)
            self.load_ratings('new_ratings.csv')
            self.count_ratings()
            self.create_merged_interactions()
            self.clean_data()
            self.create_matrices(chunks=False)



    def load_ratings(self, name):
        print('Reading data ... ')
        self.ratings = pd.read_csv(path + name,
                                   usecols=[Data.user_id,
                                            Data.movie_id,
                                            Data.rating])
        print('Summary: ')
        print('Number of users: ', len(set(self.ratings[Data.user_id])))
        print('Number of movies: ', len(set(self.ratings[Data.movie_id])))
        print('-------------------------------')


    def count_ratings(self):
        current = 1
        count = 0
        lst = pd.DataFrame({Data.user_id : [], 'rated' : []}, dtype=int)
        lst.append({Data.user_id: -1, 'rated' : -1}, ignore_index=True)
        for index, row in self.ratings.iterrows():
            if row[Data.user_id] == current:
                count += 1
            else:
                lst = lst.append({Data.user_id: int(current), 'rated': count}, ignore_index=True)
                print('user ' + str(current) + ' rated ' + str(count) + ' movies')
                current += 1
                count = 0
        lst = lst.append({Data.user_id: int(current), 'rated': count}, ignore_index=True)
        print('user ' + str(current) + ' rated ' + str(count) + ' movies')
        lst.to_csv(path + 'interactions.csv')

    def get_interactions(self):
        print('Reading interactions ... ')
        df = pd.read_csv(self.path + 'interactions.csv')
        self.interactions = df

    def create_merged_interactions(self):
        if self.interactions is None:
            self.get_interactions()
        if self.ratings is None:
            self.load_ratings()
        df = pd.merge(self.ratings, self.interactions, on=[Data.user_id], how='inner')
        df.to_csv(self.path + 'merged_ratings.csv')

    def clean_data(self):
        interactions = pd.read_csv(self.path + 'merged_ratings.csv',
                                   usecols=[Data.user_id, Data.movie_id, Data.rating, 'rated'])
        lst = {}
        i = 0
        num_users = len(set(interactions['userId']))
        for index, row in interactions.iterrows():
            if row['rated'] >= 20:
                lst[i] = {Data.user_id:row[Data.user_id],
                            Data.movie_id:row[Data.movie_id],
                            Data.rating:row[Data.rating]}
                i+=1
            else:
                print('user ', int(row['userId']), ' removed')

        df = pd.DataFrame().from_dict(lst, "index")
        df = df.astype({Data.user_id:'int64',
                            Data.movie_id: 'int64',
                            Data.rating:'float'})
        print(df.shape)
        df.to_csv(path + 'train_data.csv')

    def save_new_movies(self, year):
        df = pd.read_csv(self.path + 'movies.csv')
        count = 0
        lst = {}
        for index, row in df.iterrows():
            title = row['title'].split(' ')
            try:
                y = int(title[len(title)-1].replace('(','').replace(')',''))
                if y >= year:
                    lst[count] = {'movieId':row['movieId']}
                    count += 1
                    print(count)
            except:
                continue
        new_movies = pd.DataFrame().from_dict(lst, 'index')
        new_movies = new_movies.astype({'movieId':'int64'})
        new_movies.to_csv(self.path + 'new_movies.csv')
        df_ratings = pd.read_csv(self.path + 'ratings.csv')
        df = pd.merge(df_ratings, new_movies, on='movieId', how='inner')
        #df = df.drop(['timestamp', 'Unnamed: 0'], axis=1)
        df = df.sort_values(by='userId', ascending=True)
        df.to_csv(path + 'new_ratings.csv')
        print(df.shape)

    def divide_dataset(self, chunk_size):
        self.chunk_size = chunk_size
        df = pd.read_csv(self.path + 'train_data.csv')
        num_chunks = len(df) / chunk_size
        if num_chunks - int(num_chunks) > 0:
            num_chunks = int(num_chunks) + 1
        chunks = np.array_split(df, num_chunks)
        path, i = self.path + 'chunks/', 1
        for chunk in chunks:
            chunk.to_csv(path + 'chunk' + str(i) + '.csv')
            print('chunk ' + str(i) + ' saved,', num_chunks - i, 'chunks remaining')
            i += 1

        print('0 chunks remaining')

    def create_matrices(self, chunks=True):
        m_dir = self.path + 'matrices/'
        c_dir = self.path + 'chunks/'
        if chunks:
            for chunk in range(4,self.chunk_size + 1):
                tmp = c_dir + 'chunk' + str(chunk) + '.csv'
                df = pd.read_csv(tmp)\
                    .pivot_table(index=Data.user_id, columns=Data.movie_id, values=Data.rating)\
                    .fillna(0)\
                .to_csv(m_dir + 'matrix' + str(chunk) + '.csv')
                print('matrix ', chunk, 'saved')
        else:
            self.ratings\
                    .pivot_table(index=Data.user_id, columns=Data.movie_id, values=Data.rating)\
                    .fillna(0)\
                    .to_csv(m_dir + 'matrix.csv')









data = Data(path, create=True)

