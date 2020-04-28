from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import Adam
from keras.models import model_from_json


import os
import pandas as pd
import matplotlib.pyplot as plt


class AutoEncoder:
    version = 1
    path = 'recommender/data/'

    def __init__(self, batch_size, lr, epochs):
        with open(AutoEncoder.path + 'v.txt', 'r') as f:
            AutoEncoder.version = int(f.read())
        self.ENC_LAYER1 = 512
        self.ENC_LAYER2 = 256
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.model = None
        self.trained = False

    def build_model(self, input_shape):
        """

        Build Auto encoder for collaborative filtering
        :return: Auto encoder model

        """

        # Input
        input_layer = Input(shape=(input_shape,), name='UserScore')

        # Encoder
        encoder_layer = Dense(self.ENC_LAYER1, activation='selu', name='EncoderLayer')(input_layer)

        # Latent space
        latent_space = Dense(self.ENC_LAYER2, activation='selu', name='LatentSpace')(encoder_layer)
        latent_space = Dropout(0.8, name="Dropout")(latent_space)
        # latent_space = Dropout(0.8, name="Dropout2")(latent_space)
        # Decoder
        decoder_layer = Dense(self.ENC_LAYER1, activation='selu', name='DecoderLayer')(latent_space)

        # Output
        output_layer = Dense(input_shape, activation='linear', name='UserPrediction')(decoder_layer)
        self.model = Model(input_layer, output_layer)
        return self.model

    def train(self, X, y):
        """
        train the model
        :param X: data input
        :param y: data input
        :return: None
        """
        if self.model is None:
            self.model = self.build_model(X.shape[1])
        print('Begin training ...')
        self.model.compile(optimizer=Adam(lr=self.lr), loss='binary_crossentropy')
        hist = self.model.fit(x=X, y=y,
                              epochs=self.epochs,
                              batch_size=self.batch_size,
                              shuffle=True,
                              validation_split=0.1)
        print('Saving model ...')
        path = 'recommender/models/' + 'model-v' + str(AutoEncoder.version)
        os.mkdir(path)
        self.model.save_weights(path + '/model_weights.h5')
        with open(path + '/model_architecture.json', 'w') as f:
            f.write(self.model.to_json())
        print('Saved to' + path)
        AutoEncoder.version += 1
        with open(AutoEncoder.path + 'v.txt', 'w') as f:
            f.write(str(AutoEncoder.version))
        self.trained = True
        return hist

    def reconstruct(self, X):
        if self.model is None or not self.trained:
            raise PermissionError('Cannot reconstruct Non-trained model, use train(X, y)')
        preds = self.model.predict(X) * (X == 0)
        return preds

    def __str__(self):
        if self.model is None:
            return 'empty Model()'
        else:
            return str(self.model.summary())

    @staticmethod
    def import_model(path):
        """
        import keras auto encoder model
        :param path: path to model directory
        :return: keras model
        """
        # Model reconstruction from JSON file
        f = open(path + '/model_architecture.json', 'r')
        model = model_from_json(f.read())
        f.close()
        # Load weights into the new model
        model.load_weights(path + '/model_weights.h5')
        return model

    @staticmethod
    def plot(history):
        # Plot training & validation loss values
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

'''
df = pd.read_csv('../data/chunks/chunk1.csv')
matrix = df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)
print(matrix.head())
num_movies = len(matrix.iloc[0])
num_users = len(df['userId'])


model = AutoEncoder(32,0.01,20)
model.build_model(num_movies)
model.train(matrix.values,matrix.values)
new = model.reconstruct(matrix)
print(new)
pd.DataFrame(new).to_csv('reconstructed.csv')
'''











