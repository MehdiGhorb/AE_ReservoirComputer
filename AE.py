from keras.layers import Input, Dense
from keras.models import Model

class Autoencoder:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.autoencoder, self.encoder = self.build_autoencoder()

    def build_autoencoder(self):
        # Encoder
        input_img = Input(shape=(self.input_dim,))
        encoded = Dense(512, activation='relu')(input_img)
        encoded = Dense(256, activation='relu')(encoded)
        encoded = Dense(128, activation='relu')(encoded)

        # Decoder
        decoded = Dense(256, activation='relu')(encoded)
        decoded = Dense(512, activation='relu')(decoded)
        decoded = Dense(self.input_dim, activation='sigmoid')(decoded)

        # Autoencoder model
        autoencoder = Model(input_img, decoded)

        # Encoder model
        encoder = Model(input_img, encoded)

        # Compile autoencoder
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        
        return autoencoder, encoder
