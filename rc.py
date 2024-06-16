import numpy as np
from scipy.special import softmax

class ESN:
    def __init__(self, input_size, reservoir_size, output_size, spectral_radius=0.9, alpha=0.99):
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.spectral_radius = spectral_radius
        self.alpha = alpha

        self.W_in = np.random.rand(reservoir_size, input_size + 1) - 0.5  # bias term
        self.W_res = np.random.rand(reservoir_size, reservoir_size) - 0.5 # bias term
        self.W_out = np.random.rand(output_size, reservoir_size) - 0.5    # bias term
        self.x0 = np.random.rand(reservoir_size)

        self.W_res *= spectral_radius / np.max(np.abs(np.linalg.eigvals(self.W_res)))

    def train(self, X_train, y_train, transient=100):
        X_train = np.concatenate((np.ones((len(X_train), 1)), X_train), axis=1)  # Add bias term to input
        X_res = np.zeros((len(X_train), self.reservoir_size))
        #x = np.zeros(self.reservoir_size)
        #x0 = np.random.rand(self.reservoir_size)

        for t in range(len(X_train)):
            u = X_train[t]
            #x = (1 - self.alpha) * x + self.alpha * np.tanh(np.dot(self.W_in, u) + np.dot(self.W_res, x))
            x = self.alpha * np.tanh(np.dot(self.W_in, u) + np.dot(self.W_res, self.x0))
            if t > transient:
                X_res[t] = x

        self.W_out = np.dot(np.linalg.pinv(X_res[transient:]), y_train[transient:])

    def predict(self, X_test):
        X_test = np.concatenate((np.ones((len(X_test), 1)), X_test), axis=1)  # Add bias term to input
        X_res = np.zeros((len(X_test), self.reservoir_size))
        #x = np.zeros(self.reservoir_size)
        #x0 = np.random.rand(self.reservoir_size)

        for t in range(len(X_test)):
            u = X_test[t]
            #x = (1 - self.alpha) * x + self.alpha * np.tanh(np.dot(self.W_in, u) + np.dot(self.W_res, x))
            x = self.alpha * np.tanh(np.dot(self.W_in, u) + np.dot(self.W_res, self.x0))
            X_res[t] = x

        return np.dot(X_res, self.W_out)
    
    def identity(self, x):
        return softmax(x)