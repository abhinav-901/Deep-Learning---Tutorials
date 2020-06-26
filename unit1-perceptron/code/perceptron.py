import numpy as np
import os


class Perceptron():
    def __init__(self, X: np.ndarray, Y: np.ndarray, lr=0.01):
        self.X = X
        self.Y = Y
        self.lr = lr
        self.updated_weights = list()
        self.accuracy: int = 0

    def linear_node(self, X: np.ndarray, W: np.ndarray, b: float):
        linear_sum: np.ndarray = np.matmul(X, W) + b
        return linear_sum[0]

    def step_function(self, linear_sum: np.ndarray):
        if linear_sum >= 0:
            return 1
        return 0

    def back_propogation(self, W: np.ndarray, b: float):
        count: float = 0.0
        for i in range(len(self.X)):
            prediction: float = self.step_function(
                self.linear_node(self.X[i], W, b))
            if self.Y[i] - prediction == 1:
                W[0] += self.X[i][0] * self.lr
                W[1] += self.X[i][1] * self.lr
                b += self.lr
            elif self.Y[i] - prediction == -1:
                W[0] -= self.X[i][0] * self.lr
                W[1] -= self.X[i][1] * self.lr
                b -= self.lr
            elif self.Y[i] - prediction == 0:
                count += 1
        self.accuracy = 100 * count / len(self.X)
        return W, b

    def train(self, num_epochs=10):
        W = np.random.rand(2, 5)
        b = np.random.rand()
        for i in range(num_epochs):
            W, b = self.back_propogation(W, b)
            print(self.accuracy)
            self.updated_weights.append([W, b])


if __name__ == "__main__":
    print(np.random.rand())
    data_dir = os.path.dirname(__file__)
    file_name = 'data.txt'
    file_path = os.path.join(data_dir, file_name)
    data = np.loadtxt(file_path, delimiter=',')
    perceptron = Perceptron(data[:, :-1], data[:, -1])
    perceptron.train(num_epochs=25)
