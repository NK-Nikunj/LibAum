import numpy as np
import time as t

class Perceptron:
    def __init__(self, learning_rate, num_iter):
        self.learning_rate = learning_rate
        self.num_iter = num_iter

    def train(self, X, Y):
        self.w = np.zeros(len(X[0]))
        eta = self.learning_rate
        epochs = self.num_iter

        for t in range(epochs):
            for i in range(len(self.w)):
                val = np.dot(X[i], self.w)
                if (val*Y[i]) <= 0:
                    self.w[i] = self.w[i] + eta*val

    def predict(self, X_test):
        return np.dot(X_test, self.w)

start = t.time()

X_train = np.ones((10000, 100))
y_train = np.ones(10000)
X_test = np.ones((1000, 100))

lr = 0.001
iters = 1000

logit = Perceptron(learning_rate=lr, num_iter=iters)
logit.train(X_train, y_train)
logit.predict(X_test)


end = t.time()

print(f'Numpy Execution time {end-start}s')
