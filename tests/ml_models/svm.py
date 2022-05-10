import numpy as np
import time as t

class svm:
    def __init__(self, learning_rate, num_iter):
        self.learning_rate = learning_rate
        self.num_iter = num_iter

    def train(self, X, Y):
        self.w = np.zeros(len(X[0]))
        epochs = self.num_iter

        for e in range(epochs):
            for i in range(len(self.w)):
                val1 = np.dot(X[i], self.w)
                if (Y[i]*val1 < 1):
                    self.w[i] = self.w[i] + self.learning_rate * (val1 - (2*(1/epochs)*self.w[i]))
                else:
                    self.w[i] = self.w + self.learning_rate * (-2*(1/epochs)*self.w[i])

    def predict(self, X_test):
        return np.dot(X_test, self.w)

start = t.time()

X_train = np.ones((10000, 100))
y_train = np.ones(10000)
X_test = np.ones((1000, 100))

lr = 0.001
iters = 1000

logit = svm(learning_rate=lr, num_iter=iters)
logit.train(X_train, y_train)
logit.predict(X_test)


end = t.time()

print(f'NumPy Execution time {end-start}s')
