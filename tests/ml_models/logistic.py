import numpy as np
import time as t

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iter=100, fit_intercept=False, verbose=False):
        self.learning_rate = learning_rate  # learning_rate of the algorithm
        self.num_iter = num_iter  #  number of iterations of the gradient descent
        self.fit_intercept = fit_intercept  # boolean indicating whether we`re adding base X0 feature vector or not
        self.verbose = verbose  

    def _add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))  #  creating X0 features vector(M x 1)
        return np.concatenate((intercept, X), axis=1)  # concatenating X0 features vector with our features making intercept

    def _sigmoid(self, z):
        '''Defines our "logit" function based on which we make predictions
           parameters:
              z - product of the our features with weights
           return:
              probability of the attachment to class
        '''

        return 1 / (1 + np.exp(-z))

    def _loss(self, h, y):
        '''
        Functions have parameters or weights and we want to find the best values for them.
        To start we pick random values and we need a way to measure how well the algorithm performs using those random weights.
        That measure is computed using the loss function
        '''

        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

    def train(self, X, y):
        '''
        Function for training the algorithm.
            parameters:
              X - input data matrix (all our features without target variable)
              y - target variable vector (1/0)
            
            return:
              None
        '''

        if self.fit_intercept:
            X = self._add_intercept(X)  # X will get a result with "zero" feature

        self._weights = np.zeros(X.shape[1])  #  inicializing our weights vector filled with zeros
        
        for i in range(self.num_iter):  # implementing Gradient Descent algorithm
            z = np.dot(X, self._weights)  #  calculate the product of the weights and predictor matrix
            h = self._sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self._weights -= self.learning_rate * gradient
            
            if (self.verbose == True and i % 10000 == 0):
                z = np.dot(X, self._weights)
                h = self._sigmoid(z)
                print(f'loss: {self._loss(h, y)} \t')

    def predict_prob(self, X):  
        if self.fit_intercept:
            X = self._add_intercept(X)
    
        return self._sigmoid(np.dot(X, self._weights))
    
    def predict(self, X):
        return self.predict_prob(X)

start = t.time()

X_train = np.ones((10000, 100))
y_train = np.ones(10000)
X_test = np.ones((1000, 100))

lr = 0.001
iters = 1000

logit = LogisticRegression(learning_rate=lr, num_iter=iters)
logit.train(X_train, y_train)
logit.predict(X_test)

end = t.time()

print(f'Execution time {end-start}s')

import torch

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs

start = t.time()

model = LogisticRegression(100, 1)

criterion = torch.nn.BCELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=lr)

X_train, X_test = torch.Tensor(X_train),torch.Tensor(X_test)
y_train = torch.Tensor(y_train)

losses = []
losses_test = []
Iterations = []
iter = 0
for epoch in range(iters):
    x = X_train
    labels = y_train
    optimizer.zero_grad() # Setting our stored gradients equal to zero
    outputs = model(X_train)
    loss = criterion(torch.squeeze(outputs), labels) # [200,1] -squeeze-> [200]
    
    loss.backward() # Computes the gradient of the given tensor w.r.t. graph leaves 
    
    optimizer.step() # Updates weights and biases with the optimizer (SGD)

end = t.time()

print(f'PyTorch Execution time {end-start}s')