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

print(f'Execution time {end-start}s')

import torch

class SVM(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SVM, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs

start = t.time()

model = SVM(100, 1)

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