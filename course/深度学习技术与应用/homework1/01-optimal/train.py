import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class MLP:
    def __init__(self, cost_fun, layers_size=[]):
        self.layers_size = layers_size
        self.parameters = {}
        self.L = len(self.layers_size)
        self.N = 0
        self.costs = []
        self.cost_function = cost_fun

    def fit(self, X, y, lr=0.1, n_epochs=1000):
        np.random.seed(123)
        X = np.asarray(X)
        y = np.asarray(y)
        print(X.shape)
        self.N = X.shape[0]  # 样本数

        # 根据样本数据插入输入层
        self.layers_size.insert(0, X.shape[1])

        self.initialzie_parameters()
        for epoch in range(n_epochs):
            # forward
            # A:the activation of final layer; AWZ: the activations,Weights and Zs
            A, AWZ = self.forward(X)

            # compute the cost or loss
            # cost = self.cost_function(A, y)
            # cost = np.squeeze(-(y.dot(np.log(A.T)) + (1 - y).dot(np.log(1 - A.T))) / self.N)
            cost = np.squeeze(
                -1.0 / self.N * (y @ np.log(A.T) + (1 - y) @ np.log(1 - A.T))
            )

            # backpropagarion
            partial_derivatives = self.backward(X, y, AWZ)

            # update the weights and bias
            for i in range(1, self.L + 1):
                self.parameters["W" + str(i)] -= lr * partial_derivatives["dW" + str(i)]
                self.parameters["B" + str(i)] -= lr * partial_derivatives["dB" + str(i)]
            if epoch % 100 == 99:
                print(f"[{epoch+1} / {n_epochs}]: {cost}")
                self.costs.append(cost)

    def initialzie_parameters(self):
        for i in range(1, self.L + 1):
            self.parameters["W" + str(i)] = np.random.randn(
                self.layers_size[i], self.layers_size[i - 1]
            )
            self.parameters["B" + str(i)] = np.zeros((self.layers_size[i], 1))

    def forward(self, X):
        A = X.T
        AWZ = {}
        for i in range(1, self.L):
            Z = self.parameters["W" + str(i)] @ A + self.parameters["B" + str(i)]
            A = self.sigmoid(Z)

            AWZ["A" + str(i)] = A
            AWZ["W" + str(i)] = self.parameters["W" + str(i)]
            AWZ["Z" + str(i)] = Z

        ZL = self.parameters["W" + str(self.L)] @ A + self.parameters["B" + str(self.L)]
        AL = self.sigmoid(ZL)
        AWZ["A" + str(self.L)] = AL
        AWZ["W" + str(self.L)] = self.parameters["W" + str(self.L)]
        AWZ["Z" + str(self.L)] = ZL
        return AL, AWZ

    def backward(self, X, y, AWZ):
        derivatives = {}
        AWZ["A0"] = X.T
        A = AWZ["A" + str(self.L)]
        dA = -np.divide(y, A) + np.divide(
            1 - y, 1 - A
        )  # the derivative of cross-entropy function
        # dA = (-1 * y @ A.T + (1 - y) @ (1 - A).T) / self.N
        dZL = dA * self.sigmoid_derivative(AWZ["Z" + str(self.L)])  # hadamard product
        dWL = dZL @ AWZ["A" + str(self.L - 1)].T / self.N * 1.0
        dBL = np.sum(dZL, axis=1, keepdims=True) / self.N * 1.0
        dZ = dZL
        derivatives["dW" + str(self.L)] = dWL
        derivatives["dB" + str(self.L)] = dBL

        for i in range(self.L - 1, 0, -1):
            dA_next = AWZ["W" + str(i + 1)].T @ dZ
            dZ = dA_next * self.sigmoid_derivative(AWZ["Z" + str(i)])
            dW = dZ @ AWZ["A" + str(i - 1)].T / self.N * 1.0
            dB = np.sum(dZ, axis=1, keepdims=True) / self.N * 1.0

            derivatives["dW" + str(i)] = dW
            derivatives["dB" + str(i)] = dB
        return derivatives

    def predict(self, X, y):
        A, _ = self.forward(X)
        n = X.shape[0]
        p = np.zeros((1, n))
        for i in range(A.shape[1]):
            if A[0, i] > 0.5:
                p[0, i] = 1
            else:
                p[0, i] = 0
        print(f"Accuracy: {np.sum(y == p) / n}")

    def sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def plot_cost(self):
        plt.figure()
        plt.plot(np.arange(len(self.costs)), self.costs)
        plt.xlabel("epochs")
        plt.ylabel("cost")
        plt.show()
