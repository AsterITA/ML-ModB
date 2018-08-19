import numpy as np
import sklearn


# FUNZIONI DI ATTIVAZIONE

def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # activation function


def sigmoid_(x):
    return sigmoid(x) * (1 - sigmoid(x))


def ReLU(x):
    return x * (x > 0)


def ReLU_(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def tanh(x):
    return np.tanh(x)


def tanh_(x):
    return 1 - tanh(x) ** 2


def linear(x):
    return x


def linear_(x):
    return 1


# FUNZIONI DI ERRORE

def sum_square(t, y):
    # err = 0
    # for i in range(y.size):
    #     err += (y[i] - t[i]) ** 2
    #     err /= 2
    err = (y - t) ** 2
    err = sum(err)
    err /= 2
    return err


def cross_entropy(t, y):
    return sklearn.metrics.log_loss(t, y)
