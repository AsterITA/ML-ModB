import numpy as np
from sympy import sympify
from sympy import var, diff
from sympy.utilities.lambdify import lambdify


# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


class Net:
    def __init__(self, dimensions, activations):

        self.n_layers = len(dimensions)
        self.activations = {}
        self.primes = {}

        self.W = {}
        self.B = {}

        for i in range(len(dimensions) - 1):

            self.W[i + 1] = np.random.randn(dimensions[i], dimensions[i + 1]) / np.sqrt(dimensions[i])  # Pesi
            # The drawn weights are eventually divided by the square root of the current layers dimensions.
            # This is called Xavier initialization and helps prevent neuron activations from being too large or too
            # small. <--- decidiamo se lasciarlo o no
            self.B[i + 1] = np.zeros(dimensions[i + 1])  # Bias
            self.activations[i + 1] = activations[i]
            if activations[i] == sigmoid:
                self.primes[i + 1] = sigmoid_
            else:
                self.primes[i + 1] = lambdify(x, diff(activations[i](x), x))

                # Le righe successive sono una prova se le funzioni da input le prende bene
                print(self.activations[i + 1](3))
                print(self.primes[i + 1](3))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # activation function


def sigmoid_(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Test della rete neurale
np.random.seed(1)
functions = {}
functions[0] = sigmoid

x = var('x')  # the possible variable names must be known beforehand...
user_input = 'x **2'  # Simulo l'input dell'utente
expr = sympify(user_input)
f = lambdify(x, expr)  # Con questo si trasforma l'input in una funzione

functions[1] = f

# f_ = lambdify(x, diff(f(x), x))     # Con questo si ottiene la derivata della funzione in input

NN = Net([2, 1, 2], functions)
