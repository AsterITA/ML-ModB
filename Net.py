import copy as cp

import numpy as np
from sympy import sympify
from sympy import var, diff
from sympy.utilities.lambdify import lambdify
from tensorflow.examples.tutorials.mnist import input_data


class Net:
    def __init__(self, dimensions, activations):

        self.n_layers = len(dimensions)
        self.activations = activations
        self.primes = {}

        self.W = {}
        self.B = {}

        for i in range(len(dimensions) - 1):

            self.W[i + 1] = np.random.randn(dimensions[i], dimensions[i + 1]) / np.sqrt(dimensions[i])  # Pesi
            # The drawn weights are eventually divided by the square root of the current layers dimensions.
            # This is called Xavier initialization and helps prevent neuron activations from being too large or too
            # small. <--- decidiamo se lasciarlo o no
            self.B[i + 1] = np.zeros(dimensions[i + 1])  # Bias
            if self.activations[i + 1] == sigmoid:
                self.primes[i + 1] = sigmoid_
            else:
                self.primes[i + 1] = lambdify(x, diff(self.activations[i + 1](x), x))

                # Le righe successive sono una prova se le funzioni da input le prende bene
                # print(self.activations[i + 1](3))
                # print(self.primes[i + 1](3))

    # learning online
    def train_net_online(self, training_set, validation_set, max_epoch, eta, error_function, alpha):
        for t in range(max_epoch):
            # permutazione del training set
            np.random.shuffle(training_set)
            for n in range(len(training_set)):
                act_and_out = self.forward_propagation_with_output_hidden(training_set[n]['input'])
                deltas = self.back_propagation(training_set[n]['label'], act_and_out['outputs'],
                                               act_and_out['activations'])
                derivatives = self.calc_derivatives(training_set[n]['input'], act_and_out['outputs'], deltas)
                self.update_weights(derivatives, eta)
            # Calcolo dell'errore sul training set
            error_training = 0
            for n in range(len(training_set)):
                output = self.forward_propagation(training_set[n]['input'])
                error_training += error_function(output, training_set[n]['label'])
            # Calcolo dell'errore sul validation set
            error_validation = 0
            for n in range(len(validation_set)):
                output = self.forward_propagation(validation_set[n]['input'])
                error_validation += error_function(output, validation_set[n]['label'])
            # La rete di questo passo è migliore?
            try:
                if best_error_validation > error_validation:
                    best_error_validation = error_validation
                    best_W = cp.copy(self.W)
                    best_B = cp.copy(self.B)
            except NameError:
                best_error_validation = error_validation
                best_W = cp.copy(self.W)
                best_B = cp.copy(self.B)
            # Volendo si può applicare il criterio di fermata
            glt = 100 * (error_validation / best_error_validation - 1)
            if glt > alpha:
                break
        try:
            self.W = best_W
            self.B = best_B
        except:
            None

    def back_propagation(self, labels, outputs, node_act):
        # Calcolo delta
        deltas = np.ndarray(self.n_layers, dtype=np.ndarray)
        deltas[self.n_layers - 1] = self.primes[self.n_layers - 1](node_act[self.n_layers - 1]) * (
                outputs[self.n_layers - 1] - labels)
        for l in range(self.n_layers - 2, -1, -1):
            deltas[l] = np.dot(deltas[l + 1], self.W[l + 1])
            deltas[l] = self.primes[l](node_act[l]) * deltas[l]
        return deltas

    def calc_derivatives(self, input, outputs, deltas):
        # Calcolo derivate
        derivate_W = []
        derivate_B = []
        Z = input
        for l in range(self.n_layers):
            derivate_W.append(np.dot(deltas[l][:, np.newaxis], Z[np.newaxis, :]))
            derivate_B.append(deltas[l])
            Z = outputs[l]
        derivatives = {'weights': derivate_W, 'bias': derivate_B}
        return derivatives

    def update_weights(self, derivatives, eta):
        # Aggiornamento dei pesi
        for l in range(self.n_layers - 1):
            self.W[l] = self.W[l] - eta * derivatives['weights'][l]
            self.B[l] = self.B[l] - eta * derivatives['bias'][l]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # activation function


def sigmoid_(x):
    return sigmoid(x) * (1 - sigmoid(x))


def sum_square(t, y):
    err = 0
    for i in range(y.size):
        err += (y[i] - t[i]) ** 2
        err /= 2
    return err


# Test della rete neurale
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Data= np.concatenate((mnist.train.images, mnist.validation.images, mnist.test.images))
# Labels= np.concatenate((mnist.train.labels, mnist.validation.labels, mnist.test.labels))
functions = {}
functions[1] = sigmoid

x = var('x')  # the possible variable names must be known beforehand...
user_input = 'x **2'  # Simulo l'input dell'utente
expr = sympify(user_input)
f = lambdify(x, expr)  # Con questo si trasforma l'input in una funzione

functions[2] = f

# f_ = lambdify(x, diff(f(x), x))     # Con questo si ottiene la derivata della funzione in input

NN = Net([2, 1, 2], functions)
