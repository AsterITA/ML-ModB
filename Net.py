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
            self.B[i + 1] = np.zeros(dimensions[i + 1])  # Bias
            if self.activations[i + 1] == sigmoid:
                self.primes[i + 1] = sigmoid_
            else:
                self.primes[i + 1] = lambdify(x, diff(self.activations[i + 1](x), x))

    # Forward propagation
    def feed_forward(self, x):
        z = {}
        # The first layer has no ‘real’ activations, so we consider the inputs x as the activations of the previous layer.
        a = {1: x}

        for i in range(1, self.n_layers):
            z[i + 1] = np.dot(a[i], self.W[i]) + self.B[i]
            a[i + 1] = self.activations[i](z[i + 1])

        # Output
        return z, a

    # learning online
    def train_net(self, training_set, validation_set, max_epoch, eta, error_function, alpha, online_flag):
        best_error_validation = float("inf")  # float("inf") è il valore float dell'inifinito, quindi garantisce
        for t in range(max_epoch):  # che è il numero più grande rappresentabile dal sistema
            derivatives_tot = {}
            if online_flag:
                # permutazione del training set
                np.random.shuffle(training_set)
            for n in range(len(training_set)):
                act, out = self.feed_forward(training_set[n]['input'])
                derivatives = self.back_propagation(training_set[n]['input'], training_set[n]['label'], out,
                                                    act)
                if online_flag:
                    self.update_weights(derivatives, eta)
                elif t == 0:
                    derivatives_tot['weights'] = derivatives['weights']
                    derivatives_tot['bias'] = derivatives['bias']
                else:
                    derivatives_tot['weights'] += derivatives['weights']
                    derivatives_tot['bias'] += derivatives['bias']
            if not online_flag:
                self.update_weights(derivatives_tot, eta)
            # Calcolo dell'errore sul training set
            error_training = 0
            for n in range(len(training_set)):
                _, out = self.feed_forward(training_set[n]['input'])
                error_training += error_function(out[self.n_layers], training_set[n]['label'])
            # Calcolo dell'errore sul validation set
            error_validation = 0
            for n in range(len(validation_set)):
                _, out = self.feed_forward(validation_set[n]['input'])
                error_validation += error_function(out[self.n_layers], validation_set[n]['label'])
            # La rete di questo passo è migliore?
            if best_error_validation > error_validation:
                best_error_validation = error_validation
                best_W = cp.copy(self.W)
                best_B = cp.copy(self.B)
            # Applico il criterio di fermata GL descritto in "Early Stopping, but when?"
            glt = 100 * (error_validation / best_error_validation - 1)
            if glt > alpha:
                break
        self.W = best_W
        self.B = best_B

    def back_propagation(self, input, labels, outputs, node_act):
        # Calcolo delta
        # Probabilmente il calcolo del delta va virtualizzato.
        # Il seguente è valido se si utilizza la somma dei quadrati e la cross entropy
        deltas = {self.n_layers: self.primes[self.n_layers - 1](node_act[self.n_layers]) * (
                outputs[self.n_layers] - labels)}
        for l in range(self.n_layers - 1, 1, -1):
            deltas[l] = np.dot(deltas[l + 1], self.W[l].transpose())
            deltas[l] = self.primes[l - 1](node_act[l]) * deltas[l]
        # Calcolo derivate
        derivate_W = []
        derivate_B = []
        z = input
        for l in range(2, self.n_layers + 1):
            derivate_W.append(np.dot(deltas[l][:, np.newaxis], z[np.newaxis, :]))
            derivate_B.append(deltas[l])
            z = outputs[l]
        derivatives = {'weights': derivate_W, 'bias': derivate_B}
        return derivatives

    def update_weights(self, derivatives, eta):
        # Aggiornamento dei pesi Metodo discesa del gradiente
        for l in range(1, self.n_layers):
            self.W[l] = self.W[l] - eta * derivatives['weights'][l - 1].transpose()
            self.B[l] = self.B[l] - eta * derivatives['bias'][l - 1]


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

NN = Net([2, 3, 1], functions)

#  IL SEGUENTE E' IL CODICE CHE HO USATO PER TESTARE IL LEARNING ONLINE E IL BATCH
# genero un training set
training_set = []
for i in range(10):
    inpu = np.random.randint(0, 4, 4)
    app = np.random.randint(0, 2, 1)
    out = np.zeros(3)
    out[app] = 1
    elem = {'input': inpu, 'label': out}
    training_set.append(elem)
# print(training_set)
functions[1] = sigmoid
functions[2] = sigmoid
mia_net = Net([4, 4, 3], functions)
print("W= ", mia_net.W)
print("B= ", mia_net.B)
error_function = sum_square
mia_net.train_net(training_set, [], 50, 0.5, error_function, 10, False)  # False = BATCH; True = ONLINE
print("W= ", mia_net.W)
print("B= ", mia_net.B)
