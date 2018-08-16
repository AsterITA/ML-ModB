import copy as cp
import sys

import numpy as np
from sympy import sympify
from sympy import var, diff
from sympy.utilities.lambdify import lambdify
from tensorflow.examples.tutorials.mnist import input_data

ETA_P = 1.2
ETA_M = 0.5
MAX_STEP = 50
MIN_STEP = 0


class Net:
    def __init__(self, dimensions, activations, error_function):

        self.n_layers = len(dimensions)
        self.activations = activations
        self.primes = {}
        self.error_function = error_function
        self.W = {}
        self.B = {}
        dimensions = dimensions.astype(int)

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
    def train_net(self, training_set, validation_set, max_epoch, eta, alpha, online_flag=False):
        best_error_validation = float("inf")  # float("inf") è il valore float dell'inifinito, quindi garantisce
        error_training = np.zeros(max_epoch)
        error_validation = np.zeros(max_epoch)
        for t in range(max_epoch):  # che è il numero più grande rappresentabile dal sistema
            derivatives_tot = {}
            lastDerivatives = {}
            if online_flag:
                # permutazione del training set
                np.random.shuffle(training_set)
            for n in range(len(training_set)):
                act, out = self.feed_forward(training_set['input'][n])
                derivatives = self.back_propagation(training_set['input'][n], training_set['label'][n], out, act)
                if online_flag:
                    self.update_weights(derivatives, eta)
                elif n == 0:
                    derivatives_tot['weights'] = derivatives['weights']
                    derivatives_tot['bias'] = derivatives['bias']
                    error_training[t] = self.error_function(out[self.n_layers], training_set['label'][n])
                else:
                    # last_error_training = error_training
                    derivatives_tot['weights'] += derivatives['weights']
                    derivatives_tot['bias'] += derivatives['bias']
                    error_training[t] += self.error_function(out[self.n_layers], training_set['label'][n])
                if n % 1000 == 0:  # Queste due righe le ho messe per capire
                    print("t", t, "n ", n)  # in fase di run a che punto sta
            if not online_flag:
                if not lastDerivatives:
                    self.update_weights(derivatives_tot, eta)
                else:
                    # Applico la RPROP
                    if t == 1:
                        updateValuesW = np.empty(self.W)
                        updateValuesB = np.empty(self.B)
                    for l in range(1, self.n_layers):
                        for n in range(len(self.W[l])):
                            for w in range(len(self.W[l][n])):
                                if t == 1:
                                    updateValuesW[l][n][w] = 0.0125  # DELTA0 per ogni connessione, di ogni neurone,
                                    # di ogni livello
                                # Aggiorno i pesi
                                updateValuesW[l][n][w] = self.RPROP(derivatives_tot['weights'][l][n],
                                                                    lastDerivatives['weights'][l][n],
                                                                    updateValuesW[l][n], w, error_training[t],
                                                                    error_training[t - 1])
                                self.W[l][n][w] += updateValuesW[l][n][w]
                            if t == 1:
                                updateValuesB[l][n] = 0.0125  # DELTA0 per ogni bias, di ogni neurone, di ogni livello
                            # Aggiorno i Bias
                            self.RPROP(derivatives_tot['bias'][l], lastDerivatives['bias'][l], updateValuesB[l], n,
                                       error_training[t], error_training[t - 1])
            if online_flag:
                # Calcolo dell'errore sul training set
                for n in range(len(training_set)):
                    _, out = self.feed_forward(training_set['input'][n])
                    error_training[t] += self.error_function(out[self.n_layers], training_set['label'][n])
                    if n % 1000 == 0:
                        print("t", t, "err train ", n)
            # Calcolo dell'errore sul validation
            for n in range(len(validation_set)):
                _, out = self.feed_forward(validation_set['input'][n])
                error_validation += self.error_function(out[self.n_layers], validation_set['label'][n])
                if n % 1000 == 0:
                    print("t ", t, "err_valid ", n)
            # La rete di questo passo è migliore?
            if best_error_validation > error_validation[t]:
                best_error_validation = error_validation[t]
                best_W = cp.copy(self.W)
                best_B = cp.copy(self.B)
            # Applico il criterio di fermata GL descritto in "Early Stopping, but when?"
            glt = 100 * (error_validation[t] / best_error_validation - 1)
            if glt > alpha:
                break
        self.W = best_W
        self.B = best_B
        return error_training, error_validation

    def back_propagation(self, input, labels, outputs, node_act):
        # Calcolo delta
        # Probabilmente il calcolo del delta va virtualizzato.
        # Il seguente è valido se si utilizza la somma dei quadrati e la cross entropy
        deltas = {self.n_layers: self.primes[self.n_layers - 1](node_act[self.n_layers]) * (
                outputs[self.n_layers] - labels)}  # out - target
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


    def RPROP(self, derivatives, lastDerivatives, lastDelta, i, actualError, lastError):
        change = np.sign(derivatives[i] * lastDerivatives[i])
        if change > 0:
            delta = min(lastDelta[i] * ETA_P, MAX_STEP)
            deltaW = -np.sign(derivatives[i]) * delta
            lastDelta[i] = delta
            lastDerivatives[i] = derivatives[i]
        elif change < 0:
            deltaW = 0
            lastWeightChange = lastDelta[i]
            delta = max(lastDelta[i] * ETA_M, MIN_STEP)
            if (actualError > lastError):
                deltaW = -lastDelta[i]
            lastDelta[i] = delta
            lastDerivatives[i] = derivatives[i] = 0
        else:
            deltaW = -np.sign(derivatives[i]) * lastDelta[i]
            lastDerivatives[i] = derivatives[i]
        return deltaW

    def update_weights(self, derivatives, eta):
        # Aggiornamento dei pesi Metodo discesa del gradiente
        for l in range(1, self.n_layers):
            self.W[l] = self.W[l] - eta * derivatives['weights'][l - 1].transpose()
            self.B[l] = self.B[l] - eta * derivatives['bias'][l - 1]


def PCA(data_set, soglia):
    # calcola il vettore media del dataset
    cov_mat = np.cov(data_set.T) \
        # Calcolo autovalori e autovettori
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    print('Eigenvectors ', eig_vecs.shape)
    print('Eigenvalues ', eig_vals.shape)
    # Calcolo la somma totale degli autovalori
    eig_vals_tot = sum(eig_vals)
    # Make a list of (eigenvalue, eigenvector) tuples
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
    # Sort the (eigenvalue, eigenvector) tuples from high to low
    eig_pairs.sort(reverse=True, key=(lambda x: x[0]))
    # prendo le componenti, le quali insieme soddisfano la soglia
    counter = 0.0
    new_dim = 0
    for i in eig_pairs:
        counter += i[0]
        new_dim += 1
        if (counter / eig_vals_tot) >= soglia:
            break
    # creazione matrice di proiezione
    matrix_w = np.hstack(eig_pairs[i][1].reshape(len(data_set[0]), 1) for i in range(new_dim))
    return np.dot(data_set, matrix_w), matrix_w


def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # activation function


def sigmoid_(x):
    return sigmoid(x) * (1 - sigmoid(x))


def sum_square(t, y):
    # err = 0
    # for i in range(y.size):
    #     err += (y[i] - t[i]) ** 2
    #     err /= 2
    err = (y - t) ** 2
    err = sum(err)
    err /= 2
    return err


def getUserAmount(min, max):
    while True:
        amount = input("Inserisci un numero compreso tra {} e {}\n".format(min, max))
        try:
            value = int(amount)
            if min <= value <= max:
                break
            else:
                print("Il numero dev'essere compreso tra {} e {}, riprova\n".format(min, max))
        except ValueError:
            print("Devi inserire un numero, riprova\n")
    return value


def getUserFunction(n_variables):
    if n_variables == 1:
        var('x')
        user_input = input("Definisci una funzione matematica con una variabile x\n")
        func = lambdify(x, sympify(user_input))
    elif n_variables == 2:
        var('x y')
        user_input = input("Definisci una funzione matematica con due variabili x e y\n")
        func = lambdify((x, y), sympify(user_input))
    return func


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#   Mi prendo le prime 200 immagini per il training e le prime 100 per il validation e per il test
train_images = mnist.train.images[:200]
train_labels = mnist.train.labels[:200]
validation_images = mnist.validation.images[:100]
validation_labels = mnist.validation.labels[:100]
test_images = mnist.test.images[:100]
test_labels = mnist.test.labels[:100]

training = {'input': train_images, 'label': train_labels}
validation = {'input': validation_images, 'label': validation_labels}
test = {'input': test_images, 'label': test_labels}

print("Scegli cosa vuoi fare dalla seguente lista:")
while True:
    print("0) Chiudi il programma\n"
          "1) Effettua un training di una rete neurale in maniera manuale\n"
          "2) Confronto tra training con PCA e training con rete autoassociativa\n")
    choice = getUserAmount(0, 2)
    if choice == 0:
        sys.exit()
    elif choice == 1:
        print("Quanti strati interni vuoi all'interno della tua rete?")
        n_layers = getUserAmount(1, 10)
        dimensions = np.zeros(n_layers + 2)
        dimensions[0] = len(train_images[0])
        dimensions[n_layers + 1] = 10
        functions = {}
        for l in range(1, n_layers + 1):
            print("Inserisci il numero di nodi al livello {}".format(l))
            dimensions[l] = getUserAmount(1, 900)
            print("Che funzione di attivazione vuoi utilizzare in questo strato?\n"
                  "1) Sigmoide\n"
                  "2) Definita da input\n")
            choice = getUserAmount(1, 2)
            if choice == 1:
                functions[l] = sigmoid
            else:
                functions[l] = getUserFunction(1)
        print("Che funzione di attivazione vuoi utilizzare nello strato di output?\n"
              "1) Sigmoide\n"
              "2) Definita da input\n")
        choice = getUserAmount(1, 2)
        if choice == 1:
            functions[n_layers + 1] = sigmoid
        else:
            functions[n_layers + 1] = getUserFunction(1)
        print("Che funzione di errore vuoi utilizzare?\n"
              "1) Somma dei quadrati\n"
              "2) Definita da input\n")
        choice = getUserAmount(1, 2)
        if choice == 1:
            error_function = sum_square
        else:
            error_function = getUserFunction(2)

        NN = Net(dimensions, functions, error_function)
        print("Vuoi utilizzare il learning batch o online?\n"
              "1) Batch\n"
              "2) Online\n")
        method = getUserAmount(1, 2)
        if method == 1:
            method = False
        else:
            method = True
        NN.train_net(test, validation, 50, 0.5, 10, method)

        print("\n" * 10)
        continue
    elif choice == 2:
        # TEST PCA
        soglia_pca = 0.7
        new_dataset, matrix_w = PCA(training['input'], soglia_pca)
        train_PCA = {'input': new_dataset, 'label': train_labels}
        dimensions = np.zeros(3)
        dimensions[0] = len(new_dataset[0])
        dimensions[2] = 10
        functions = {}
        print("inserisci il numero di nodi nello strato nascosto")
        dimensions[1] = getUserAmount(1, 900)
        print("Che funzione di attivazione vuoi utilizzare in questo strato?\n"
              "1) Sigmoide\n"
              "2) Definita da input\n")
        choice = getUserAmount(1, 2)
        if choice == 1:
            functions[1] = sigmoid
        else:
            functions[1] = getUserFunction(1)
        print("Che funzione di attivazione vuoi utilizzare nello strato di output?\n"
              "1) Sigmoide\n"
              "2) Definita da input\n")
        choice = getUserAmount(1, 2)
        if choice == 1:
            functions[2] = sigmoid
        else:
            functions[2] = getUserFunction(1)
        print("Che funzione di errore vuoi utilizzare?\n"
              "1) Somma dei quadrati\n"
              "2) Definita da input\n")
        choice = getUserAmount(1, 2)
        if choice == 1:
            error_function = sum_square
        else:
            error_function = getUserFunction(2)
        NN_PCA = Net(dimensions, functions, error_function)
        NN_PCA.train_net(train_PCA, validation, 50, 0.5, 10)

        # Test Rete Autoassociativa

        print("\n" * 10)
        continue

"""
# Codice per stampare a video un immagine del mnist in bianco e nero
import matplotlib.image as mpimg
import sklearn as skl

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Data= np.concatenate((mnist.train.images, mnist.validation.images, mnist.test.images))
# Labels= np.concatenate((mnist.train.labels, mnist.validation.labels, mnist.test.labels))
print(type(mnist.train.images), len(mnist.train.images))
a = mnist.train.images[1]
print(type(a), len(a))
plt.imshow(np.ndarray.reshape(a, (28, 28)), cmap=plt.cm.binary)
plt.show()
print(mnist.train.labels[1])

# con quello di sopra faccio la riduzione della dimensionalità PCA built-in
from sklearn import decomposition

soglia_pca = 0.7
pca = decomposition.PCA(soglia_pca)
pca.fit(mnist.train.images)
Data = pca.transform(mnist.train.images)
print("nuova dimensione = ", len(Data[1]))
data2 = pca.inverse_transform(Data)
print(len(data2[1]))
plt.imshow(np.ndarray.reshape(data2[1], (28, 28)), cmap=plt.cm.binary)
plt.show()

# adesso testo la PCA che ho sviluppato
soglia_pca = 0.7
new_dataset, matrix_w = PrincipalComponentAnalisys(mnist.train.images, soglia_pca)

print(new_dataset.shape)
data2 = np.dot(new_dataset, matrix_w.transpose())
print(data2.shape)
plt.imshow(np.ndarray.reshape(data2[1], (28, 28)), cmap=plt.cm.binary)
plt.show()

functions = {}
#
x = var('x')  # the possible variable names must be known beforehand...
user_input = 'x'  # Simulo l'input dell'utente
expr = sympify(user_input)
f = lambdify(x, expr)  # Con questo si trasforma l'input in una funzione
# functions[1] = functions[2] = f
functions[1] = functions[2] = sigmoid
# genero un training set
training_set = []
for i in range(len(mnist.train.images)):
    elem = {'input': mnist.train.images[i], 'label': mnist.train.images[i]}
    training_set.append(elem)
# genero un validation set
validation_set = []
for i in range(len(mnist.validation.images)):
    elem = {'input': mnist.validation.images[i], 'label': mnist.validation.images[i]}
    validation_set.append(elem)

print("prova rete autoassociativa")
print(len(mnist.train.images[0]), new_dataset.shape[1], len(mnist.train.images[0]))
NN = Net([len(mnist.train.images[0]), new_dataset.shape[1], len(mnist.train.images[0])], functions)
NN.train_net(training_set, validation_set, 20, 0.5, sum_square, float("inf"), True)  # criterio di fermata annullato

# test con lo stesso numero di sopra
_, risposta = NN.feed_forward(mnist.train.images[1])
plt.imshow(np.ndarray.reshape(risposta[NN.n_layers], (28, 28)), cmap=plt.cm.binary)
plt.show()
# adesso provo con un 'immagine del test set
plt.imshow(np.ndarray.reshape(mnist.test.images[1000], (28, 28)), cmap=plt.cm.binary)
plt.show()
_, risposta = NN.feed_forward(mnist.test.images[1000])
plt.imshow(np.ndarray.reshape(risposta[NN.n_layers], (28, 28)), cmap=plt.cm.binary)
plt.show()
"""
