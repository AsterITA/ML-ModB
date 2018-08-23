import copy as cp
import sys

import numpy as np
from sympy import var, diff
from sympy.utilities.lambdify import lambdify
from tensorflow.examples.tutorials.mnist import input_data

import netFunctions as nf
import utils as ut

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
            if self.activations[i + 1] == nf.sigmoid:
                self.primes[i + 1] = nf.sigmoid_
            elif self.activations[i + 1] == nf.ReLU:
                self.primes[i + 1] = nf.ReLU_
            elif self.activations[i + 1] == nf.tanh:
                self.primes[i + 1] = nf.tanh_
            elif self.activations[i + 1] == nf.linear:
                self.primes[i + 1] = nf.linear_
            else:
                self.primes[i + 1] = lambdify(x, diff(self.activations[i + 1](x), x))

    # Forward propagation
    def feed_forward(self, x):
        z = {}
        # The first layer has no ‘real’ activations, so we consider the inputs x as the activations of the previous
        # layer.
        a = {1: x}

        for i in range(1, self.n_layers):
            z[i + 1] = np.dot(a[i], self.W[i]) + self.B[i]
            a[i + 1] = self.activations[i](z[i + 1])

        # Output
        return z, a

    def predict(self, x):
        _, a = self.feed_forward(x)
        return a[self.n_layers]

    def compute_error(self, data_set):
        # Calcolo dell'errore sul data set
        error = 0
        for n in range(len(data_set)):
            out = self.predict(data_set[n]['input'])
            error += self.error_function(out, data_set[n]['label'])
        return error

    # learning online
    def train_net_online(self, training_set, validation_set, max_epoch, eta, alpha):
        best_error_validation = float("inf")  # float("inf") è il valore float dell'inifinito, quindi garantisce
        error_training = np.zeros(max_epoch)    # che è il numero più grande rappresentabile dal sistema
        error_validation = np.zeros(max_epoch)
        for t in range(max_epoch):
            # permutazione del training set
            np.random.shuffle(training_set)
            for n in range(len(training_set)):
                act, out = self.feed_forward(training_set[n]['input'])
                derivatives = self.back_propagation(training_set[n]['input'], training_set[n]['label'], out, act)
                self.update_weights(derivatives, eta)
                if n % 1000 == 0:  # Queste due righe le ho messe per capire
                    print("t", t, "n ", n)  # in fase di run a che punto sta
            # Calcolo dell'errore sul training set
            error_training[t] += self.compute_error(training_set)
            # Calcolo dell'errore sul validation
            error_validation[t] += self.compute_error(validation_set)
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
        return np.resize(error_training, t), np.resize(error_validation, t)

    def train_net_batch(self, training_set, validation_set, max_epoch, eta, alpha):
        best_error_validation = float("inf")  # float("inf") è il valore float dell'inifinito, quindi garantisce
        error_training = np.zeros(max_epoch)    # che è il numero più grande rappresentabile dal sistema
        error_validation = np.zeros(max_epoch)
        lastDerivatives = {}
        updateValuesW = {}
        updateValuesB = {}
        for t in range(max_epoch):
            derivatives_tot = {}
            for n in range(len(training_set)):
                act, out = self.feed_forward(training_set[n]['input'])
                derivatives = self.back_propagation(training_set[n]['input'], training_set[n]['label'], out, act)
                if n == 0:
                    derivatives_tot = derivatives
                else:
                    for l in range(self.n_layers - 1):
                        derivatives_tot['weights'][l] = np.add(derivatives_tot['weights'][l], derivatives['weights'][l])
                        derivatives_tot['bias'][l] = np.add(derivatives_tot['bias'][l], derivatives['bias'][l])
                error_training[t] += self.error_function(out[self.n_layers], training_set[n]['label'])
                # if n % 1000 == 0:  # Queste due righe le ho messe per capire
                #     print("t", t, "n ", n)  # in fase di run a che punto sta
            print("errore a ", t , " = ", error_training[t])
            for l in range(self.n_layers - 1):
                derivatives_tot['weights'][l] /= len(training_set)
                derivatives_tot['bias'][l] /= len(training_set)
            if not any(lastDerivatives):
                self.update_weights(derivatives_tot, eta)
            else:
                # Applico la RPROP
                if t == 1:
                    for k in range(self.n_layers - 1):
                        updateValuesW[k + 1] = np.empty([self.W[k + 1].shape[1], self.W[k + 1].shape[0]])
                        updateValuesB[k + 1] = np.empty(self.B[k + 1].shape)
                for l in range(1, self.n_layers):
                    for n in range(self.W[l].shape[1]):
                        for w in range(self.W[l].shape[0]):
                            if t == 1:
                                updateValuesW[l][n][w] = 0.0125  # DELTA0 per ogni connessione, di ogni neurone,
                                                                # di ogni livello
                            # Aggiorno i pesi
                            self.W[l][w][n] += self.RPROP(derivatives_tot['weights'][l - 1][n],
                                                          lastDerivatives['weights'][l - 1][n],
                                                          updateValuesW[l][n], w, error_training[t],
                                                          error_training[t - 1])
                        if t == 1:
                            updateValuesB[l][n] = 0.0125  # DELTA0 per ogni bias, di ogni neurone, di ogni livello
                        # Aggiorno i Bias
                        self.B[l][n] += self.RPROP(derivatives_tot['bias'][l - 1], lastDerivatives['bias'][l - 1],
                                                   updateValuesB[l], n, error_training[t], error_training[t - 1])
            lastDerivatives = derivatives_tot
            # Calcolo dell'errore sul validation
            error_validation[t] += self.compute_error(validation_set)
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
        return np.resize(error_training, t + 1), np.resize(error_validation, t + 1)

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
            lastDelta[i] = delta = min(lastDelta[i] * ETA_P, MAX_STEP)
            deltaW = -np.sign(derivatives[i]) * delta
        elif change < 0:
            deltaW = 0
            delta = max(lastDelta[i] * ETA_M, MIN_STEP)
            if actualError > lastError:
                deltaW = -lastDelta[i]
            lastDelta[i] = delta
            derivatives[i] = 0
        else:
            deltaW = -np.sign(derivatives[i]) * lastDelta[i]
        return deltaW


    def update_weights(self, derivatives, eta):
        # Aggiornamento dei pesi Metodo discesa del gradiente
        for l in range(1, self.n_layers):
            self.W[l] = self.W[l] - eta * derivatives['weights'][l - 1].transpose()
            self.B[l] = self.B[l] - eta * derivatives['bias'][l - 1]


def PCA(data_set, soglia):
    # calcola il vettore media del dataset
    # cov_mat = np.cov(data_set.T)
    mean_vec = np.mean(data_set, axis=0)
    cov_mat = (data_set - mean_vec).T.dot((data_set - mean_vec))  # / (data_set.shape[0])
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


mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
training_set = []
validation_set = []
test_set = []
#   Mi prendo le prime 200 immagini per il training e le prime 100 per il validation e per il test
for i in range(200):
    elem = {'input': mnist.train.images[i], 'label': mnist.train.labels[i]}
    training_set.append(elem)
    if i < 100:
        elem = {'input': mnist.validation.images[i], 'label': mnist.validation.labels[i]}
        validation_set.append(elem)
        elem = {'input': mnist.test.images[i], 'label': mnist.test.labels[i]}
        test_set.append(elem)
var('x')
print("Scegli cosa vuoi fare dalla seguente lista:")
while True:
    print("0) Chiudi il programma\n"
          "1) Effettua un training di una rete neurale in maniera manuale\n"
          "2) Confronto tra training con PCA e training con rete autoassociativa\n")
    choice = ut.getUserAmount(0, 2)
    if choice == 0:
        sys.exit()
    elif choice == 1:
        print("Quanti strati interni vuoi all'interno della tua rete?")
        n_layers = ut.getUserAmount(1, 10)
        dimensions = np.zeros(n_layers + 2)
        dimensions[0] = len(training_set[0]['input'])
        dimensions[n_layers + 1] = 10
        functions = {}
        for l in range(1, n_layers + 1):
            print("Inserisci il numero di nodi al livello {}".format(l))
            dimensions[l] = ut.getUserAmount(1, 900)
            functions[l] = ut.getActivation(l)
        functions[n_layers + 1] = ut.getActivation(n_layers + 1)
        NN = Net(dimensions, functions, ut.getErrorFunc())
        print("Vuoi utilizzare il learning batch o online?\n"
              "0) Batch\n"
              "1) Online\n")
        if ut.getUserAmount(0, 1):
            NN.train_net_online(training_set, validation_set, 50, 0.5, 10)
        else:
            NN.train_net_batch(training_set, validation_set, 50, 0.5, 10)

        print("\n" * 10)
        continue
    elif choice == 2:
        # TEST PCA
        soglia_pca = 0.7
        new_dataset, matrix_w = PCA(mnist.train.images[:200], soglia_pca)
        training_set_PCA = []
        validation_set_PCA = []
        test_set_PCA = []
        for i in range(200):
            elem = {'input': new_dataset[i], 'label': mnist.train.labels[i]}
            training_set_PCA.append(elem)
            if i < 100:
                elem = {'input': np.dot(validation_set[i]['input'], matrix_w), 'label': mnist.validation.labels[i]}
                validation_set_PCA.append(elem)
                elem = {'input': np.dot(test_set[i]['input'], matrix_w), 'label': mnist.test.labels[i]}
                test_set_PCA.append(elem)
        dimensions = np.zeros(3)
        dimensions[0] = len(training_set_PCA[0]['input'])
        print("inserisci il numero di nodi nello strato nascosto")
        dimensions[1] = ut.getUserAmount(1, 900)
        dimensions[2] = 10
        functions = {1: ut.getActivation(1), 2: ut.getActivation(2)}
        NN_PCA = Net(dimensions, functions, ut.getErrorFunc())
        NN_PCA.train_net_batch(training_set_PCA, validation_set_PCA, 50, 0.5, 10)

        # Test Rete Autoassociativa

        print("\n" * 10)
        continue

