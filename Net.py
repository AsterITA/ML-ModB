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

        for i in range(1, len(dimensions)):

            self.W[i] = np.random.randn(dimensions[i - 1], dimensions[i]) / np.sqrt(dimensions[i - 1])  # Pesi
            # The drawn weights are eventually divided by the square root of the current layers dimensions.
            # This is called Xavier initialization and helps prevent neuron activations from being too large or too
            self.B[i] = np.zeros(dimensions[i])  # Bias
            if self.activations[i] == nf.sigmoid:
                self.primes[i] = nf.sigmoid_
            elif self.activations[i] == nf.ReLU:
                self.primes[i] = nf.ReLU_
            elif self.activations[i] == nf.tanh:
                self.primes[i] = nf.tanh_
            elif self.activations[i] == nf.identity:
                self.primes[i] = nf.identity_
            else:
                self.primes[i] = lambdify(x, diff(self.activations[i](x), x))

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
        best_W = best_B = 0
        for t in range(max_epoch):
            # permutazione del training set
            np.random.shuffle(training_set)
            for n in range(len(training_set)):
                act, out = self.feed_forward(training_set[n]['input'])
                derivatives = self.back_propagation(training_set[n]['input'], training_set[n]['label'], out, act)
                self.update_weights(derivatives, eta)
            # Calcolo dell'errore sul training set
            error_training[t] += self.compute_error(training_set)
            # Calcolo dell'errore sul validation
            error_validation[t] += self.compute_error(validation_set)
            print("errore a ", t, " = ", error_training[t])
            # La rete di questo passo è migliore?
            if best_error_validation > error_validation[t]:
                best_error_validation = error_validation[t]
                best_W = cp.copy(self.W)
                best_B = cp.copy(self.B)
            # Applico il criterio di fermata GL descritto in "Early Stopping, but when?"
            glt = 100 * (error_validation[t] / best_error_validation - 1)
            if glt > alpha:
                break
        if best_W != 0:
            self.W = best_W
            self.B = best_B
        return np.resize(error_training, t), np.resize(error_validation, t)

    def train_net_batch(self, training_set, validation_set, max_epoch, eta, alpha):
        best_error_validation = float("inf")  # float("inf") è il valore float dell'inifinito, quindi garantisce
        error_training = np.zeros(max_epoch)    # che è il numero più grande rappresentabile dal sistema
        error_validation = np.zeros(max_epoch)
        best_W = best_B = 0
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
            print("errore a ", t , " = ", error_training[t])
            for l in range(self.n_layers - 1):
                derivatives_tot['weights'][l] /= len(training_set)
                derivatives_tot['bias'][l] /= len(training_set)
            if not any(lastDerivatives):
                self.update_weights(derivatives_tot, eta)
            else:
                # Applico la RPROP
                if t == 1:
                    for k in range(1, self.n_layers):
                        updateValuesW[k] = np.empty([self.W[k].shape[1], self.W[k].shape[0]])
                        updateValuesB[k] = np.empty(self.B[k].shape)
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
        if best_W != 0:
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
    mean_vec = np.mean(data_set, axis=0)
    cov_mat = (data_set - mean_vec).T.dot((data_set - mean_vec))  # / (data_set.shape[0])
    # Calcolo autovalori e autovettori
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    # Calcolo la somma totale degli autovalori
    eig_vals_tot = sum(eig_vals)
    #Creazione liste di tuple del tipo (autovalore, autovettore)
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
    # Ordinamento decrescente della lista delle tuple in base agli autovalori
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
        print("inserisci il valore di eta : ")
        eta = ut.getUserAmountFloat(0, 1, False, False)
        print("inserisci il numero massimo di epoche : ")
        max_epoche = ut.getUserAmount(10, 3000)
        print("inserisci il valore di alpha per la Generalization Loss : ")
        alpha = ut.getUserAmountFloat(1, 100)
        print("Vuoi utilizzare il learning batch o online?\n"
              "1) Batch\n"
              "2) Online\n")
        if ut.getUserAmount(1, 2) == 2:
            error_train, error_valid = NN.train_net_online(training_set, validation_set, max_epoche, eta, alpha)
        else:
            error_train, error_valid = NN.train_net_batch(training_set, validation_set, max_epoche, eta, alpha)
        ut.plotGraphErrors(error_train, error_valid, "Addestramento della rete senza riduzione delle dimensioni")
        risp_giuste = ut.getRightNetResponse(NN, test_set)
        print("La rete con input l'output interno della rete autoassociativa ha risposto correttamente a ", risp_giuste,
              "in percentuale ", 100 * risp_giuste / len(test_set), "%")
        print("\n" * 3)
        continue
    elif choice == 2:
        # TEST PCA
        print("Inserisci la soglia del quantitativo di informazione da preservare dalla PCA")
        soglia_pca = ut.getUserAmountFloat(50, 100) / 100
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
        dimensions = np.empty(3)
        dimensions[0] = len(training_set_PCA[0]['input'])
        print("inserisci il numero di nodi nello strato nascosto")
        dimensions[1] = ut.getUserAmount(1, 900)
        dimensions[2] = 10
        functions = {1: ut.getActivation(1), 2: ut.getActivation(2)}
        print("inserisci il valore di eta : ")
        eta = ut.getUserAmountFloat(0, 1, False, False)
        print("inserisci il numero massimo di epoche : ")
        max_epoche = ut.getUserAmount(10, 3000)
        print("inserisci il valore di alpha per la Generalization Loss : ")
        alpha = ut.getUserAmountFloat(1, 100)
        error_function = ut.getErrorFunc()
        NN_PCA = Net(dimensions, functions, error_function)
        err_train_PCA, err_valid_PCA = NN_PCA.train_net_batch(training_set_PCA, validation_set_PCA, max_epoche, eta, alpha)
        ut.plotGraphErrors(err_train_PCA, err_valid_PCA, "Addestramento rete con input l'out dell PCA")
        print("Rete con input della PCA addestrata\n\n")
        print("Creazione della rete autoassociativa")
        # Test Rete Autoassociativa
        #addestramento rete autoassociativa
        #genero dataset per training rete autoassociativa
        #training set
        training_set_R = []
        for i in range(200):
            elem = {'input': mnist.train.images[i], 'label': mnist.train.images[i]}
            training_set_R.append(elem)
        #validation set
        validation_set_R = []
        for i in range(100):
            elem = {'input': mnist.validation.images[i], 'label': mnist.validation.images[i]}
            validation_set_R.append(elem)
        #creazione rete associativa
        hidden_layers = ut.getNumbHiddenLayerRA()
        dimensions_RA = np.empty(hidden_layers + 2)
        #fisso il numero di nodi del livello di input e del livello di output
        dimensions_RA[0] = dimensions_RA[hidden_layers + 1] = len(mnist.train.images[0])
        #fisso il numero di nodi interni
        if hidden_layers == 1:
            dimensions_RA[1] = len(training_set_PCA[0]['input'])
        else:
            dimensions_RA[2] = len(training_set_PCA[0]['input'])
            print("inserisci il numero di nodi del primo strato nascosto")
            dimensions_RA[1] = ut.getUserAmount(int(dimensions_RA[2] + 1), int(dimensions_RA[0]))
            print("inserisci il numero di nodi del terzo strato nascosto")
            dimensions_RA[3] = ut.getUserAmount(dimensions_RA[2] + 1, dimensions_RA[0])
        #fisso le funzioni di attivazione per ogni livello
        functions_RA = {}
        for l in range(1, len(dimensions_RA)):
            functions_RA[l] = ut.getActivation(l)
        #inserimento parametri di learning
        print("inserisci il valore di eta : ")
        eta_RA = ut.getUserAmountFloat(0, 1, False, False)
        print("inserisci il numero massimo di epoche : ")
        max_epoche_RA = ut.getUserAmount(10, 3000)
        print("inserisci il valore di alpha per la Generalization Loss : ")
        alpha_RA = ut.getUserAmountFloat(1, 100)

        NN_R = Net(dimensions_RA, functions_RA, nf.sum_square)
        print("Addestramento della rete autoassociativa iniziato")
        err_train_R, err_valid_R = NN_R.train_net_batch(training_set_R, validation_set_R, 100, 0.5, 10)
        ut.plotGraphErrors(err_train_R, err_valid_R, "Addestramento rete autoassociativa")
        print("Addestramento della rete autoassociativa completato")
        print("conversione dataset con la rete autoassociativa")
        #creazione data set con dimensione ridotta
        training_set_RA = []
        validation_set_RA = []
        test_set_RA = []
        #Creo una rete NN_R2 uguale a NN_R che non ha lo strato di output
        livelli_R2 = int(len(dimensions_RA) / 2) + 1
        NN_R2 = Net(dimensions_RA[:livelli_R2], functions_RA, nf.sum_square)
        for l in range(1, livelli_R2 + 1):
            NN_R2.W[l] = NN_R.W[l]
            NN_R2.B[l] = NN_R.B[l]
        for i in range(200):
            out = NN_R2.predict(training_set[i]['input'])
            elem = {'input': out, 'label': mnist.train.labels[i]}
            training_set_RA.append(elem)
            if i < 100:
                out = NN_R2.predict(validation_set[i]['input'])
                elem = {'input': out, 'label': mnist.validation.labels[i]}
                validation_set_RA.append(elem)
                out = NN_R2.predict(test_set[i]['input'])
                elem = {'input': out, 'label': mnist.test.labels[i]}
                test_set_RA.append(elem)
        #Addestramento di una rete con le dimensioni del dataset ridotte con una rete associativa
        print("Creazione  e addestramento della rete con input l'out interno della rete autoassociativa")
        NN_RA = Net(dimensions, functions, error_function)
        err_train_RA, err_valid_RA = NN_RA.train_net_batch(training_set_RA, validation_set_RA, max_epoche, eta, alpha)
        ut.plotGraphErrors(err_train_RA, err_valid_RA, "Addestramento rete con out interno della rete autoassociativa")
        print("Elaborazione delle due reti cui rispettivi test set e calcolo delle risposte giuste")
        #PCA
        pca_giuste = ut.getRightNetResponse(NN_PCA, test_set_PCA)
        print("La rete con input l'output della PCA ha risposto correttamente a ", pca_giuste,
              "in percentuale ", 100 * pca_giuste / len(test_set_PCA), "%")
        #RA
        ra_giuste = ut.getRightNetResponse(NN_RA, test_set_RA)
        print("La rete con input l'output interno della rete autoassociativa ha risposto correttamente a ", ra_giuste,
              "in percentuale ", 100 * ra_giuste/len(test_set_RA), "%")
       
        print("\n" * 3)
        continue

