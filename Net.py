import copy as cp
import numpy as np
from sympy import var, diff
from sympy.utilities.lambdify import lambdify
import netFunctions as nf


class Net:
    #Inizializzatore
    def __init__(self, dimensions, activations, error_function):
        self.n_layers = len(dimensions)
        self.activations = activations
        self.primes = {}
        self.error_function = error_function
        self.W = {}
        self.B = {}
        dimensions = dimensions.astype(int)

        #Eventuale calcolo della derivata della funzione di errore
        if error_function != nf.cross_entropy and error_function != nf.sum_square:
            var('t y')
            self.error_function_ = lambdify((t, y), diff(error_function(t, y), y))
        for i in range(1, len(dimensions)):
            #Pesi generati casualmente mediante il metodo di Xavier
            self.W[i] = np.random.randn(dimensions[i - 1], dimensions[i]) / np.sqrt(dimensions[i - 1])
            #Bias
            self.B[i] = np.zeros(dimensions[i])
            #Derivate delle rispettive funzioni di attivazione
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
        a = {}
        #L'input è considerato il primo strato della rete
        z = {1: x}
        for i in range(1, self.n_layers):
            #Attivazione
            a[i + 1] = np.dot(z[i], self.W[i]) + self.B[i]
            #Uscita di ogni neurone
            z[i + 1] = self.activations[i](a[i + 1])
        # Output
        return a, z

    #Estrapola l'ultimo output
    def predict(self, x):
        _, z = self.feed_forward(x)
        return z[self.n_layers]

    #Calcolo dell'errore sul data set
    def compute_error(self, data_set):
        error = 0
        for n in range(len(data_set)):
            out = self.predict(data_set[n]['input'])
            #Controlla se la funzione d'errore è stata definita da input
            if hasattr(self, 'error_function_'):
                error += sum(self.error_function(data_set[n]['label'], out))
            else:
                error += self.error_function(data_set[n]['label'], out)
        return error

    #Learning online
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

    def train_net_batch(self, training_set, validation_set, max_epoch, eta, alpha, eta_p=1.2, eta_m=0.5, max_step=50, min_step=0):
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
                if hasattr(self, 'error_function_'):  # controllo se la funzione d'errore è stata definita da input
                    error_training[t] += sum(self.error_function(training_set[n]['label'], out[self.n_layers]))
                else:
                    error_training[t] += self.error_function(training_set[n]['label'], out[self.n_layers])
            print("errore a ", t , " = ", error_training[t])
            for l in range(self.n_layers - 1):
                derivatives_tot['weights'][l] /= len(training_set)
                derivatives_tot['bias'][l] /= len(training_set)
            if not any(lastDerivatives):
                updateValuesW = cp.copy(self.W)
                updateValuesB = cp.copy(self.B)
                self.update_weights(derivatives_tot, eta)
                for l in range(1, self.n_layers):
                    updateValuesW[l] = abs(self.W[l] - updateValuesW[l])
                    #trasposta
                    updateValuesW[l] = updateValuesW[l].reshape([updateValuesW[l].shape[1], updateValuesW[l].shape[0]])
                    updateValuesB[l] = abs(self.B[l] - updateValuesB[l])
            else:
                # Applico la RPROP
                for l in range(1, self.n_layers):
                    for n in range(self.W[l].shape[1]):
                        for w in range(self.W[l].shape[0]):
                            # Aggiorno i pesi
                            self.W[l][w][n] += self.RPROP(derivatives_tot['weights'][l - 1][n],
                                                          lastDerivatives['weights'][l - 1][n],
                                                          updateValuesW[l][n], w, error_training[t],
                                                          error_training[t - 1], eta_p, eta_m,
                                                          max_step, min_step)
                        # Aggiorno i Bias
                        self.B[l][n] += self.RPROP(derivatives_tot['bias'][l - 1], lastDerivatives['bias'][l - 1],
                                                   updateValuesB[l], n, error_training[t], error_training[t - 1],
                                                   eta_p, eta_m, max_step, min_step)
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
        if hasattr(self, 'error_function_'):
            deltas = {self.n_layers: self.primes[self.n_layers - 1](node_act[self.n_layers]) *
                                     self.error_function_(labels, outputs[self.n_layers])}
        elif self.error_function == nf.cross_entropy:
            deltas = {self.n_layers: self.primes[self.n_layers - 1](node_act[self.n_layers]) *
                                     (-labels / outputs[self.n_layers])}
        else:
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

    def RPROP(self, derivatives, lastDerivatives, lastDelta, i, actualError, lastError, eta_p, eta_m, max_step, min_step):
        change = np.sign(derivatives[i] * lastDerivatives[i])
        if change > 0:
            lastDelta[i] = delta = min(lastDelta[i] * eta_p, max_step)
            deltaW = -np.sign(derivatives[i]) * delta
        elif change < 0:
            deltaW = 0
            delta = max(lastDelta[i] * eta_m, min_step)
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