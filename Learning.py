import numpy as np
import copy as cp


class MultiPerceptronLearning:
    def __init__(self, max_epoch, eta, error_function, alpha):
        self.max_epoch = max_epoch
        self.eta = eta
        self.error_function = error_function
        self.alpha = alpha

    # learning online
    def train_net_online(self, training_set, validation_set, net):
        for t in range(self.max_epoch):
            #permutazione del training set
            np.random.shuffle(training_set)
            for n in range(training_set.__sizeof__()):
                ########################
                # PROBABILMENTE DA MODIFICARE
                outputs = net.forward_propagation_with_output_hidden(training_set[n].input)
                deltas = self.back_propagation(net, training_set[n].labels, outputs)
                derivatives = self.calc_derivatives(net, training_set[n].input, outputs, deltas)
                self.update_weights(net, derivatives)
            #Calcolo dell'errore sul training set
            error_training = 0
            for n in range(training_set.__sizeof__()):
                output = net.forward_propagation(training_set[n].input)
                error_training += self.error_function #questo di sicuro è da modificare
            #Calcolo dell'errore sul validation set
            error_validation = 0
            for n in range(validation_set.__sizeof__()):
                output = net.forward_propagation(validation_set[n].input)
                error_validation += self.error_function  # questo di sicuro è da modificare
            #La rete di questo passo è migliore?
            try:
                if best_error_validation > error_validation:
                    best_error_validation = error_validation
                    best_net = cp.copy.copy(net)
            except NameError:
                best_error_validation = error_validation
                best_net = cp.copy.copy(net)
            #Volendo si può applicare il criterio di fermata
            glt = 100 * (error_validation / best_error_validation -1)
            if glt > self.alpha:
                break
        return best_net

    def back_propagation(self, net, labels, outputs):
        #Calcolo delta
        delta_out = net.primes[net.n_layers - 1](net.activations[net.n_layers - 1]) * (outputs[net.n_layers - 1] - labels)
        deltas = np.ndarray(net.n_layers, dtype=np.ndarray)
        deltas[net.n_layers - 1] = delta_out
        for l in range(net.n_layers - 2, -1, -1):
            deltas[l] = np.dot(deltas[l + 1], net.W[l+1])
            deltas[l] = net.primes[l](net.activations[l]) * deltas[l]#da modificare
        return deltas

    def calc_derivatives(self, net, input, outputs, deltas):
        #Calcolo derivate
        derivate_W = np.ndarray(net.livelli, dtype=np.ndarray)
        derivate_B = np.ndarray(net.livelli, dtype=np.ndarray)
        Z = input
        Z = Z.reshape(1, len(Z))
        for l in range(0, net.n_layers):
            derivate_W[l] = np.dot(deltas[l].T, Z)
            derivate_B[l] = deltas[l]
            Z = outputs[l]
        derivatives = {'weights': derivate_W, 'bias': derivate_B}
        return derivatives

    def update_weights(self, net, derivatives):
        #Aggiornamento dei pesi
        for l in range(net.n_layers - 1):
            self.W[l] = self.W[l] - self.eta * derivatives['weights'][l]
            self.B[l] = self.B[l] - self.eta * derivatives['bias'][l]
