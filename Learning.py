import numpy as np
import copy as cp
import Net
class MultiPerceptronLearning:
    def __init__(self, max_epoch, eta, error_function):
        self.max_epoch = max_epoch
        self.eta = eta
        self.error_function = error_function

    # learning online
    def train_net(self, training_set, validation_set, net):
        for t in range(self.max_epoch):
            for n in range(training_set.__sizeof__()):
                ########################
                # PROBABILMENTE DA MODIFICARE
                net = Net(net)#forse questa riga è inutile
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
            #PROPRIO QUI
        return best_net

    def back_propagation(self, net, labels, outputs):
        #Calcolo delta
        delta_out = net.primes[net.n_layers - 1](net.activations[net.n_layers - 1]) * (
                    outputs[net.n_layers - 1] - labels)
        deltas = np.ndarray(net.n_layers, dtype=np.ndarray)
        deltas[net.n_layers - 1] = delta_out
        for l in range(net.n_layers - 2, -1, -1):
            deltas[l] = np.dot(deltas[l + 1], net.W[l+1])
            deltas[l] *= net.primes[l](net.activations[l])
        return deltas

    def calc_derivatives(self, net, input, outputs, deltas):
        #Calcolo derivate
        return  deivatives

    def update_weights(self, net, derivatives):
        #Aggiornamento dei pesi