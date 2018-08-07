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
            for n in range(len(training_set)):
                act_and_out = net.forward_propagation_with_output_hidden(training_set[n]['input'])
                deltas = self.back_propagation(net, training_set[n]['label'], act_and_out['outputs'], act_and_out['activations'])
                derivatives = self.calc_derivatives(net, training_set[n]['input'], act_and_out['outputs'], deltas)
                self.update_weights(net, derivatives)
            #Calcolo dell'errore sul training set
            error_training = 0
            for n in range(len(training_set)):
                output = net.forward_propagation(training_set[n]['input'])
                error_training += self.error_function(output, training_set[n]['label'])
            #Calcolo dell'errore sul validation set
            error_validation = 0
            for n in range(len(validation_set)):
                output = net.forward_propagation(validation_set[n]['input'])
                error_validation += self.error_function(output, validation_set[n]['label'])
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

    def back_propagation(self, net, labels, outputs, node_act):
        #Calcolo delta
        deltas = np.ndarray(net.n_layers, dtype=np.ndarray)
        deltas[net.n_layers - 1] = net.primes[net.n_layers - 1](node_act[net.n_layers - 1]) * (outputs[net.n_layers - 1] - labels)
        for l in range(net.n_layers - 2, -1, -1):
            deltas[l] = np.dot(deltas[l + 1], net.W[l+1])
            deltas[l] = net.primes[l](node_act[l]) * deltas[l]
        return deltas

    def calc_derivatives(self, net, input, outputs, deltas):
        #Calcolo derivate
        derivate_W = []
        derivate_B = []
        Z = input
        for l in range(net.n_layers):
            derivate_W.append(np.dot(deltas[l][:,np.newaxis]), Z[np.newaxis,:])
            derivate_B.append(deltas[l])
            Z = outputs[l]
        derivatives = {'weights': derivate_W, 'bias': derivate_B}
        return derivatives

    def update_weights(self, net, derivatives):
        #Aggiornamento dei pesi
        for l in range(net.n_layers - 1):
            net.W[l] = net.W[l] - self.eta * derivatives['weights'][l]
            net.B[l] = net.B[l] - self.eta * derivatives['bias'][l]



#LE RIGHE SUCCESSIVE MI SERVIRANNO PER FARE UN TEST
import Net
def sum_square(t,y):
    err=0
    for i in range(y.size):
        err+=(y[i]-t[i])**2
        err/=2
    return err

#Genero un training set
training_set = []
for i in range(10):
    inpu = np.random.randint(0,4,4)
    app = np.random.randint(0,2,1)
    out = np.zeros(3)
    out[app] = 1
    elem = {'input':inpu, 'label': out}
    training_set.append(elem)
print(training_set)
#Creazione rete
#Addestramento rete
