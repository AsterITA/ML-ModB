import numpy as np

class ForwardPropagation:
    def __init__(self, x):
        # Input
        self.x=x

        self.z={}
        self.a={}

        # The first layer has no ‘real’ activations, so we consider the inputs x as the activations of the previous layer.
        self.a={1: self.x}

        for i in range(1, net.n_layers):
            self.z[i+1]=np.dot(self.a[i], net.W[i]) + net.B[i]
            self.a[i+1]=net.activations[i+1](self.z[i+1])

        # Output of the neural network
        return self.a[net.n_layers]