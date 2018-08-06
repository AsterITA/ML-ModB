import numpy as np
import Net

class ForwardPropagation:
    def feed_forward (self, X):
        z={}

        a={1:x}
        for i in range(1, net.n_layers):
            z[1+1]=np.dot(a[i], self.w[i])+self.b[i]
            a[i]+1=net.activations[i+1](3)
        return a[net.n_layers]