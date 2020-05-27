import numpy

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * numpy.random.randn(n_inputs, n_neurons)
        self.biases = numpy.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = numpy.dot(inputs, self.weights) + self.biases