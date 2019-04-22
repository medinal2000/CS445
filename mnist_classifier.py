import numpy
import math

class MNISTClassifier():
    def __init__(self):
        self.num_weights = 785
        self.num_outputs = 10
        self.last_outputs = numpy.empty(self.num_outputs)
        self.max_index = None
        self.input_weights = numpy.empty((self.num_weights, self.num_outputs))
        for i in range(self.num_weights):
            for j in range(self.num_outputs):
                self.input_weights[i][j] = numpy.random.uniform(-0.05, 0.05)        


    def forward(self, inputs):
        self.last_outputs = numpy.empty(self.num_outputs)
        max_val = -math.inf
        self.max_index = None
        for i in range(self.num_outputs):
            weights = self.input_weights[:,i]
            self.last_outputs[i] = numpy.dot(inputs, weights)
            if self.last_outputs[i] > max_val:
                max_val = self.last_outputs[i]
                self.max_index = i
        return self.max_index
            
        
    def back_prop(self, learning_rate, inputs, label):
        for i in range(self.num_outputs):
            if i == label:
                t = 1
            else:
                t = 0
            if self.last_outputs[i] > 0:
                y = 1
            else:
                y = 0
            
            weight_update = learning_rate * (t - y) * inputs
            self.input_weights[:,i] += weight_update
        #print(self.input_weights[0][0])

    