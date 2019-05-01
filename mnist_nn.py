import numpy
import math

class MNISTClassifier():
    def __init__(self, num_hidden):
        self.num_inputs = 785
        self.num_hidden_nodes = num_hidden + 1
        self.num_outputs = 10
        
        # saves the last values of the output/hidden layer for the last time forward is called
        self.last_outputs = None
        self.last_hidden_output = None
        
        # weight matrices
        self.hidden_weights = numpy.empty((self.num_inputs, self.num_hidden_nodes))
        self.output_weights = numpy.empty((self.num_hidden_nodes, self.num_outputs))
        
        # save previous weight updates for applying momentum
        self.hidden_weight_update = numpy.zeros((self.num_inputs, self.num_hidden_nodes))
        self.output_weight_update = numpy.zeros((self.num_hidden_nodes, self.num_outputs))
        
        # set initial weights of hidden and output layers
        for i in range(self.num_inputs):
            for j in range(self.num_hidden_nodes):
                self.hidden_weights[i][j] = numpy.random.uniform(-0.05, 0.05)
        
        for i in range(self.num_hidden_nodes):
            for j in range(self.num_outputs):
                self.output_weights[i][j] = numpy.random.uniform(-0.05, 0.05)    


    def forward(self, inputs):
        self.last_outputs = numpy.empty(self.num_outputs)
        self.last_hidden_output = numpy.empty(self.num_hidden_nodes)
        
        max_val = -math.inf
        max_index = None
        
        # hidden layer
        for i in range(self.num_hidden_nodes - 1):
            weights_h = self.hidden_weights[:,i]
            self.last_hidden_output[i] = numpy.dot(inputs, weights_h)
        self.last_hidden_output[self.num_hidden_nodes - 1] = 1        # for the bias in output layer
        self.last_hidden_output = self.sigmoid_func(self.last_hidden_output)
        
        # output layer
        for i in range(self.num_outputs):
            self.last_outputs[i] = numpy.dot(self.last_hidden_output, self.output_weights[:,i])
            if self.last_outputs[i] > max_val:
                max_val = self.last_outputs[i]
                max_index = i
        self.last_outputs = self.sigmoid_func(self.last_outputs)
        
        return max_index
            
        
    def back_prop(self, inputs, label, learning_rate=0.1, momentum=0.9):
        # calculate loss for output layer
        output_loss = numpy.empty(self.num_outputs)
        for i in range(self.num_outputs):
            if i == label:
                t = 0.9
            else:
                t = 0.1
            if self.last_outputs[i] >= 0.9:
                y = 0.9
            else:
                y = 0.1
            error = t - y 
            output_loss[i] = self.last_outputs[i] * (1.0 - self.last_outputs[i]) * error
            #print(self.last_outputs[i], error)
        
        # calculate loss for hidden layer
        hidden_loss = numpy.empty(self.num_hidden_nodes)
        for i in range(self.num_hidden_nodes):
            weights = self.output_weights[i,:]
            hidden_loss[i] = (self.last_hidden_output[i] * (1.0 - self.last_hidden_output[i]) 
                             * numpy.dot(weights, output_loss)) 
            
        # output layer weights update
        for i in range(self.num_hidden_nodes):
            new_output_weight_update = learning_rate * output_loss * self.last_hidden_output[i]
            self.output_weights[i,:] += new_output_weight_update + momentum * self.output_weight_update[i]
            self.output_weight_update[i,:] = new_output_weight_update
            
        for i in range(self.num_inputs):
            new_hidden_weight_update = learning_rate * hidden_loss * inputs[i]
            self.hidden_weights[i,:] += new_hidden_weight_update + momentum * self.hidden_weight_update[i]
            self.hidden_weight_update[i,:] = new_hidden_weight_update
            
    
    def sigmoid_func(self, inputs):
        result = 1.0 / (1.0 + numpy.exp(-inputs))
        return result
        

    
