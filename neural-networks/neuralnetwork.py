import numpy as np


class NeuralNetwork:

    def __init__(self, i_nodes, h_nodes, o_nodes):
        self.i_nodes = i_nodes
        self.h_nodes = h_nodes
        self.o_nodes = o_nodes
        
        self.bias_ih = [[0.1] for j in range(h_nodes)]
        self.bias_ho = [[0.2] for i in range(o_nodes)]

        self.weigths_ih = [[0.1 for i in range(i_nodes)] for i in range(h_nodes)]
        self.weigths_ho = [[0.2 for i in range(h_nodes)] for i in range(o_nodes)]

        self.learning_rate = 0.1

    def sigmoidOperation(self, inputs, weights, bias):

        response = np.dot(weights, np.array([[2], [1]]))   

        return response

    def train(self, dataset, answers):
        # feedfoward 
        print(self.weigths_ih)
        hidden_output = self.sigmoidOperation(self.i_nodes, self.weigths_ih, self.bias_ih)

        print(hidden_output)