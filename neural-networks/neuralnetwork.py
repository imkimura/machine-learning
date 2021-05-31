import numpy as np
import random as rd

class NeuralNetwork:

    def __init__(self, i_nodes, h_nodes, o_nodes):
        # Rede Parametros
        self.i_nodes = i_nodes
        self.h_nodes = h_nodes
        self.o_nodes = o_nodes
        
        # Bias
        self.bias_ih = [rd.randint(-1, 1) for i in range(h_nodes)]
        self.bias_ho = [rd.randint(-1, 1) for i in range(o_nodes)]
        # self.bias_ih = [-0.46, 0.45, 0.6]
        # self.bias_ho = [0.78]

        # Pesos
        self.weigths_ih = [[rd.randint(-1, 1) for i in range(i_nodes)] for i in range(h_nodes)]
        self.weigths_ho = [[rd.randint(-1, 1) for i in range(h_nodes)] for i in range(o_nodes)]
        # self.weigths_ih = [[0.40, 0.50, 0.5], [0.94, 0.46, 0.4]]
        # self.weigths_ho = [[0.22, 0.58, 0.5]]

        self.vb_bias_ih = [0 for i in range(h_nodes)]
        self.vb_bias_ho = [0 for i in range(o_nodes)]
        self.vb_weights_ih = [[0 for i in range(i_nodes)] for i in range(h_nodes)]
        self.vb_weights_ho = [[0 for i in range(h_nodes)] for i in range(o_nodes)]

        # Learning Rate
        self.learning_rate = 0.2
        self.a_variation = 0.6

    def fu(self, inputs, weights, outputs, bias):
        hidden_n = []
        for i in range(outputs):
            u = 0
            for j in range(len(inputs)):
                u = u + (inputs[j] * weights[i][j])                
            hidden_n.append(self.sigmoide(u + bias[i]))
        return hidden_n
    
    def sigmoide(self, u):
        return 1 / (1 + (2.7183 ** (-1 * u)))

    def delt_o(self, outputs, answer):
        delta_o = []

        for o in outputs:
            d = (o * (1-o)) * (answer - o)
            delta_o.append(d)

        return delta_o
    
    def delt_h(self, hiddens, delta_o):
        delta_h = []
        for i, h in zip(range(len(hiddens)), hiddens):
            d = ((h) * (1 - h)) * self.weigths_ho[0][i] * delta_o[0]
            delta_h.append(d)
        return delta_h

    def update_weights(self, inputs, hidden_o, delta_o, delta_h, second):

        for i in range(self.h_nodes):
            for j in range(self.i_nodes):
                percent = self.a_variation * self.vb_weights_ih[i][j]
                variant = self.learning_rate * (inputs[j] * delta_h[j])
                # if second: variant = variant * percent
                new_weight = self.weigths_ih[i][j] + (variant)
                self.vb_weights_ih[i][j] = variant
                self.weigths_ih[i][j] = new_weight

                percent_ho = self.a_variation * self.vb_weights_ho[0][j]
                variant_ho = self.learning_rate * (hidden_o[j] * delta_o[0])
                # if second: variant_ho = variant_ho * percent_ho
                new_weight_ho = self.weigths_ho[0][j] + variant_ho
                self.weigths_ho[0][j] = new_weight_ho
                self.vb_weights_ho[0][j] = variant_ho
            
        percent_bo = self.a_variation * self.vb_bias_ho[0]
        variant_bo = self.learning_rate * (1 * delta_o[0])
        # if second: variant_bo = variant_bo * percent_bo
        new_weight_bo = self.bias_ho[0] + variant_bo
        self.bias_ho[0] = new_weight_bo
        self.vb_bias_ho[0] = variant_bo
            
        for i in range(self.i_nodes):
            percent_bi = self.a_variation * self.vb_bias_ih[i]
            variant_bi = self.learning_rate * (1 * delta_o[0])
            # if second: variant_bi = variant_bi * percent_bi
            new_weight_bi = self.bias_ih[i] + variant_bi
            self.bias_ih[i] = new_weight_bi
            self.vb_bias_ih[i] = variant_bi

    def mse(self, outputs, answers):
        e = 0
        for i in range(len(answers)):
            e = e + ((answers[i] - outputs[i]) ** 2)
        
        mse = e / len(answers)

        return mse

    def train(self, dataset, answers):
        all_outputs = []
        second = False
        
        for i, inputs in zip(range(len(dataset)), dataset):
            # feedfoward         
            hidden_outputs = self.fu(inputs, self.weigths_ih, self.h_nodes, self.bias_ih)
            # print(hidden_outputs)
        
            outputs = self.fu(hidden_outputs, self.weigths_ho, self.o_nodes, self.bias_ho)
            # print(outputs)

            # backpropagation
            delta_o = self.delt_o(outputs, answers[i])
            # print(delta_o)
            
            delta_h = self.delt_h(hidden_outputs, delta_o)
            # print(delta_h)
            
            if i != 0: second = True

            self.update_weights(inputs, hidden_outputs, delta_o, delta_h, second)
            
            all_outputs.append(outputs[0])

        return self.mse(all_outputs, answers)
    
    def test(self, dataset):
        for i in range(len(dataset)):
            hidden_outputs = self.fu(dataset[i], self.weigths_ih, self.h_nodes, self.bias_ih)            
        
            outputs = self.fu(hidden_outputs, self.weigths_ho, self.o_nodes, self.bias_ho)
            
            print(outputs)
