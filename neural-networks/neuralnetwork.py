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

        # Pesos
        self.weigths_ih = [[rd.randint(-1, 1) for i in range(i_nodes)] for i in range(h_nodes)]
        self.weigths_ho = [[rd.randint(-1, 1) for i in range(h_nodes)] for i in range(o_nodes)]

        # dW(t - 1)
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

    def updates(self, weights, vb_weights, inputs, delt):
        
        hidden = len(weights)
        inputsL = len(inputs)

        for i in range(hidden):
            for j in range(inputsL):
                percent = self.a_variation * vb_weights[i][j]
                variant = (self.learning_rate * (inputs[j] * delt[i]) ) + percent
                new_weight = weights[i][j] + variant
                
                vb_weights[i][j] = variant
                weights[i][j] = new_weight

        return vb_weights, weights
    
    def update(self, weights, vb_weights, inputs, delt):
        
        inputsL = len(inputs)
        
        for j in range(inputsL):
            percent = self.a_variation * vb_weights[j]
            variant = (self.learning_rate * (inputs[j] * delt[j]))  + percent
            new_weight = weights[j] + variant
            
            vb_weights[j] = variant
            weights[j] = new_weight

        return vb_weights, weights

    def update_weights(self, inputs, hidden_o, delta_o, delta_h):
        qtd_bias_ho = [1 for i in range(self.o_nodes)]
        qtd_bias_ih = [1 for i in range(self.h_nodes)]

        # Weights Bias Hidden   
        self.vb_weights_ih, self.weigths_ih = self.updates(self.weigths_ih, self.vb_weights_ih, inputs, delta_h)

        # Weights Bias Output      
        self.vb_bias_ho, self.bias_ho = self.update(self.bias_ho, self.vb_bias_ho, qtd_bias_ho, delta_o)        

        # Weights Output                        
        self.vb_weights_ho, self.weigths_ho = self.updates(self.weigths_ho, self.vb_weights_ho, hidden_o, delta_o)
        
        # Weights Bias Hidden
        self.vb_bias_ih, self.bias_ih = self.update(self.bias_ih, self.vb_bias_ih, qtd_bias_ih, delta_h)


    def mse(self, outputs, answers):
        e = 0
        for i in range(len(answers)):
            e = e + ((answers[i] - outputs[i]) ** 2)
        
        mse = e / len(answers)

        return mse

    def train(self, dataset, answers):
        all_outputs = []
        
        for i in range(len(dataset)):
            # feedfoward         
            hidden_outputs = self.fu(dataset[i], self.weigths_ih, self.h_nodes, self.bias_ih)
        
            outputs = self.fu(hidden_outputs, self.weigths_ho, self.o_nodes, self.bias_ho)

            # backpropagation
            delta_o = self.delt_o(outputs, answers[i])
            
            delta_h = self.delt_h(hidden_outputs, delta_o)            

            self.update_weights(dataset[i], hidden_outputs, delta_o, delta_h)

            all_outputs.append(outputs[0])

        return self.mse(all_outputs, answers)
    
    def predict(self, dataset):
        for i in range(len(dataset)):
            hidden_outputs = self.fu(dataset[i], self.weigths_ih, self.h_nodes, self.bias_ih)            
        
            outputs = self.fu(hidden_outputs, self.weigths_ho, self.o_nodes, self.bias_ho)
            
            print('')
            print(outputs)

            if (outputs[0] >= 0) and (outputs[0] < 0.45):
                print('Iris-Setosa')
                
            elif (outputs[0] >= 0.45) and (outputs[0] < 0.8):
                print('Iris-virginica')
                
            elif (outputs[0] >= 0.8) and (outputs[0] <= 1):                
                print('Iris-versicolor')
            