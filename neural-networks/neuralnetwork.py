class NeuralNetwork:

    def __init__(self, i_nodes, h_nodes, o_nodes):
        self.i_nodes = i_nodes
        self.h_nodes = h_nodes
        self.o_nodes = o_nodes
        
        # self.bias_ih 
        # self.bias_ho

        # self.weigths_ih.randomize()
        # self.weigths_ho.randomize()

        self.learning_rate = 0.1