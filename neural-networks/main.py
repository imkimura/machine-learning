from neuralnetwork import NeuralNetwork


if __name__ == '__main__':


    nn = NeuralNetwork(2, 2, 1)

    nn.train([1, 2, 3], [0, 0, 0])
    