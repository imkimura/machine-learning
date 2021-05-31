from neuralnetwork import NeuralNetwork


if __name__ == '__main__':


    nn = NeuralNetwork(2, 3, 1)

    dataset = [
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ]

    answers = [0, 1, 1, 0]

    while (nn.train(dataset, answers) > 0.1):        
        print(nn.train(dataset, answers))

    nn.test([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    