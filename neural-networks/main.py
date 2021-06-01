import os
import csv
from neuralnetwork import NeuralNetwork

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    train = True
    dataset = []

    if 'iris_train.txt' in os.listdir(ROOT_DIR):
        with open(f'{ROOT_DIR}/iris_train.txt') as f:
            iris = csv.reader(f, delimiter=',')

            answers = []

            for data in iris:
                inputs = []

                inputs.append(float(data[0]))
                inputs.append(float(data[1]))
                inputs.append(float(data[2]))
                inputs.append(float(data[3]))

                answers.append(float(data[4]))

                dataset.append(inputs)

    training = []

    if 'iris_test.txt' in os.listdir(ROOT_DIR):
        with open(f'{ROOT_DIR}/iris_test.txt') as f:
            iris = csv.reader(f, delimiter=',')

            for row in iris:
                inputs = []
                
                inputs.append(float(row[0]))
                inputs.append(float(row[1]))
                inputs.append(float(row[2]))
                inputs.append(float(row[3]))

                training.append(inputs)

    nn = NeuralNetwork(4, 8, 1)

    while (train):        
        result = nn.train(dataset, answers)
        if (result <= 0.001):
            train = False
        print(result)
        
    nn.predict(training)