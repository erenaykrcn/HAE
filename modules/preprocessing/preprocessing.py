"""
    This function reads the training and test data from the 
    files located in the data folder and normalizes the x values
    to be between 0 and 1.
"""
import os
import numpy as np
from matplotlib import pyplot as plt


dirname = os.path.dirname(__file__)


def preprocess():
    """
        Reads the training and test data from the Statlog Satellite data and normalizes
        the data set. End results are 36 dimenstional x_training and x_test arrays and 
        1 dimensional label arrays.
    """

    training_file = open(os.path.join(dirname, '../../data/data_sets/statlog/sat.trn'), 'r')
    test_file = open(os.path.join(dirname, '../../data/data_sets/statlog/sat.tst'), 'r')

    training_lines = training_file.readlines()
    test_lines = test_file.readlines()

    # Each x data point is an array with 36 elements.
    x_training = []
    x_test = []

    y_training = []
    y_test = []

    for line in training_lines:
        line_elements = line.split(" ")
        training_data = [int(line_elements[index]) / 255 for index in range(len(line_elements) - 1)]
        x_training.append(training_data)
        y_training.append(int(line_elements[-1]))

    for line in test_lines:
        line_elements = line.split(" ")
        test_data = [int(line_elements[index]) / 255 for index in range(len(line_elements) - 1)]
        x_test.append(test_data)
        y_test.append(int(line_elements[-1]))

    return (x_training, y_training, x_test, y_test)


def visualize_test_data(label, data_index):
    x_training, y_training, x_test, y_test = preprocess()

    y_test = np.array(y_test)
    x_test = np.array(x_test)
    x_test_2 =  x_test[np.where(y_test == label)][data_index]

    array_to_visualize = [[[],[],[]],
                          [[],[],[]],
                          [[],[],[]]]

    index = 0
    for m in range(3):
        for n in range(3):
            for i in range(4):
                array_to_visualize[m][n].append(x_test_2[index*4 + i])
            index += 1

    fig_x_test_2 = plt.figure()
    plt.imshow(array_to_visualize)

    relative_file_path = f"../../data/visualized_test_data/statlog/class_{label}_index_{data_index}.png"
    fig_x_test_2.savefig(os.path.join(dirname, relative_file_path))
