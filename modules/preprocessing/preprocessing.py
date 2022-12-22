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

    x_training = np.array(x_training)
    y_training = np.array(y_training)

    # First 3 classes are seen as abnormal data.
    x_training = x_training[np.where(y_training>3)].tolist()
    y_training = y_training[np.where(y_training>3)].tolist()

    return (x_training, y_training, x_test, y_test)


def sample_training_data(n_samples):
    x_training, y_training, x_test, y_test = preprocess()
    sample_per_class = n_samples//3

    x_training = np.array(x_training)
    y_training = np.array(y_training)

    class4 = x_training[np.where(y_training==4)][:sample_per_class].tolist()
    label4 = np.ones(sample_per_class) * 4
    class5 = x_training[np.where(y_training==5)][:sample_per_class].tolist()
    label5 = np.ones(sample_per_class) * 5
    class7 = x_training[np.where(y_training==7)][:sample_per_class].tolist()
    label7 = np.ones(sample_per_class) * 7

    filtered_x = class4 + class5 + class7
    filtered_y = label4.tolist() + label5.tolist() + label7.tolist()

    return (filtered_x, filtered_y)

