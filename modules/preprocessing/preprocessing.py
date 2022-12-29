"""
    This function reads the training and test data from the 
    files located in the data folder and normalizes the x values
    to be between 0 and 1.
"""
import os
dirname = os.path.dirname(__file__)

import numpy as np



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


def sample_test_data(n_samples, offset=0):
    x_training, y_training, x_test, y_test = preprocess()

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    class_normal = x_test[np.where(y_test>3)][offset:offset+int(n_samples*0.7)].tolist()
    label_normal = np.ones(int(n_samples*0.7))

    class_abnormal = x_test[np.where(y_test<4)][offset:offset+int(n_samples*0.3)].tolist()
    label_abnormal = np.ones(int(n_samples*0.3)) * (-1)

    filtered_x = class_normal + class_abnormal
    filtered_y = label_normal.tolist() + label_abnormal.tolist()

    return (filtered_x, filtered_y)


def sample_vqc_training_data(n_samples, offset=0):
    x_training, y_training, x_test, y_test = preprocess()
    sample_per_class = n_samples//6

    x_training = np.array(x_test)
    y_test = np.array(y_test)

    class1 = x_training[np.where(y_test==1)][:sample_per_class].tolist()
    label1 = np.ones(sample_per_class) * 1
    class2 = x_training[np.where(y_test==2)][:sample_per_class].tolist()
    label2 = np.ones(sample_per_class) * 2
    class3 = x_training[np.where(y_test==3)][:sample_per_class].tolist()
    label3 = np.ones(sample_per_class) * 3
    class4 = x_training[np.where(y_test==4)][:sample_per_class].tolist()
    label4 = np.ones(sample_per_class) * 4
    class5 = x_training[np.where(y_test==5)][:sample_per_class].tolist()
    label5 = np.ones(sample_per_class) * 5
    class7 = x_training[np.where(y_test==7)][:sample_per_class].tolist()
    label7 = np.ones(sample_per_class) * 7

    filtered_x = class1 + class2 + class3 + class4 + class5 + class7
    filtered_y = label1.tolist() + label2.tolist() + label3.tolist() + label4.tolist() + label5.tolist() + label7.tolist()

    return (filtered_x, filtered_y)

