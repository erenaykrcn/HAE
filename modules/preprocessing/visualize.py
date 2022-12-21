import os
import numpy as np
from matplotlib import pyplot as plt

dirname = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(dirname, './'))
from preprocessing import preprocess, sample_training_data


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


def visualize(data, loss_value, data_index, qc_index=0, custom_qc={}, output=False):
    array_to_visualize = [[[],[],[]],
                          [[],[],[]],
                          [[],[],[]]]

    index = 0
    for m in range(3):
        for n in range(3):
            for i in range(4):
                array_to_visualize[m][n].append(data[index*4 + i])
            index += 1

    fig = plt.figure()
    plt.imshow(array_to_visualize)

    relative_file_path = f"../../data/visualize_constr_data/pqc{qc_index if qc_index else '_custom'}/loss_{loss_value}/{data_index}_{'reconstr' if output else 'original'}.png"
    fig.savefig(os.path.join(dirname, relative_file_path))