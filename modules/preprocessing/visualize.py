import os
import numpy as np
from matplotlib import pyplot as plt
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA

dirname = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(dirname, '../../'))
from modules.preprocessing.preprocessing import preprocess, sample_training_data, sample_test_data
from modules.classical_autoencoder.classical_autoencoder import ClassicalAutoencoder

import torch
from torch.autograd import Variable



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


def visualize(data, loss_value="test", data_index="test", qc_index=0, custom_qc={}, output=False):
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


def project_to_2D(x):
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    return principalComponents


def plot_PCA_2D(n_samples=200, test_data=None, test_labels=None, path_save=""):
    path = f'../../data/training_results/classical/training_result_loss_0.022.pt'
    cae = ClassicalAutoencoder()
    cae.load_state_dict(torch.load(os.path.join(dirname, path)))
    cae.eval()

    if not test_data or not test_labels:
        test_data, test_labels = sample_test_data(n_samples, True)
    test_data_latent = cae.get_latent_space_state(Variable(torch.FloatTensor(test_data))).tolist()
    test_data = project_to_2D(test_data_latent)

    x_values = []
    y_values = []
    for i, data in enumerate(test_data):
        x_values.append(data[0])
        y_values.append(data[1])
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    test_labels = np.array(test_labels)

    df = pd.DataFrame(dict(
            Principal_Component_1 = x_values,
            Principal_Component_2 = y_values,
            color= np.where(test_labels==1.0, "Normal Data", "Anomaly")
        ))
    fig = px.scatter(df, x='Principal_Component_1', y='Principal_Component_2', color='color',
                    title="PCA Projection of the Test Data"
        )

    if path_save:
        fig.write_image(path_save)
    else:
        fig.write_image(f"2D-plot-n_samples-{n_samples}.png")
    return fig
