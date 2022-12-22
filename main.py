import os
dirname = os.path.dirname(__file__)

import torch
import torch.nn as nn
from torch.autograd import Variable

from HAE import HAE
from modules.preprocessing.visualize import visualize, visualize_test_data
from modules.preprocessing.preprocessing import preprocess
from modules.classical_autoencoder.classical_autoencoder import ClassicalAutoencoder


qc_index = 1
loss_value = 0.048
loss_value_classical = 0.023
n_samples = 20

offset=900
index=1


loss_func = nn.MSELoss()

path = f'./data/training_results/pqc{qc_index}/training_result_loss_{loss_value}.pt'
hae = HAE(qc_index=qc_index)
hae.load_state_dict(torch.load(os.path.join(dirname, path)))
hae.eval()

input_data = preprocess()[2][offset:offset+n_samples]
labels = preprocess()[3][offset:offset+n_samples]

input_data = Variable(torch.FloatTensor(input_data))
output_data = hae(input_data)


path_cl = f'./data/training_results/classical/training_result_loss_{loss_value_classical}.pt'
classical_ae = ClassicalAutoencoder()
classical_ae.load_state_dict(torch.load(os.path.join(dirname, path_cl)))
output_data_classical = classical_ae(input_data)


for i, data in enumerate(input_data):
	loss_h = torch.sqrt(loss_func(output_data[i], data))
	loss_cl = torch.sqrt(loss_func(output_data_classical[i], data))
	print(f"Label {labels[i]} Hybrid Loss: {loss_h.item()} ; Classical Loss: {loss_cl.item()}")


visualize(data=input_data[index], loss_value=loss_value, data_index=offset+index, qc_index=qc_index, output=False)
visualize(data=output_data.cpu().detach()[index], loss_value=loss_value, data_index=offset+index, qc_index=qc_index, output=True)
visualize(data=output_data_classical.cpu().detach()[index], loss_value=loss_value, data_index=str(offset+index) + "_classical", qc_index=qc_index, output=True)


input_data = [int(pix*255) for pix in input_data[index]]
output_data = [int(pix*255) for pix in output_data.cpu().detach()[index]]
output_data_classical = [int(pix*255) for pix in output_data_classical.cpu().detach()[index]]


print("Input Data")
print(input_data)
print("Output Data, Hybrid")
print(output_data)
print("Output Data, Classical")
print(output_data_classical)

print(f"Hybrid Loss: {torch.sqrt(loss_func(torch.FloatTensor(output_data), torch.FloatTensor(input_data)))}")
print(f"Classical Loss: {torch.sqrt(loss_func(torch.FloatTensor(output_data_classical), torch.FloatTensor(input_data)))}")
