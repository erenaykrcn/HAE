import os
dirname = os.path.dirname(__file__)

import torch
import torch.nn as nn
from torch.autograd import Variable

from HAE import HAE
from modules.preprocessing.preprocessing import preprocess, visualize


qc_index = 2
loss_value = 0.051
n_samples = 20


loss_func = nn.MSELoss()

path = f'./data/training_results/pqc{qc_index}/training_result_loss_{loss_value}.pt'
hae = HAE(qc_index=qc_index)
hae.load_state_dict(torch.load(os.path.join(dirname, path)))
hae.eval()

input_data = preprocess()[2][:n_samples]
input_data = Variable(torch.FloatTensor(input_data))
output_data = hae(input_data)


for i, data in enumerate(input_data):
	loss = torch.sqrt(loss_func(output_data[i], data))
	print(f"Loss for data point {i+1}: {loss.item()}")


visualize(data=input_data[5], loss_value=loss_value, data_index=4, qc_index=2, output=False)
visualize(data=output_data[5].cpu().detach(), loss_value=loss_value, data_index=4, qc_index=2, output=True)

