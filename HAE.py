import os
dirname = os.path.dirname(__file__)

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np

from qiskit import Aer
from qiskit_machine_learning.connectors import TorchConnector

from modules.qnn.qnn import create_qnn
from modules.qnn.utils import convert_prob_to_exp_batch
from modules.preprocessing.preprocessing import preprocess, sample_training_data


class HAE(nn.Module):
	def __init__(self, qc_index=0, custom_qc={}, epochs=50, batchSize=32, learningRate=1e-3, n_samples=200):
		super(HAE, self).__init__()
		self.encoder = nn.Sequential(
									nn.Linear(36, 18),
									nn.Tanh(),
									nn.Linear(18, 9),
									nn.Tanh(),
									nn.Linear(9, 4),
									nn.Tanh())
		self.qnn = TorchConnector(create_qnn(Aer.get_backend('aer_simulator'), 250, qc_index, custom_qc))
		self.decoder = nn.Sequential(
									nn.Linear(4, 9),
									nn.Tanh(),
									nn.Linear(9, 18),
									nn.Tanh(),
									nn.Linear(18, 36),
									nn.Tanh())
		self.epochs = epochs
		self.batchSize = batchSize
		self.learningRate = learningRate
		self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learningRate, weight_decay=1e-5)
		self.loss_func = nn.MSELoss()
		self.qc_index = qc_index
		self.custom_qc = custom_qc
		self.n_samples = n_samples


	def forward(self, x):
		x = self.encoder(x)
		x = self.qnn(x)
		x = convert_prob_to_exp_batch(x)
		x = self.decoder(x)
		return x


	def trainHybrid(self):
		"""
			The model is trained based on a mean 
			square root reconstruction loss.
		"""
		min_loss = 1
		best_params = self.state_dict()

		data_set = sample_training_data(self.n_samples)[0]
		data_set = Variable(torch.FloatTensor(data_set))

		loss_list = []  # Store loss history
		self.train()
		print(f"Training Started. \n Data points in Data set: {len(data_set)} \n Epochs: {self.epochs}")

		data_set = DataLoader(data_set, batch_size=self.batchSize, shuffle=True)

		for epoch in range(self.epochs):
			total_loss = []
			for i, data in enumerate(data_set):
				self.optimizer.zero_grad(set_to_none=True)
				output = self(data)  # Forward pass
				loss = torch.sqrt(self.loss_func(output, data))  # Calculate loss
				loss.backward()  # Backward pass
				self.optimizer.step()  # Optimize weights
				total_loss.append(loss.item())
			average_loss = sum(total_loss) / len(total_loss)
			loss_list.append(average_loss)
			print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, self.epochs, average_loss))

			if min_loss > average_loss:
				print("New min loss found!")
				best_params = self.state_dict()
				min_loss = average_loss

		path = ''
		if self.qc_index:
			path = f'./data/training_results/pqc{self.qc_index}/training_result_loss_{round(min_loss, 3)}.pt'
		elif self.custom_qc:
			path = f'./data/training_results/custom_qc/training_result_loss_{round(min_loss, 3)}.pt'

		torch.save(best_params, os.path.join(dirname, path))
