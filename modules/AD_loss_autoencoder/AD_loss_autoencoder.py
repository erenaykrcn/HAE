import os
dirname = os.path.dirname(__file__)

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from qiskit import Aer

from hybrid import Hybrid

import sys
sys.path.append(os.path.join(dirname, '../../'))
from modules.classical_autoencoder.classical_autoencoder import ClassicalAutoencoder
from modules.preprocessing.preprocessing import preprocess


class ADLossAutoencoder(nn.Module):
	def __init__(self, qc_index, epochs=50, batchSize=256, learningRate=1e-3, n_samples=1024, loss_value_classical=0.023):
		super(ADLossAutoencoder, self).__init__()

		path_cl = f'../../data/training_results/classical/training_result_loss_{loss_value_classical}.pt'
		self.cae = ClassicalAutoencoder(epochs=epochs, batchSize=batchSize, learningRate=learningRate, n_samples=n_samples)
						.load_state_dict(torch.load(os.path.join(dirname, path_cl)))
		self.cae.eval()

		self.hybrid = Hybrid(backend=Aer.get_backend('aer_simulator'), qc_index=qc_index, shots=250, shift=np.pi / 2)

		self.epochs = epochs
		self.batchSize = batchSize
		self.learningRate = learningRate

		self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learningRate, weight_decay=1e-5)
		self.loss_func = nn.MSELoss()

		self.qc_index = qc_index
		self.n_samples = n_samples


	def forward(self, x):
		x = self.cae(x)
		x = self.hybrid(x)
		return x


	def get_latent_space_state(self, x):
		x = self.cae(x)
		return x


	def trainADLoss(self, epochs=None, n_samples=None):
		"""
			The model is trained based on a mean 
			square root loss.

			TODO

		data_set = Variable(torch.FloatTensor(sample_training_data(n_samples)[0]))

		loss_list = []  # Store loss history
		self.train()
		print(f"Training Started. \n Data points in Data set: {len(data_set)} \n Epochs: {epochs}")

		data_set = DataLoader(data_set, batch_size=self.batchSize, shuffle=True)

		for epoch in range(epochs):
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
			print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, average_loss))

			if min_loss > average_loss:
				print("New min loss found!")
				best_params = self.state_dict()
				min_loss = average_loss

		path = ''
		if self.qc_index:
			path = f'../../data/training_results/pqc{self.qc_index}/training_result_loss_{round(min_loss, 3)}.pt'
		elif self.custom_qc:
			path = f'../../data/training_results/custom_qc/training_result_loss_{round(min_loss, 3)}.pt'

		torch.save(best_params, os.path.join(dirname, path))
		"""

		epochs = epochs if epochs else self.epochs
		n_samples = n_samples if n_samples else self.n_samples
		min_loss = 100
		best_params = self.state_dict()


