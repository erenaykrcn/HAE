import os
dirname = os.path.dirname(__file__)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np


import sys
sys.path.append(os.path.join(dirname, '../../'))
from modules.classical_autoencoder.classical_autoencoder import ClassicalAutoencoder
from modules.preprocessing.preprocessing import sample_vqc_training_data


class BinaryClassification(nn.Module):
	def __init__(self, learning_rate=0.005, epochs=500, n_samples=200, loss_value_classical=0.023):
		super(BinaryClassification, self).__init__()
		# Number of input features is 4.
		self.layer = nn.Sequential( nn.Linear(4,3),
									nn.ReLU(),
									nn.Linear(3, 1),
									)

		self.criterion = nn.BCEWithLogitsLoss()
		self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
		self.epochs = epochs

		self.n_samples = n_samples
		path_cl = f'../../data/training_results/classical/training_result_loss_{loss_value_classical}.pt'
		self.cae = ClassicalAutoencoder()
		self.cae.load_state_dict(torch.load(os.path.join(dirname, path_cl)))
		self.cae.eval()

	def forward(self, inputs):
		x = self.layer(inputs)
		return x

	def trainBinary(self, epochs=0, n_samples=0):
		n_samples = n_samples if n_samples else self.n_samples
		epochs = epochs if epochs else self.epochs

		x, labels = sample_vqc_training_data(n_samples, True)
		labels = np.array(labels)
		labels = np.where(labels>3, 1, 0)
		x = self.cae.get_latent_space_state(Variable(torch.FloatTensor(x)))

		min_loss = 1
		best_params = self.state_dict()
		loss_list = []  # Store loss history
		self.train()
		print(f"Training Started. \n Data points in Data set: {len(x)} \n Epochs: {epochs}")

		for epoch in range(epochs):
			total_loss = []
			for i, data in enumerate(x):
				self.optimizer.zero_grad(set_to_none=True)
				output = self(data)  # Forward pass
				loss = self.criterion(output, torch.FloatTensor([labels[i]]))  # Calculate loss
				loss.backward(retain_graph=True)  # Backward pass
				self.optimizer.step()  # Optimize weights
				total_loss.append(loss.item())
			average_loss = sum(total_loss) / len(total_loss)
			loss_list.append(average_loss)
			print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, average_loss))

			if min_loss > average_loss:
				print("New min loss found!")
				best_params = self.state_dict()
				min_loss = average_loss
		torch.save(best_params, os.path.join(dirname, f'../../data/training_results/classical_binary_cl/training_result_loss_{round(min_loss, 3)}.pt'))
