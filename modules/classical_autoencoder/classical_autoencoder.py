import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


from torch import nn
import torch

import os
dirname = os.path.dirname(__file__)

import sys
sys.path.append(os.path.join(dirname, '../preprocessing'))
from preprocessing import preprocess, sample_training_data


class ClassicalAutoencoder(nn.Module):
	def __init__(self, epochs=30, batchSize=32, learningRate=1e-3, n_samples=600):
		super(ClassicalAutoencoder, self).__init__()
		# Encoder Network
		self.encoder = nn.Sequential(nn.Linear(36, 18),
									nn.Tanh(),
									nn.Linear(18, 9),
									nn.Tanh(),
									nn.Linear(9, 4),
									nn.Tanh())
        # Decoder Network
		self.decoder = nn.Sequential(nn.Linear(4, 9),
									nn.Tanh(),
									nn.Linear(9, 18),
									nn.Tanh(),
									nn.Linear(18, 36),
									nn.Tanh())

		self.epochs = epochs
		self.batchSize = batchSize
		self.learningRate = learningRate
		self.data = preprocess()[0]

		self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learningRate, weight_decay=1e-5)
		self.loss_func = nn.MSELoss()
		self.n_samples = n_samples


	def forward(self, x):
		x = self.encoder(x)
		x = self.decoder(x)
		return x

	def get_latent_space_state(self, x):
		x = self.encoder(x)
		return x

	def trainClassical(self):
		"""
			The model is trained based on a mean 
			square root reconstruction loss.
		"""
		min_loss = 1
		best_params = self.state_dict()

		data_set = Variable(torch.FloatTensor(sample_training_data(self.n_samples)[0]))

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

		torch.save(best_params, os.path.join(dirname, f'../../data/training_results/classical/training_result_loss_{round(min_loss, 3)}.pt'))

