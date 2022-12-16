import numpy as np
import torchvision
from torch.autograd import Variable
from torchvision.utils import save_image
import matplotlib.pyplot as plt


from torch import nn
import torch

import os
dirname = os.path.dirname(__file__)

import sys
sys.path.append(os.path.join(dirname, '../preprocessing'))
from preprocessing import preprocess


class ClassicalAutoencoder(nn.Module):
	def __init__(self, epochs=100, batchSize=128, learningRate=1e-3):
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
		self.criterion = nn.MSELoss()

	def forward(self, x):
		x = self.encoder(x)
		x = self.decoder(x)
		return x

	def trainClassical(self):
		"""
			The model is trained based on a mean 
			square root reconstruction loss.
		"""
		min_loss = 1
		best_params = self.state_dict()

		for epoch in range(self.epochs):
			for data in self.data:
				data = Variable(torch.FloatTensor(data))

				# Predict
				output = self(data)
				# Loss
				loss = torch.sqrt(self.criterion(output, data))

				# Backpropagation
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

			if min_loss > loss.item():
				print("New min loss found!")
				best_params = self.state_dict()
				min_loss = loss.item()
			print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, self.epochs, loss.data))

		torch.save(best_params, os.path.join(dirname, f'../../data/training_results/training_result_loss_{round(min_loss, 3)}'))

