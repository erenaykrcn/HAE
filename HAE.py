import torch
import torch.nn as nn
import numpy as np

from qiskit import Aer

from modules.PQC import PQC
from modules.hybrid_layer.hybrid_layer import HybridLayer
from modules.preprocessing.preprocessing import preprocess


class HAE(nn.Module):
	def __init__(self, qc_index=0, epochs=100, batchSize=128, learningRate=1e-3):
		super(HAE, self).__init__()
		self.encoder = nn.Sequential(nn.Linear(36, 18),
									nn.Tanh(),
									nn.Linear(18, 9),
									nn.Tanh(),
									nn.Linear(9, 4),
									nn.Tanh())
		self.hybrid_layer = HybridLayer(Aer.get_backend('aer_simulator'), 100, np.pi / 2, qc_index)
		self.decoder = nn.Sequential(nn.Linear(4, 9),
									nn.Tanh(),
									nn.Linear(9, 18),
									nn.Tanh(),
									nn.Linear(18, 36),
									nn.Tanh())

		self.epochs = epochs
		self.batchSize = batchSize
		self.learningRate = learningRate
		self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learningRate, weight_decay=1e-5)
		self.criterion = nn.MSELoss()


	def forward(self, x):
		x = self.encoder(x)
		x = self.hybrid_layer(x)
		x = self.decoder(x)
		return x


	def train(self):
		"""
			The model is trained based on a mean 
			square root reconstruction loss.
		"""
		min_loss = 1
		best_params = self.state_dict()
		data_set = preprocess()

		for epoch in range(self.epochs):
			for data in data_set:
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

		torch.save(best_params, os.path.join(dirname, f'./data/training_results/training_result_loss_{round(min_loss, 3)}'))


hae = HAE(qc_index=1)
hae.train()