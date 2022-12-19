import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from qiskit import Aer
from qiskit_machine_learning.connectors import TorchConnector

from modules.qnn.qnn import create_qnn
from modules.qnn.utils import convert_prob_to_exp
from modules.preprocessing.preprocessing import preprocess


class HAE(nn.Module):
	def __init__(self, qc_index=0, custom_qc={}, epochs=100, batchSize=128, learningRate=1e-3):
		super(HAE, self).__init__()
		self.encoder = nn.Sequential(nn.Linear(36, 18),
									nn.Tanh(),
									nn.Linear(18, 9),
									nn.Tanh(),
									nn.Linear(9, 4),
									nn.Tanh())
		self.qnn = TorchConnector(create_qnn(Aer.get_backend('aer_simulator'), 250, qc_index, custom_qc))
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
		self.loss_func = nn.MSELoss()


	def forward(self, x):
		x = self.encoder(x)
		x = self.qnn(x)
		x = convert_prob_to_exp(x)
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
		loss_list = []  # Store loss history

		for epoch in range(self.epochs):
			for data in data_set:
				data = Variable(torch.FloatTensor(data))
				
				# Predict
				output = self(data)
				# Loss
				loss = torch.sqrt(self.loss_func(output, data))

				# Backpropagation
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
			loss_list.append(loss.data)

			if min_loss > loss.item():
				print("New min loss found!")
				best_params = self.state_dict()
				min_loss = loss.item()
			print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, self.epochs, loss.data))

		torch.save(best_params, os.path.join(dirname, f'./data/training_results/training_result_loss_{round(min_loss, 3)}'))


hae = HAE(qc_index=2)
hae.train()