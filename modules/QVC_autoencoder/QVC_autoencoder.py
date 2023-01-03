import os
dirname = os.path.dirname(__file__)

from qiskit.algorithms.optimizers import SPSA
import torch
from torch.autograd import Variable
from sklearn.ensemble import IsolationForest
import numpy as np

import sys
sys.path.append(os.path.join(dirname, '../../'))
from modules.classical_autoencoder.classical_autoencoder import ClassicalAutoencoder
from modules.preprocessing.preprocessing import preprocess, sample_training_data, sample_vqc_training_data
from modules.QVC_autoencoder.utils import OptimizerLog, get_classification_probabilities


class QVCAutoencoder:
	def __init__(self, qc_index, max_iter=100, loss_value_classical=0.023, n_samples=150):
		path_cl = f'../../data/training_results/classical/training_result_loss_{loss_value_classical}.pt'
		self.cae = ClassicalAutoencoder()
		self.cae.load_state_dict(torch.load(os.path.join(dirname, path_cl)))
		self.cae.eval()

		fit_data = Variable(torch.FloatTensor(sample_training_data(1000)[0]))
		fit_data_latent = self.cae.get_latent_space_state(fit_data).tolist()
		self.if_model = IsolationForest().fit(fit_data_latent)

		self.max_iter = max_iter
		self.n_samples = n_samples
		self.log = OptimizerLog()
		self.optimizer = SPSA(maxiter=max_iter, callback=self.log.update)
		self.qc_index = qc_index


	def loss_function_multiclass(self, theta):
		cost = 0
		x, labels = sample_vqc_training_data(self.n_samples, True)
		x = self.cae.get_latent_space_state(Variable(torch.FloatTensor(x))).tolist()

		classifications = get_classification_probabilities(x,theta, self.qc_index)
		for i, classification in enumerate(classifications):
			p = classification.get(labels[i])
			cost += -np.log(p + 1e-10)
		cost /= len(x)
		return cost


	def loss_function_binary(self, theta):
		cost = 0
		x, labels = sample_vqc_training_data(self.n_samples, True)
		labels = np.array(labels)
		labels = np.where(labels>3, 1, 0)
		x = self.cae.get_latent_space_state(Variable(torch.FloatTensor(x))).tolist()

		classifications = get_classification_probabilities(x,theta, self.qc_index, True)
		for i, classification in enumerate(classifications):
			p = classification.get(labels[i])
			cost += -np.log(p + 1e-10)
		cost /= len(x)
		return cost


	def train(self, initial_point, job=None, is_binary=False):
		if is_binary:
			loss_func = self.loss_function_binary
		else:
			loss_func = self.loss_function_multiclass

		result = self.optimizer.minimize(loss_func, initial_point)
		opt_theta = result.x
		min_cost = result.fun

		# TODO: use job and save opt-theta, return data path
		print(f"optimal theta values: {opt_theta}; min_cost: {min_cost}")
		return opt_theta, min_cost


	def eval(self, theta, test_data, is_binary=False):
		x = self.cae.get_latent_space_state(Variable(torch.FloatTensor(test_data))).tolist()

		probs = get_classification_probabilities(x,theta, self.qc_index, is_binary)

		if is_binary:
			predictions = [-1 if p[0] >= p[1] else 1 for p in probs]
		else:
			predictions = [-1 if p[1]+p[2]+p[3] >= p[4]+p[5]+p[6]+p[7] else 1 for p in probs]
		
		return predictions
		
