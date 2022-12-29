import os
dirname = os.path.dirname(__file__)

import torch
import torch.nn as nn
from torch.autograd import Function, Variable
import numpy as np
from sklearn.ensemble import IsolationForest

from qiskit.circuit import ParameterVector

import sys
sys.path.append(os.path.join(dirname, '../../'))
from modules.qnn.qcircuits.circuit_map import circuit_map, N_PARAMS
from modules.qnn.utils import PQC
from modules.classical_autoencoder.classical_autoencoder import ClassicalAutoencoder
from modules.preprocessing.preprocessing import sample_training_data


class Hybrid(nn.Module):
	def __init__(self, backend, shots, shift, qc_index, if_model=None, fit_data=None):
		super(Hybrid, self).__init__()
		self.quantum_circuit = PQC(backend, shots, qc_index)
		n_theta = N_PARAMS[qc_index]
		self.theta = nn.Parameter((torch.rand(n_theta) * 2 * np.pi) - np.pi)
		self.shift = shift
		self.if_model = if_model
		self.fit_data = fit_data

	def forward(self, x):
		return HybridFunction.apply(x, self.theta, self.quantum_circuit, self.shift, self.if_model, self.fit_data)


class HybridFunction(Function):
	""" Hybrid quantum - classical function definition """

	@staticmethod
	def forward(ctx, x, theta, quantum_circuit, shift, if_model, fit_data):
		""" 
			Forward pass computation.
			If the IsolationForest Model is given during initialization 
			of Hybrid Module, it uses the pre-fitted IsolationForest Model.
			If not given, cretes a new IsolationForest Model and fit the exp values
			after PQC processing to get the anomaly scores. 
		"""
		ctx.shift = shift
		ctx.quantum_circuit = quantum_circuit

		exp = ctx.quantum_circuit.run(x.tolist(), theta.tolist())

		if if_model:
			ctx.if_model = if_model
		elif fit_data != None:
			fit_data_exp = []
			for data in fit_data:
				fit_data_exp.append(ctx.quantum_circuit.run(data, theta.tolist()))
			ctx.if_model = IsolationForest().fit(fit_data_exp)
		else:
			raise ValueError("Either an IsolationForest model or fit data to be used has to be provided")

		result = ctx.if_model.decision_function([exp])
		result = Variable(torch.FloatTensor(result))

		ctx.save_for_backward(x, theta)
		return result

	@staticmethod
	def backward(ctx, grad_output):
		"""
			Backward pass computation.
			For the computation of anomaly scores, it uses
			the IsolationForest Model that was saved during the
			forward pass.
		"""
		x, theta = ctx.saved_tensors
		theta = theta.tolist()
		input_list = x.tolist()

		theta_gradients = []
		for i in range(len(theta)):
			theta_right = theta.copy()
			theta_left = theta.copy()

			theta_right[i] = theta[i] + ctx.shift
			theta_left[i] = theta[i] - ctx.shift

			exp_right = ctx.quantum_circuit.run(x=input_list, theta=theta_right)
			exp_left = ctx.quantum_circuit.run(x=input_list, theta=theta_left)

			anomaly_right = ctx.if_model.decision_function([exp_right])[0]
			anomaly_left  = ctx.if_model.decision_function([exp_left])[0]

			#gradient = (anomaly_right - anomaly_left) / (2 * ctx.shift)
			gradient = (anomaly_right - anomaly_left) / 2
			theta_gradients.append(gradient)

		return None, torch.tensor(theta_gradients).float() * grad_output.float(), None, None, None, None, None
