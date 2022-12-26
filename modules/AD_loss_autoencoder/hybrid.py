import torch
import torch.nn as nn
from torch.autograd import Function
from qiskit.circuit import ParameterVector

from sklearn.ensemble import IsolationForest

import sys
sys.path.append(os.path.join(dirname, '../../'))
from modules.qnn.qcircuits.circuit_map import circuit_map, N_PARAMS
from modules.qnn.utils import PQC


class Hybrid(nn.Module):
	def __init__(self, backend, shots, shift, qc_index, isolation_forest_model):
		super(Hybrid, self).__init__()
		self.quantum_circuit = PQC(backend, shots, qc_index)
		n_theta = N_PARAMS[qc_index]
		self.theta = nn.Parameter((torch.rand(n_theta) * 2 * np.pi) - np.pi)
		self.shift = shift
		self.isolation_forest_model = isolation_forest_model

	def forward(self, x):
		return HybridFunction.apply(x, self.theta, self.quantum_circuit, self.shift, self.isolation_forest_model)


class HybridFunction(Function):
    """ Hybrid quantum - classical function definition """
	
	@staticmethod
	def forward(self, x, theta, quantum_circuit, shift):
		""" Forward pass computation """
		self.shift = shift
		self.quantum_circuit = quantum_circuit
		self.theta = theta

		input_list = x.cpu().detach()
		exp_values = []
		for input_value in input_list:
			exp = self.quantum_circuit.run(input_value, theta.cpu().detach())
			exp_values.append(exp)
		if_model = IsolationForest().fit(exp_values)
		self.if_model = if_model

		result = if_model.decision_function(exp_values)
		result = Variable(torch.FloatTensor(result))

		self.save_for_backward(x, result)
		return result

	@staticmethod
	def backward(self, grad_output):
		""" Backward pass computation """
		x, expectation_z = self.saved_tensors
		input_list = np.array(x.cpu().detach())

		shift_right = input_list + np.ones(input_list.shape) * self.shift
		shift_left = input_list - np.ones(input_list.shape) * self.shift

		gradients = []
		for i in range(len(input_list)):
			expectation_right = self.if_model.decision_function(self.quantum_circuit.run(shift_right[i], self.theta.cpu().detach()))
			expectation_left  = self.if_model.decision_function(self.quantum_circuit.run(shift_left[i], self.theta.cpu().detach()))

			gradient = torch.tensor([expectation_right]) - torch.tensor([expectation_left])
			gradients.append(gradient)
		gradients = np.array([gradients]).T
		return torch.tensor([gradients]).float() * grad_output.float()
