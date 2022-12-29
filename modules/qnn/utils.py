import torch
from torch.autograd import Variable
from qiskit.circuit import ParameterVector
from qiskit import transpile, assemble
from qiskit.circuit.library import ZZFeatureMap, TwoLocal
import numpy as np

import os
dirname = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(dirname, './qcircuits'))
from circuit_map import circuit_map, N_PARAMS


def convert_prob_to_exp_batch(probs):
	"""
		Helper tool to get the expectation values
		of the single qubits from prob dict.
	"""
	x0 = probs[:,1] + probs[:,3] + probs[:,5] + probs[:,7] + probs[:,9] + probs[:,11] + probs[:,13] + probs[:,15]
	x1 = probs[:,2] + probs[:,3] + probs[:,6] + probs[:,7] + probs[:,10] + probs[:,11] + probs[:,14] + probs[:,15]
	x2 = probs[:,4] + probs[:,5] + probs[:,6] + probs[:,7] + probs[:,12] + probs[:,13] + probs[:,14] + probs[:,15]
	x3 = probs[:,8] + probs[:,9] + probs[:,10] + probs[:,11] + probs[:,12] + probs[:,13] + probs[:,14] + probs[:,15]

	x = torch.stack((x0, x1, x2, x3), -1)
	return x


def convert_prob_to_exp(probs):
	"""
		Helper tool to get the expectation values
		of the single qubits from prob dict.
	"""
	x0 = probs[1] + probs[3] + probs[5] + probs[7] + probs[9] + probs[11] + probs[13] + probs[15]
	x1 = probs[2] + probs[3] + probs[6] + probs[7] + probs[10] + probs[11] + probs[14] + probs[15]
	x2 = probs[4] + probs[5] + probs[6] + probs[7] + probs[12] + probs[13] + probs[14] + probs[15]
	x3 = probs[8] + probs[9] + probs[10] + probs[11] + probs[12] + probs[13] + probs[14] + probs[15]

	x = torch.stack((x0, x1, x2, x3), -1)
	return x


class PQC:
	""" 
	This class provides a simple interface for interaction 
	with the quantum circuit 
	"""
	def __init__(self, backend, shots, qc_index):
		if qc_index != 9:
			n_theta = N_PARAMS[qc_index]
			self.x = ParameterVector('x', 4)
			self.theta = ParameterVector('θ', n_theta)

			self._circuit = circuit_map[qc_index](x=self.x, theta=self.theta)
		else:
			encoder = ZZFeatureMap(feature_dimension=4, reps=2)
			ansatz = TwoLocal(4, ['ry', 'rz'], 'cz', reps=1)
			AD_HOC_CIRCUIT = encoder.compose(ansatz)
			self.x = encoder.ordered_parameters
			self.theta = ansatz.ordered_parameters
			self._circuit = AD_HOC_CIRCUIT

		self._circuit.measure_all()
		self.backend = backend
		self.shots = shots
	

	def run(self, x, theta):
		t_qc = transpile(self._circuit,
						 self.backend)
		params = [{self.theta: theta, self.x: x}]
		qobj = assemble(t_qc, shots=self.shots, parameter_binds = params)
		job = self.backend.run(qobj)

		result = job.result().get_counts()
		counts = np.zeros(16)

		for key in result.keys():
			index = int(key, 2) 
			counts[index] = result[key]

		# Compute probabilities for each state
		probabilities = counts / self.shots

		expectations = convert_prob_to_exp(Variable(torch.FloatTensor(probabilities))).tolist()
		return expectations


	def assign_parameters(self, x, theta):
		parameters = {}
		for i, p in enumerate(self.x):
			parameters[p] = x[i]
		for i, p in enumerate(self.theta):
			parameters[p] = theta[i]
		return self._circuit.assign_parameters(parameters)
