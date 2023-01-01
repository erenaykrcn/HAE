import torch
from torch.autograd import Variable
from qiskit.circuit import ParameterVector
from qiskit import Aer, execute, QuantumCircuit
from qiskit.quantum_info import state_fidelity, DensityMatrix, partial_trace
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


def KL(P,Q):
	""" Epsilon is used here to avoid conditional code for
	checking that neither P nor Q is equal to 0. """
	epsilon = 0.00001

	# You may want to instead make copies to avoid changing the np arrays.
	P = P+epsilon
	Q = Q+epsilon
	divergence = np.sum(P*np.log(P/Q))
	return divergence


def sim_expr(qc_index, custom_qc={}):
	if not custom_qc:
		x = ParameterVector('x', 4)
		theta = ParameterVector('theta', N_PARAMS[qc_index])
		qc = circuit_map[qc_index](x=x, theta=theta)
		backend = Aer.get_backend('statevector_simulator')

	sf_array = []
	for i in range(1000):
		x1 = (np.random.rand(4) * 2) - 1 
		x2 = (np.random.rand(4) * 2) - 1 

		theta1 = ( np.random.rand(N_PARAMS[qc_index]) * np.pi * 2 ) - np.pi 
		theta2 = ( np.random.rand(N_PARAMS[qc_index]) * np.pi * 2 ) - np.pi

		qc_param1 = qc.assign_parameters({
				x: x1,
				theta: theta1,
			})
		qc_param2 = qc.assign_parameters({
				x: x2,
				theta: theta2,
			})

		job1 = execute(qc_param1, backend=backend, shots=100)
		job2 = execute(qc_param2, backend=backend, shots=100)

		job_result1 = job1.result()
		job_result2 = job2.result()

		sv1 = job_result1.get_statevector(qc_param1)
		sv2 = job_result2.get_statevector(qc_param2)

		sf = round(state_fidelity(sv1, sv2), 3)
		sf_array.append(sf)
	haar_rand = np.random.rand(100000)

	n_haar, bins_haar, patches_haar = plt.hist(haar_rand, 100, density=True, color="b", label="Haar", alpha=0.7)
	n_pqc, bins_pqc, patches_pqc = plt.hist(sf_array, 100, density=True, color="orange", label="PQC", alpha=0.7)

	plt.xlabel('Fidelity')
	plt.ylabel('Probability')
	plt.title('Histogram of Fidelities of PQC' + str(qc_index))
	plt.grid(True)
	plt.xlim(0, 1)
	plt.savefig(f"hist_pqc{qc_index}.png")

	return KL(n_pqc, n_haar)/100


def meyer_wallach_measure(qc_index, custom_qc={}):
	"""
		TODO: iterate over the x and theta values to average
		iterate over k = 0,1,2,3 
	"""
	if not custom_qc:
		x = ParameterVector('x', 4)
		theta = ParameterVector('theta', N_PARAMS[qc_index])
		qc = circuit_map[qc_index](x=x, theta=theta)
		backend = Aer.get_backend('statevector_simulator')

	dm = DensityMatrix(qc.assign_parameters({
		x: [1,1,1,1],
		theta: [1,1,1,1],
		}))

	partial_tr = partial_trace(dm, [0])

