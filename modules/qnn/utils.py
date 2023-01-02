import torch
from torch.autograd import Variable
from matplotlib import pyplot as plt

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


def draw_circuit(qc_index):
	pqc = PQC(qc_index)
	pqc.draw()


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
	def __init__(self, qc_index, backend=Aer.get_backend("aer_simulator"), shots=1):
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

		self.backend = backend
		self.shots = shots
		self.qc_index = qc_index
	

	def run(self, x, theta):
		self._circuit.measure_all()
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


	def draw(self, path=""):
		path = path if path else f"./qcircuits/circuit_images/pqc{self.qc_index}.png"
		self._circuit.decompose().draw(output="mpl").savefig(os.path.join(dirname, path), dpi=300, transparent=True)


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
		backend = Aer.get_backend('statevector_simulator')
		qc = PQC(qc_index=qc_index, backend=backend, shots=1)

	sf_array = []
	for i in range(1000):
		x1 = (np.random.rand(4) * 2) - 1 
		x2 = (np.random.rand(4) * 2) - 1 

		theta1 = ( np.random.rand(N_PARAMS[qc_index]) * np.pi * 2 ) - np.pi 
		theta2 = ( np.random.rand(N_PARAMS[qc_index]) * np.pi * 2 ) - np.pi

		qc_param1 = qc.assign_parameters(x=x1, theta=theta1)
		qc_param2 = qc.assign_parameters(x=x2, theta=theta2)

		job1 = execute(qc_param1, backend=backend, shots=1)
		job2 = execute(qc_param2, backend=backend, shots=1)

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
	plt.ylabel('Probability Density')
	plt.title('Histogram of Fidelities of PQC' + str(qc_index))
	plt.grid(True)
	plt.xlim(0, 1)
	plt.legend()
	plt.savefig(f"sim_expr/hist/hist_pqc{qc_index}.png", transparent=True)
	plt.clf()

	sim_expr = KL(n_pqc, n_haar)/100
	print(f"KL divergence of the fidelities: {sim_expr}")

	return sim_expr


def meyer_wallach_measure(qc_index, custom_qc={}):
	if not custom_qc:
		backend = Aer.get_backend('statevector_simulator')
		qc = PQC(qc_index=qc_index, backend=backend, shots=1)

	cap_sum = 0
	for i in range(1000):
		x = (np.random.rand(4) * 2) - 1
		theta = ( np.random.rand(N_PARAMS[qc_index]) * np.pi * 2 ) - np.pi

		dm = DensityMatrix(qc.assign_parameters(x=x, theta=theta))

		tr_sum = 0
		for k in range(4):
			partial_tr = np.array(partial_trace(dm, [k]))
			partial_tr_quad = np.dot(partial_tr, partial_tr)
			tr = np.trace(partial_tr_quad)
			tr_sum += tr
		cap = 2*(1-(tr_sum)/4)
		cap_sum += cap
	cap_average = cap_sum / 1000

	path = "./meyer-wallach/meyer-wallach.txt"
	f = open(os.path.join(dirname, path), 'a')
	f.write(f"MW-Measure of PQC{qc_index}: {np.around(cap_average, 3)}\n")
	f.close()

	return cap_average