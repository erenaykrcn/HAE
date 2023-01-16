import torch
from torch.autograd import Variable
from matplotlib import pyplot as plt

from qiskit.circuit import ParameterVector
from qiskit import Aer, execute, QuantumCircuit
from qiskit.quantum_info import state_fidelity, DensityMatrix, partial_trace
from qiskit import transpile, assemble
from qiskit.circuit.library import ZZFeatureMap, TwoLocal, ZFeatureMap, NLocal, PauliFeatureMap, RealAmplitudes, EfficientSU2
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
	def __init__(self, qc_index=0, backend=Aer.get_backend("aer_simulator"), shots=1, custom_qc={}):
		if qc_index != 9 and qc_index:
			n_theta = N_PARAMS[qc_index]
			self.x = ParameterVector('x', 4)
			self.theta = ParameterVector('θ', n_theta)
			self._circuit = circuit_map[qc_index](x=self.x, theta=self.theta)
		elif qc_index==9:
			encoder = ZZFeatureMap(feature_dimension=4, reps=2)
			ansatz = TwoLocal(4, ['ry', 'rz'], 'cz', reps=1)
			AD_HOC_CIRCUIT = encoder.compose(ansatz)
			self.x = encoder.ordered_parameters
			self.theta = ansatz.ordered_parameters
			self._circuit = AD_HOC_CIRCUIT
		elif custom_qc:
			"""
				Custom QC dictionary have to specify the encoder and ansatz.
				
				Encoder choices (from least to most primitive; Usage of PauliFeatureMap is adviced):
					-> ZFeatureMap(reps=2)
					-> ZZFeatureMap(reps=2)
					-> PauliFeatureMap(reps=2, entanglement=("full", "linear", "circular", "reverse_linear", "pairwise", "sca") or 
						  	[(0, 1), (1, 2), (2, 0),...] or [entangler_map_layer_1, entangler_map_layer_2,...],
						paulis=['Z', 'ZZ'], alpha=2.0
					)
					-> NLocal(rotation_blocks=['rx', 'ry' ...],
						entanglement_blocks=['cx', 'cy'...],
						reps=1,
						skip_final_rotation_layer=False
						)
				
				Ansatz choices (from least to most primitive; Usage of TwoLocal is adviced):
					-> RealAmplitudes(entanglement=("full", "linear", "circular", "reverse_linear", "pairwise", "sca") or 
						  	[(0, 1), (1, 2), (2, 0),...]  or [entangler_map_layer_1, entangler_map_layer_2,...],
							reps=3, skip_unentangled_qubits=False, 
							skip_final_rotation_layer=False)
					-> EfficientSU2(su2_gates=['rx', 'y'],
									entanglement=("full", "linear", "circular", "reverse_linear", "pairwise", "sca") or 
						  				[(0, 1), (1, 2), (2, 0),...]  or [entangler_map_layer_1, entangler_map_layer_2,...],
									reps=3, skip_unentangled_qubits=False,
									skip_final_rotation_layer=False)
					-> TwoLocal(rotation_blocks=['rx', 'ry' ...],
						entanglement_blocks=['cx', 'cy'...],
						entanglement=("full", "linear", "circular", "reverse_linear", "pairwise", "sca") or 
						  	[(0, 1), (1, 2), (2, 0),...]  or [entangler_map_layer_1, entangler_map_layer_2,...],
						reps=3,
						skip_unentangled_qubits=False, skip_final_rotation_layer=False,
						)

			"""
			match custom_qc["encoder"]:
				case "ZFeatureMap":
					encoder = ZFeatureMap(feature_dimension=4, reps=int(custom_qc["encoder_params"]["reps"]) if "reps" in custom_qc["encoder_params"].keys() else 2)
				case "ZZFeatureMap":
					encoder = ZZFeatureMap(feature_dimension=4, reps=int(custom_qc["encoder_params"]["reps"]) if "reps" in custom_qc["encoder_params"].keys() else 2)
				case "PauliFeatureMap":
					encoder = PauliFeatureMap(feature_dimension=4, reps=int(custom_qc["encoder_params"]["reps"]) if "reps" in custom_qc["encoder_params"].keys() else 2,
						entanglement=custom_qc["encoder_params"]["entanglement"],
						paulis=custom_qc["encoder_params"]["paulis"], alpha=custom_qc["encoder_params"]["alpha"]
						)
				case "NLocal":
					encoder = NLocal(num_qubits=4, rotation_blocks=custom_qc["encoder_params"]["rotation_blocks"], entanglement_blocks=custom_qc["encoder_params"]["entanglement_blocks"],
						reps=int(custom_qc["encoder_params"]["reps"]) if "reps" in custom_qc["encoder_params"].keys() else 2, skip_final_rotation_layer=custom_qc["encoder_params"]["skip_final_rotation_layer"],
						entanglement=custom_qc["encoder_params"]["entanglement"])
					if encoder.num_parameters != 4:
						raise ValueError("Encoder layer has to have 4 hyperparameters!")


			match custom_qc["ansatz"]:
				case "RealAmplitudes":
					ansatz = RealAmplitudes(num_qubits=4, entanglement=custom_qc["ansatz_params"]["entanglement"], reps=int(custom_qc["ansatz_params"]["reps"]) if "reps" in custom_qc["ansatz_params"].keys() else 2, 
									skip_unentangled_qubits=custom_qc["ansatz_params"]["skip_unentangled_qubits"], 
									skip_final_rotation_layer=custom_qc["ansatz_params"]["skip_final_rotation_layer"])
				case "EfficientSU2":
					ansatz = EfficientSU2(num_qubits=4, su2_gates=custom_qc["ansatz_params"]["su2_gates"],
									entanglement=custom_qc["ansatz_params"]["entanglement"], reps=int(custom_qc["ansatz_params"]["reps"]) if "reps" in custom_qc["ansatz_params"].keys() else 2, 
									skip_unentangled_qubits=custom_qc["ansatz_params"]["skip_unentangled_qubits"], 
									skip_final_rotation_layer=custom_qc["ansatz_params"]["skip_final_rotation_layer"])
				case "TwoLocal":
					ansatz = TwoLocal(num_qubits=4, rotation_blocks=custom_qc["ansatz_params"]["rotation_blocks"], entanglement_blocks=custom_qc["ansatz_params"]["entanglement_blocks"],
						reps=int(custom_qc["ansatz_params"]["reps"]) if "reps" in custom_qc["ansatz_params"].keys() else 2, skip_final_rotation_layer=custom_qc["ansatz_params"]["skip_final_rotation_layer"],
						entanglement=custom_qc["ansatz_params"]["entanglement"]
						)
			N_PARAMS["custom"] = ansatz.num_parameters
			self.num_parameters = ansatz.num_parameters
			AD_HOC_CIRCUIT = encoder.compose(ansatz)
			self.x = encoder.ordered_parameters
			self.theta = ansatz.ordered_parameters
			self._circuit = AD_HOC_CIRCUIT
		else:
			raise ValueError("Either a qc index or custom_qc dict has to be given!")

		self.backend = backend
		self.shots = shots
		self.qc_index = qc_index
		self.custom_qc = custom_qc
	

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
		path = path if path else f"./qcircuits/circuit_images/pqc{qc_index if self.qc_index else '_custom'}.png"
		self._circuit.decompose().draw(output="mpl").savefig(os.path.join(dirname, path), dpi=300, transparent=True)
		plt.clf()
		
	def circuit(self):
		return self._circuit


def KL(P,Q):
	""" Epsilon is used here to avoid conditional code for
	checking that neither P nor Q is equal to 0. """
	epsilon = 0.00001

	# You may want to instead make copies to avoid changing the np arrays.
	P = P+epsilon
	Q = Q+epsilon
	divergence = np.sum(P*np.log(P/Q))
	return divergence


def sim_expr(qc_index=0, custom_qc={}, path_custom=""):
	backend = Aer.get_backend('statevector_simulator')
	qc = PQC(qc_index=qc_index, custom_qc=custom_qc, backend=backend, shots=1)

	sf_array = []
	for i in range(1000):
		x1 = (np.random.rand(4) * 2) - 1 
		x2 = (np.random.rand(4) * 2) - 1 

		theta1 = ( np.random.rand(N_PARAMS[qc_index if qc_index else "custom"]) * np.pi * 2 ) - np.pi 
		theta2 = ( np.random.rand(N_PARAMS[qc_index if qc_index else "custom"]) * np.pi * 2 ) - np.pi

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
	haar_rand = np.random.rand(10000)

	n_haar, bins_haar, patches_haar = plt.hist(haar_rand, 100, density=True, color="b", label="Haar", alpha=0.7)
	n_pqc, bins_pqc, patches_pqc = plt.hist(sf_array, 100, density=True, color="orange", label="PQC", alpha=0.7)

	plt.xlabel('Fidelity')
	plt.ylabel('Probability Density')
	plt.title('Histogram of Fidelities of PQC' + str(qc_index if qc_index else " Custom"))
	plt.grid(True)
	plt.xlim(0, 1)
	plt.legend()
	if path_custom:
		plt.savefig(path_custom, transparent=True)
	else:
		plt.savefig(f"sim_expr/hist/hist_pqc{qc_index if qc_index else '_custom'}.png", transparent=True)
	plt.clf()

	sim_expr = KL(n_pqc, n_haar)/100
	print(f"KL divergence of the fidelities: {sim_expr}")

	return sim_expr


def meyer_wallach_measure(qc_index=0, custom_qc={}):
	backend = Aer.get_backend('statevector_simulator')
	qc = PQC(qc_index=qc_index, custom_qc=custom_qc, backend=backend, shots=1)

	cap_sum = 0
	for i in range(1000):
		x = (np.random.rand(4) * 2) - 1
		theta = ( np.random.rand(N_PARAMS[qc_index if qc_index else "custom"]) * np.pi * 2 ) - np.pi

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
	f.write(f"MW-Measure of PQC{qc_index if qc_index else ' Custom'}: {np.around(cap_average, 3)}\n")
	f.close()

	return cap_average

