import os
dirname = os.path.dirname(__file__)

from qiskit import Aer, execute

import sys
sys.path.append(os.path.join(dirname, '../../'))
from modules.qnn.utils import PQC


def parity(bitstring):
	"""Returns 1 if parity of `bitstring` is even, otherwise 0."""
	hamming_weight = sum(int(k) for k in list(bitstring))
	return (hamming_weight+1) % 2


def assign_parameters(x_data, theta, qc_index):
	pqc = PQC(Aer.get_backend("aer_simulator"), 100, qc_index)
	return pqc.assign_parameters(x_data, theta)


def sum_to_probability(result, is_binary):
	"""Converts a dict of bitstrings and their counts,
	to parities and their counts"""
	shots = sum(result.values())
	probabilities = {0: 0, 1: 0} if is_binary else {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
	for bitstring, counts in result.items():
		label = parity(bitstring) if is_binary else get_label(bitstring)
		probabilities[label] += counts / shots
	return probabilities


def get_classification_probabilities(x_data, theta, qc_index, is_binary=False):
	circuits = [assign_parameters(x, theta, qc_index) for x in x_data]
	results = execute(circuits, Aer.get_backend("aer_simulator")).result()
	return [sum_to_probability(results.get_counts(c), is_binary) for c in circuits]


class OptimizerLog():
	def __init__(self):
		self.evaluations = []
		self.theta_values = []
		self.costs = []
	def update(self, evaluation, theta, cost, _stepsize, _accept):
		self.evaluations.append(evaluation)
		self.theta_values.append(theta)
		self.costs.append(cost)

		print("Evaluations: " + str(evaluation) + "|| Loss: " + str(cost))


def get_label(bitstring):
	match bitstring:
		case "0000":
			return 1
		case "0001":
			return 1
		case "0010":
			return 2
		case "0011":
			return 2
		case "0100":
			return 3
		case "0101":
			return 3
		case "0110":
			return 4
		case "0111":
			return 4
		case "1000":
			return 5
		case "1001":
			return 5
		case "1010":
			return 3
		case "1011":
			return 4
		case "1100":
			return 7
		case "1101":
			return 7
		case "1110":
			return 3
		case "1111":
			return 4
