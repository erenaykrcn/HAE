import numpy as np

from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit.utils import QuantumInstance
from qiskit.circuit import ParameterVector
from qiskit.utils import QuantumInstance
from qiskit import Aer
from qiskit.circuit.library import ZZFeatureMap, TwoLocal

import os
dirname = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(dirname, './qcircuits'))
from circuit_map import circuit_map, N_PARAMS


def create_qnn(backend, qc_index=0, custom_qc={}):
	"""
		Given the qc_index or custom_qc params,
		it delivers a Quantum Neural Network from
		Qiskit's CircuitQNN Class.
	"""
	n_theta = 4
	if qc_index:
		n_theta = N_PARAMS[qc_index]
	elif custom_qc:
		n_theta = custom_qc["n_theta"]

	x = ParameterVector('x', 4)
	theta = ParameterVector('θ', n_theta)
        
	if custom_qc:
		# TODO: Placeholder QC
		circuit = QuantumCircuit(n_qubits)
	elif qc_index != 9:
		circuit = circuit_map[qc_index](x=x, theta=theta)
	elif qc_index == 9:
		encoder = ZZFeatureMap(feature_dimension=4, reps=2)
		ansatz = TwoLocal(4, ['ry', 'rz'], 'cz', reps=1)
		AD_HOC_CIRCUIT = encoder.compose(ansatz)
		x = encoder.ordered_parameters
		theta = ansatz.ordered_parameters
		circuit = AD_HOC_CIRCUIT
	else:
		raise ValueError("Either a qc index or a custom qc has to be given!")

	# Uses the ParamterShiftEstimatorGradient per default.
	qnn = CircuitQNN(
		circuit=circuit,
		input_params=x,
		weight_params=theta,
		input_gradients=True,
		quantum_instance=backend
    )
	
	return qnn
