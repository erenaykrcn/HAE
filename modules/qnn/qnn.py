import numpy as np

from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.utils import QuantumInstance
from qiskit.circuit import ParameterVector

import os
dirname = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(dirname, './qcircuits'))
from circuit_map import circuit_map


def create_qnn(backend, shots, qc_index=0, custom_qc={}):
	"""
		Given the qc_index or custom_qc params,
		it delivers a Quantum Neural Network from
		Qiskit's EstimatorQNN Class.
	"""

	# TODO: Number of params should depend on the QC index
	theta = ParameterVector('θ', 4)
	x = ParameterVector('x', 4)
        
	if custom_qc:
		# TODO: Placeholder QC
		circuit = QuantumCircuit(n_qubits)
	elif qc_index:
		circuit = circuit_map[qc_index](x=x, theta=theta)
	else:
		raise ValueError("Either a qc index or a custom qc has to be given!")

	# Uses the ParamterShiftEstimatorGradient per default.
	qnn = SamplerQNN(
		circuit=circuit,
		input_params=x,
		weight_params=theta,
		input_gradients=True,
    )
	
	return qnn
