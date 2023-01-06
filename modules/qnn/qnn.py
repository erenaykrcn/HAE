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
		match custom_qc["encoder"]:
			case "ZFeatureMap":
				encoder = ZFeatureMap(feature_dimension=4, reps=int(custom_qc["encoder_params"]["reps"]) if "reps" in custom_qc["encoder_params"].keys() else 2)
			case "ZZFeatureMap":
				encoder = ZZFeatureMap(feature_dimension=4, reps=int(custom_qc["encoder_params"]["reps"]) if "reps" in custom_qc["encoder_params"].keys() else 2)
			case "PauliFeatureMap":
				encoder = PauliFeatureMap(feature_dimension=4, reps=int(custom_qc["encoder_params"]["reps"]) if "reps" in custom_qc["encoder_params"].keys() else 2,
						entanglement=custom_qc["encoder_params"]["entanglement"], skip_final_rotation_layer=custom_qc["encoder_params"]["skip_final_rotation_layer"],
						paulis=custom_qc["encoder_params"]["paulis"], alpha=custom_qc["encoder_params"]["alpha"]
						)
			case "NLocal":
				encoder = NLocal(num_qubits=4, rotation_blocks=custom_qc["encoder_params"]["rotation_blocks"], entanglement_blocks=qustom_qc["encoder_params"]["entanglement_blocks"],
						reps=int(custom_qc["encoder_params"]["reps"]) if "reps" in custom_qc["encoder_params"].keys() else 2, skip_final_rotation_layer=custom_qc["encoder_params"]["skip_final_rotation_layer"],
						)
				if encoder.num_parameters != 4:
					raise ValueError("Encoder layer has to have 4 hyperparameters!")


		match custom_qc["ansatz"]:
			case "RealAmplitudes":
				ansatz = RealAmplitudes(entanglement=custom_qc["ansatz_params"]["entanglement"], reps=int(custom_qc["ansatz_params"]["reps"]) if "reps" in custom_qc["ansatz_params"].keys() else 2, 
									skip_unentangled_qubits=qustom_qc["ansatz_params"]["skip_unentangled_qubits"], 
									skip_final_rotation_layer=qustom_qc["ansatz_params"]["skip_final_rotation_layer"])
			case "EfficientSU2":
				ansatz = EfficientSU2(su2_gates=custom_qc["ansatz_params"]["su2_gates"],
									entanglement=custom_qc["ansatz_params"]["entanglement"], reps=int(custom_qc["ansatz_params"]["reps"]) if "reps" in custom_qc["ansatz_params"].keys() else 2, 
									skip_unentangled_qubits=qustom_qc["ansatz_params"]["skip_unentangled_qubits"], 
									skip_final_rotation_layer=qustom_qc["ansatz_params"]["skip_final_rotation_layer"])
			case "TwoLocal":
				ansatz = TwoLocal(num_qubits=4, rotation_blocks=custom_qc["ansatz_params"]["rotation_blocks"], entanglement_blocks=qustom_qc["ansatz_params"]["entanglement_blocks"],
						reps=int(custom_qc["ansatz_params"]["reps"]) if "reps" in custom_qc["ansatz_params"].keys() else 2, skip_final_rotation_layer=custom_qc["ansatz_params"]["skip_final_rotation_layer"],
					)

		AD_HOC_CIRCUIT = encoder.compose(ansatz)
		x = encoder.ordered_parameters
		theta = ansatz.ordered_parameters
		circuit = AD_HOC_CIRCUIT
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
