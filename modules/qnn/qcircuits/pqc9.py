from qiskit.circuit.library import ZZFeatureMap, TwoLocal
from qiskit.circuit import ParameterVector


def pqc9(x, theta):
	encoder = ZZFeatureMap(feature_dimension=4, reps=2)
	ansatz = TwoLocal(4, ['ry', 'rz'], 'cz', reps=1)
	AD_HOC_CIRCUIT = encoder.compose(ansatz)
	return AD_HOC_CIRCUIT

