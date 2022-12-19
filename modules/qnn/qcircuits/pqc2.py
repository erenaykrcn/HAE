from qiskit import QuantumCircuit


def pqc2(x, theta):
	qc = QuantumCircuit(4)
	qc.rx(x[0], 0)
	qc.rx(x[1], 1)
	qc.rx(x[2], 2)
	qc.rx(x[3], 3)
	qc.rz(theta[0], 0)
	qc.rz(theta[1], 1)
	qc.rz(theta[2], 2)
	qc.rz(theta[3], 3)
	return qc
	
