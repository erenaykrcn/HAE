from qiskit import QuantumCircuit


def pqc6(x, theta):
	qc = QuantumCircuit(4)
	qc.rx(x[0], 0)
	qc.rx(x[1], 1)
	qc.rx(x[2], 2)
	qc.rx(x[3], 3)

	qc.h([0, 1, 2, 3])

	qc.rz(theta[0], 0)
	qc.rz(theta[1], 1)
	qc.rz(theta[2], 2)
	qc.rz(theta[3], 3)

	qc.rx(theta[4], 0)
	qc.rx(theta[5], 1)
	qc.rx(theta[6], 2)
	qc.rx(theta[7], 3)
	return qc
	
