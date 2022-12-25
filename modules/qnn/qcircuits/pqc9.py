from qiskit import QuantumCircuit


def pqc9(x, theta):
	qc = QuantumCircuit(4)
	qc.rx(x[0], 0)
	qc.rx(x[1], 1)
	qc.rx(x[2], 2)
	qc.rx(x[3], 3)

	qc.crz(theta[0], 3, 0)
	qc.crz(theta[1], 0, 1)
	qc.crz(theta[2], 1, 2)
	qc.crz(theta[3], 2, 3)

	qc.rx(x[0], 0)
	qc.rx(x[1], 1)
	qc.rx(x[2], 2)
	qc.rx(x[3], 3)

	qc.crz(theta[4], 3, 0)
	qc.crz(theta[5], 0, 1)
	qc.crz(theta[6], 1, 2)
	qc.crz(theta[7], 2, 3)	
	return qc

	
