from qiskit import QuantumCircuit


def pqc1(n_qubits, x, theta):
	qc1 = QuantumCircuit(n_qubits)

	qc1.rx(x[0], 0)
	qc1.rx(x[1], 1)
	qc1.rx(x[2], 2)
	qc1.rx(x[3], 3)
	qc1.h(0)
	qc1.h(1)
	qc1.h(2)
	qc1.h(3)
	qc1.cz(0,3)
	qc1.cz(0,1)
	qc1.cz(1,2)
	qc1.cz(2,3)
	qc1.rx(theta[0], 0)
	qc1.rx(theta[1], 1)
	qc1.rx(theta[2], 2)
	qc1.rx(theta[3], 3)
	return qc1
	
