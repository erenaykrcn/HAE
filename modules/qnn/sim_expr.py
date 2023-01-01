import os
dirname = os.path.dirname(__file__)

import numpy as np
import matplotlib.pyplot as plt


from qiskit.circuit import ParameterVector
from qiskit import Aer, execute, QuantumCircuit
from qiskit.quantum_info import state_fidelity

import sys
sys.path.append(os.path.join(dirname, '../../'))
from modules.qnn.qcircuits.circuit_map import circuit_map, N_PARAMS


qc_index = 6

x = ParameterVector('x', 4)
theta = ParameterVector('theta', N_PARAMS[qc_index])
qc = circuit_map[qc_index](x=x, theta=theta)
backend = Aer.get_backend('statevector_simulator')

sf_array = []
for i in range(1000):
	x1 = (np.random.rand(4) * 2) - 1 
	x2 = (np.random.rand(4) * 2) - 1 

	theta1 = ( np.random.rand(N_PARAMS[qc_index]) * np.pi * 2 ) - np.pi 
	theta2 = ( np.random.rand(N_PARAMS[qc_index]) * np.pi * 2 ) - np.pi

	qc_param1 = qc.assign_parameters({
			x: x1,
			theta: theta1,
		})
	qc_param2 = qc.assign_parameters({
			x: x2,
			theta: theta2,
		})

	job1 = execute(qc_param1, backend=backend, shots=100)
	job2 = execute(qc_param2, backend=backend, shots=100)

	job_result1 = job1.result()
	job_result2 = job2.result()

	sv1 = job_result1.get_statevector(qc_param1)
	sv2 = job_result2.get_statevector(qc_param2)

	sf = round(state_fidelity(sv1, sv2), 3)
	sf_array.append(sf)
haar_rand = np.random.rand(100000)


def KL(P,Q):
	""" Epsilon is used here to avoid conditional code for
	checking that neither P nor Q is equal to 0. """
	epsilon = 0.00001

	# You may want to instead make copies to avoid changing the np arrays.
	P = P+epsilon
	Q = Q+epsilon
	divergence = np.sum(P*np.log(P/Q))
	return divergence

n_haar, bins_haar, patches_haar = plt.hist(haar_rand, 100, density=True, color="b", label="Haar", alpha=0.7)
n_pqc, bins_pqc, patches_pqc = plt.hist(sf_array, 100, density=True, color="orange", label="PQC", alpha=0.7)

plt.xlabel('Fidelity')
plt.ylabel('Probability')
plt.title('Histogram of Fidelities of PQC' + str(qc_index))
plt.grid(True)
plt.xlim(0, 1)
plt.savefig(f"hist_pqc{qc_index}.png")

print(KL(n_pqc, n_haar)/100)
