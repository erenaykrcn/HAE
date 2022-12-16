from qiskit import QuantumCircuit, transpile, assemble
from qiskit.circuit import ParameterVector
from circuits.circuit_map import circuit_map


class PQC():
	""" 
    This class implements a simple interface for 
    interaction with the parametrized quantum circuit. 
    """

    def __init__(self, n_qubits, backend, shots, qc_index=0, custom_qc={}):

        if custom_qc:
            # TODO: Placeholder QC
    	   self._circuit = QuantumCircuit(n_qubits)
        elif qc_index:
            self._circuit = circuit_map[qc_index](n_qubits, x, self.theta)
        else:
            raise ValueError("Either a qc index or a custom qc has to be given!")

        self.theta = ParameterVector('θ', 4)
        self._circuit.measure_all()
        self.backend = backend
        self.shots = shots


    def run(self, thetas):
        t_qc = transpile(self._circuit, self.backend)
        qobj = assemble(t_qc, shots=self.shots,
                        parameter_binds=[{self.theta: theta} for theta in thetas]
            )
        job = self.backend.run(qobj)
        result = job.result().get_counts()

        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
        
        # Compute probabilities for each state
        probabilities = counts / self.shots
        # Get state expectation
        expectation = np.sum(states * probabilities)
        
        return np.array([expectation])


    	