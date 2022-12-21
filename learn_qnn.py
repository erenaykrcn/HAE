from modules.qnn.qcircuits.circuit_map import circuit_map
from modules.qnn.utils import convert_prob_to_exp, convert_prob_to_exp_batch

from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN, CircuitQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.circuit import ParameterVector
from qiskit.utils import QuantumInstance
from qiskit import Aer

import torch
import torch.nn as nn
from torch.autograd import Variable


theta = ParameterVector('θ', 4)
x = ParameterVector('x', 4)
qc = circuit_map[2](x=x, theta=theta)
loss_func = nn.MSELoss()


def interpret(x):
    return x

# TODO: Training too slow!
qnn = TorchConnector(CircuitQNN(
    circuit=qc,
    input_params=x,
    weight_params=theta,
    input_gradients=True,
    quantum_instance=QuantumInstance(
        backend=Aer.get_backend("aer_simulator_statevector")
    )
))


optimizer = torch.optim.Adam(qnn.parameters(), lr=0.05, weight_decay=1e-5)

data = [[0.5, 0.3, 0.4, 0.3], [0.5, 0.3, 0.01, 0.3]]
input_data = Variable(torch.FloatTensor(data))
x = qnn.forward(input_data=input_data)

output_data = convert_prob_to_exp_batch(x)
print(output_data)


optimizer.zero_grad()
loss = loss_func(output_data, input_data)
loss.backward()

optimizer.step()

print(loss.data)