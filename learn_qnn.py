from modules.qnn.qcircuits.circuit_map import circuit_map
from modules.qnn.utils import convert_prob_to_exp, convert_prob_to_exp_batch

from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.circuit import ParameterVector
from qiskit.utils import QuantumInstance
from qiskit import Aer
from qiskit.quantum_info import SparsePauliOp

import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np


theta = ParameterVector('θ', 4)
x = ParameterVector('x', 4)
qc = circuit_map[2](x=x, theta=theta)
loss_func = nn.MSELoss()


def interpret(x):
    return x % 4


# TODO: Training too slow!
qnn = CircuitQNN(
    circuit=qc,
    input_params=x,
    weight_params=theta,
    input_gradients=True,
    quantum_instance=QuantumInstance(
        backend=Aer.get_backend("aer_simulator")
    )
)


#optimizer = torch.optim.Adam(qnn.parameters(), lr=0.05, weight_decay=1e-5)

data_set = np.random.rand(600,4)
input_data = Variable(torch.FloatTensor(data_set))

x = qnn.forward(input_data=data_set, weights=[0.1, 0.2, 0.12, 0.8])
print(x)

#output_data = convert_prob_to_exp_batch(x)


#optimizer.zero_grad()
#loss = loss_func(output_data, input_data)
#loss.backward()

#optimizer.step()

#print(loss.data)

back = qnn.backward(input_data=data_set, weights=[0.1, 0.2, 0.12, 0.8])
print(back)