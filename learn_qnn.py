#import os
#dirname = os.path.dirname(__file__)
#import sys
#sys.path.append(os.path.join(dirname, './modules/qnn'))
from modules.qnn.qcircuits.circuit_map import circuit_map
from modules.qnn.utils import convert_prob_to_exp

from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.circuit import ParameterVector

import torch
import torch.nn as nn
from torch.autograd import Variable


theta = ParameterVector('θ', 4)
x = ParameterVector('x', 4)
qc = circuit_map[2](x=x, theta=theta)
loss_func = nn.MSELoss()


def interpret(x):
    return x


qnn = TorchConnector(SamplerQNN(
    circuit=qc,
    input_params=x,
    weight_params=theta,
    interpret=interpret,
    input_gradients=True,
    output_shape=16
))
optimizer = torch.optim.Adam(qnn.parameters(), lr=0.05, weight_decay=1e-5)

data = [[0.5, 0.3, 0.4, 0.3], [0.6, 0.46, 0.12, 0.15]]
input_data = Variable(torch.FloatTensor(data))
x = qnn.forward(input_data=input_data)

output_data = convert_prob_to_exp(x)

optimizer.zero_grad()
loss = loss_func(output_data, input_data)
loss.backward()
optimizer.step()

print(loss.data)