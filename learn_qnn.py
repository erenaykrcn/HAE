import os
dirname = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(dirname, './modules/qnn/qcircuits'))
from circuit_map import circuit_map

from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.circuit import ParameterVector

theta = ParameterVector('θ', 4)
x = ParameterVector('x', 4)
qc = circuit_map[2](x=x, theta=theta)

def interpret(x):
    print(x)
    return x


qnn = SamplerQNN(
    circuit=qc,
    input_params=x,
    weight_params=theta,
    interpret=interpret,
    output_shape=16
)


probs = qnn.forward(input_data=[0.5, 0.3, 0.4, 0.3], weights=[0.3, 0.4, 0.4, 0.1])[0]
print(probs)

x0 = probs[1] + probs[3] + probs[5] + probs[7] + probs[9] + probs[11] + probs[13] + probs[15]
x0_not = probs[0] + probs[2] + probs[4] + probs[6] + probs[8] + probs[10] + probs[12] + probs[14]

x1 = probs[2] + probs[3] + probs[6] + probs[7] + probs[10] + probs[11] + probs[14] + probs[15]
x2 = probs[4] + probs[5] + probs[6] + probs[7] + probs[12] + probs[13] + probs[14] + probs[15]
x3 = probs[8] + probs[9] + probs[10] + probs[11] + probs[12] + probs[13] + probs[14] + probs[15]

print(f"x0: {x0}")
print(f"x1: {x1}")
print(f"x2: {x2}")
print(f"x3: {x3}")
print(x0 + x0_not)
