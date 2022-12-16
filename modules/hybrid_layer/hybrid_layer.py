import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np

import os
dirname = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(dirname, '../PQC'))
from PQC import PQC


class HybridFunction(Function):
	""" Hybrid quantum - classical function definition """
	@staticmethod
	def forward(ctx, x, quantum_circuit, shift):
		""" Forward pass computation """
		ctx.shift = shift
		ctx.quantum_circuit = quantum_circuit

		# TODO: x Params!
		expectation_z = ctx.quantum_circuit.run(x[0].tolist())
		result = torch.tensor([expectation_z])
		ctx.save_for_backward(input, result)
		return result

	@staticmethod
	def backward(ctx, grad_output):
		""" Backward pass computation """
		x, expectation_z = ctx.saved_tensors
		input_list = np.array(x.tolist())

		shift_right = input_list + np.ones(input_list.shape) * ctx.shift
		shift_left = input_list - np.ones(input_list.shape) * ctx.shift

		gradients = []
		for i in range(len(input_list)):
			expectation_right = ctx.quantum_circuit.run(shift_right[i])
			expectation_left  = ctx.quantum_circuit.run(shift_left[i])

			gradient = torch.tensor([expectation_right]) - torch.tensor([expectation_left])
			gradients.append(gradient)
		gradients = np.array([gradients]).T
		return torch.tensor([gradients]).float() * grad_output.float(), None, None


class HybridLayer(nn.Module):
    """ Hybrid quantum - classical layer definition """
    
    def __init__(self, backend, shots, shift, qc_index):
        super(HybridLayer, self).__init__()
        self.quantum_circuit = PQC(4, backend, shots, qc_index)
        self.shift = shift
        
    def forward(self, x):
        return HybridFunction.apply(x, self.quantum_circuit, self.shift)