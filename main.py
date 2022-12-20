import os
dirname = os.path.dirname(__file__)

import torch
from HAE import HAE


qc_index = 2
loss_value = 

path = f'./data/training_results/pqc{qc_index}/training_result_loss_{loss_value}'

hae = HAE(qc_index=qc_index)
hae.load_state_dict(torch.load(os.path.join(dirname, path)))

