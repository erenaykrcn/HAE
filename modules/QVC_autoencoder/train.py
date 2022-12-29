import os
dirname = os.path.dirname(__file__)

from QVC_autoencoder import QVCAutoencoder
import numpy as np

import sys
sys.path.append(os.path.join(dirname, '../../'))
from modules.qnn.qcircuits.circuit_map import N_PARAMS


qc_index = 9
is_binary = True
ip = np.random.random(N_PARAMS[qc_index])

model = QVCAutoencoder(qc_index=qc_index)
opt_theta, min_cost = model.train(ip, is_binary)


if is_binary:
	path = f"../../data/training_results_QVC/{'zzfeaturemap_twolocal' if qc_index == 9 else 'pqc' + str(qc_index)}/binary_cl/loss_{round(min_cost, 3)}.txt"
else:
	path = f"../../data/training_results_QVC/{'zzfeaturemap_twolocal' if qc_index == 9 else 'pqc' + str(qc_index)}/loss_{round(min_cost, 3)}.txt"
f = open(os.path.join(dirname, path), 'w+')
for theta in opt_theta:
	f.write(str(theta)+"\n")
f.close()

