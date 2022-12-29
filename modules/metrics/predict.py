import os
dirname = os.path.dirname(__file__)

import torch
from torch.autograd import Variable
import numpy as np

from sklearn.ensemble import IsolationForest

import sys
sys.path.append(os.path.join(dirname, '..\\..\\'))
from modules.HAE.HAE import HAE
from modules.classical_autoencoder.classical_autoencoder import ClassicalAutoencoder
from modules.preprocessing.preprocessing import preprocess, sample_training_data, sample_test_data, sample_vqc_training_data
from modules.AD_loss_autoencoder.AD_loss_autoencoder import ADLossAutoencoder
from modules.QVC_autoencoder.QVC_autoencoder import QVCAutoencoder


def predict_HAE(qc_index = 2, loss_value = 0.022, n_samples = 100):
	path = f'../../data/training_results/pqc{qc_index}/training_result_loss_{loss_value}.pt'
	hae = HAE(qc_index=qc_index)
	hae.load_state_dict(torch.load(os.path.join(dirname, path)))
	hae.eval()

	test_data = Variable(torch.FloatTensor(sample_test_data(n_samples)[0]))
	test_labels = sample_test_data(n_samples)[1]

	train_data = Variable(torch.FloatTensor(sample_training_data(1000)[0]))

	test_data_latent_HAE = hae.get_latent_space_state(test_data)
	train_data_latent_HAE = hae.get_latent_space_state(train_data)

	if_model_HAE = IsolationForest().fit(train_data_latent_HAE.tolist())
	predict_HAE = if_model_HAE.predict(test_data_latent_HAE.tolist())

	return (predict_HAE, test_labels)


def predict_classical(n_samples = 100):
	path = f'../../data/training_results/classical/training_result_loss_0.022.pt'
	cae = ClassicalAutoencoder()
	cae.load_state_dict(torch.load(os.path.join(dirname, path)))
	cae.eval()

	test_data = Variable(torch.FloatTensor(sample_test_data(n_samples)[0]))
	test_labels = sample_test_data(n_samples)[1]
	train_data = Variable(torch.FloatTensor(sample_training_data(1000)[0]))

	test_data_latent = cae.get_latent_space_state(test_data)
	train_data_latent = cae.get_latent_space_state(train_data)

	if_model = IsolationForest().fit(train_data_latent.tolist())
	predict = if_model.predict(test_data_latent.tolist())

	return (predict, test_labels)


def predict_ADL(qc_index = 2, loss_value = 1.003, n_samples = 100):
	path = f'../../data/training_results_ADLoss/pqc{qc_index}/training_result_loss_{loss_value}.pt'
	adl_ae = ADLossAutoencoder(qc_index=qc_index)
	adl_ae.hybrid.load_state_dict(torch.load(os.path.join(dirname, path)))
	adl_ae.hybrid.eval()

	test_data = Variable(torch.FloatTensor(sample_test_data(n_samples)[0]))
	test_labels = sample_test_data(n_samples)[1]

	predict = []
	for data in test_data:
		predict.append(adl_ae.predict(data))

	return (predict, test_labels)


def predict_QVC(qc_index = 9, loss_value = 1.194, n_samples = 100, is_binary=False):
	if is_binary:
		path = f"../../data/training_results_QVC/{'zzfeaturemap_twolocal' if qc_index == 9 else 'pqc' + str(qc_index)}/binary_cl/loss_{loss_value}.txt"
	else:
		path = f"../../data/training_results_QVC/{'zzfeaturemap_twolocal' if qc_index == 9 else 'pqc' + str(qc_index)}/loss_{loss_value}.txt"
	
	theta = []
	with open(os.path.join(dirname, path), 'r') as f:
		[theta.append(float(line)) for line in f.readlines()]

	test_data = Variable(torch.FloatTensor(sample_test_data(n_samples)[0]))
	test_labels = sample_test_data(n_samples)[1]

	#test_data = Variable(torch.FloatTensor(sample_vqc_training_data(n_samples)[0]))
	#test_labels = np.array(sample_vqc_training_data(n_samples)[1])
	#test_labels = np.where(test_labels>3, 1, -1)

	qvc = QVCAutoencoder(qc_index=qc_index)
	predict = qvc.eval(theta, test_data, is_binary)

	print(f"predict: {predict}")

	return (predict, test_labels)
