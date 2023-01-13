import os
dirname = os.path.dirname(__file__)

import torch
from torch.autograd import Variable
import numpy as np

from sklearn.ensemble import IsolationForest

import sys
sys.path.append(os.path.join(dirname, '../'))
from HAE.HAE import HAE
from classical_autoencoder.classical_autoencoder import ClassicalAutoencoder
from classical_autoencoder.classical_binary_classifier import BinaryClassification
from preprocessing.preprocessing import preprocess, sample_training_data, sample_test_data
from AD_loss_autoencoder.AD_loss_autoencoder import ADLossAutoencoder
from QVC_autoencoder.QVC_autoencoder import QVCAutoencoder


def predict_HAE(qc_index, path, n_samples = 100, test_data=None, test_labels=None, custom_dict={}):
	hae = HAE(qc_index=qc_index, custom_qc=custom_dict)
	hae.load_state_dict(torch.load(path))
	hae.eval()
	
	if  (test_data==None) or (test_labels==None):
		test_data, test_labels = sample_test_data(n_samples, True)
	test_data = Variable(torch.FloatTensor(test_data))

	train_data = Variable(torch.FloatTensor(sample_training_data(1000)[0]))

	test_data_latent_HAE = hae.get_latent_space_state(test_data)
	train_data_latent_HAE = hae.get_latent_space_state(train_data)

	if_model_HAE = IsolationForest().fit(train_data_latent_HAE.tolist())
	predict_HAE = if_model_HAE.predict(test_data_latent_HAE.tolist())

	return (predict_HAE, test_labels)


def predict_classical(n_samples = 100, test_data=None, test_labels=None):
	path = f'../../data/training_results/classical/training_result_loss_0.022.pt'
	cae = ClassicalAutoencoder()
	cae.load_state_dict(torch.load(os.path.join(dirname, path)))
	cae.eval()

	if (test_data==None) or (test_labels==None):
		test_data, test_labels = sample_test_data(n_samples, True)
	test_data = Variable(torch.FloatTensor(test_data))
	train_data = Variable(torch.FloatTensor(sample_training_data(1000)[0]))

	test_data_latent = cae.get_latent_space_state(test_data)
	train_data_latent = cae.get_latent_space_state(train_data)

	if_model = IsolationForest().fit(train_data_latent.tolist())
	predict = if_model.predict(test_data_latent.tolist())

	return (predict, test_labels)


def predict_classical_binary_cl(n_samples = 100, test_data=None, test_labels=None):
	path_cae = f'../../data/training_results/classical/training_result_loss_0.022.pt'
	cae = ClassicalAutoencoder()
	cae.load_state_dict(torch.load(os.path.join(dirname, path_cae)))
	cae.eval()

	path_cl = f'../../data/training_results/classical_binary_cl/training_result_loss_0.037.pt'
	model = BinaryClassification()
	model.load_state_dict(torch.load(os.path.join(dirname, path_cl)))
	model.eval()

	if (test_data==None) or (test_labels==None):
		test_data, test_labels = sample_test_data(n_samples, True)
	test_data = Variable(torch.FloatTensor(test_data))

	test_data_latent = cae.get_latent_space_state(test_data)
	predict = model(test_data_latent).detach().numpy()
	predict = np.squeeze(np.where(predict<0.5, -1, 1))

	return (predict, test_labels)


def predict_ADL(qc_index = 2, loss_value = 1.003, n_samples = 100, test_data=None, test_labels=None):
	path = f'../../data/training_results_ADLoss/pqc{qc_index}/training_result_loss_{loss_value}.pt'
	adl_ae = ADLossAutoencoder(qc_index=qc_index)
	adl_ae.hybrid.load_state_dict(torch.load(os.path.join(dirname, path)))
	adl_ae.hybrid.eval()


	if  (test_data==None) or (test_labels==None):
		test_data, test_labels = sample_test_data(n_samples, True)
	test_data = Variable(torch.FloatTensor(test_data))

	predict = []
	for data in test_data:
		predict.append(adl_ae.predict(data))

	return (predict, test_labels)


def predict_QVC(qc_index, path, n_samples = 100, is_binary=True, test_data=None, test_labels=None, custom_dict={}):
	theta = []
	with open(path, 'r') as f:
		[theta.append(float(line)) for line in f.readlines()]

	if  (test_data==None) or (test_labels==None):
		test_data, test_labels = sample_test_data(n_samples, True)
	test_data = Variable(torch.FloatTensor(test_data))

	qvc = QVCAutoencoder(qc_index=qc_index, custom_qc=custom_dict)
	predict = qvc.eval(theta, test_data, is_binary)

	return (predict, test_labels)
