import os
dirname = os.path.dirname(__file__)

import torch
from torch.autograd import Variable

from sklearn.ensemble import IsolationForest

import sys
sys.path.append(os.path.join(dirname, '../../'))
from modules.HAE.HAE import HAE
from modules.classical_autoencoder.classical_autoencoder import ClassicalAutoencoder
from modules.preprocessing.preprocessing import preprocess, sample_training_data


def predict_HAE(qc_index = 2, loss_value = 0.022, n_samples = 100, offset=0):
	path = f'../../data/training_results/pqc{qc_index}/training_result_loss_{loss_value}.pt'
	hae = HAE(qc_index=qc_index)
	hae.load_state_dict(torch.load(os.path.join(dirname, path)))
	hae.eval()

	test_data = Variable(torch.FloatTensor(preprocess()[2][offset:offset+n_samples]))
	test_labels = preprocess()[3][offset:offset+n_samples]
	train_data = Variable(torch.FloatTensor(sample_training_data(1000)[0]))

	test_data_latent_HAE = hae.get_latent_space_state(test_data)
	train_data_latent_HAE = hae.get_latent_space_state(train_data)

	if_model_HAE = IsolationForest().fit(train_data_latent_HAE.cpu().detach())
	predict_HAE = if_model_HAE.predict(test_data_latent_HAE.cpu().detach())

	return (predict_HAE, test_labels)


def predict_classical(n_samples = 100, offset=0):
	path = f'../../data/training_results/classical/training_result_loss_0.022.pt'
	cae = ClassicalAutoencoder()
	cae.load_state_dict(torch.load(os.path.join(dirname, path)))
	cae.eval()

	test_data = Variable(torch.FloatTensor(preprocess()[2][offset:offset+n_samples]))
	test_labels = preprocess()[3][offset:offset+n_samples]
	train_data = Variable(torch.FloatTensor(sample_training_data(1000)[0]))

	test_data_latent = cae.get_latent_space_state(test_data)
	train_data_latent = cae.get_latent_space_state(train_data)

	if_model = IsolationForest().fit(train_data_latent.cpu().detach())
	predict = if_model.predict(test_data_latent.cpu().detach())

	return (predict, test_labels)
