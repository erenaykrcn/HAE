import torch
from AD_loss_autoencoder import ADLossAutoencoder


model = ADLossAutoencoder(qc_index=4, learningRate=1e-1, approach=0)

if torch.cuda.is_available():
	model = model.cuda()

model.trainADLoss(n_samples=100, epochs=10)