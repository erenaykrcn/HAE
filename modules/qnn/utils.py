import torch


def convert_prob_to_exp_batch(probs):
	"""
		Helper tool to get the expectation values
		of the single qubits from prob dict.
	"""
	x0 = probs[:,1] + probs[:,3] + probs[:,5] + probs[:,7] + probs[:,9] + probs[:,11] + probs[:,13] + probs[:,15]
	x1 = probs[:,2] + probs[:,3] + probs[:,6] + probs[:,7] + probs[:,10] + probs[:,11] + probs[:,14] + probs[:,15]
	x2 = probs[:,4] + probs[:,5] + probs[:,6] + probs[:,7] + probs[:,12] + probs[:,13] + probs[:,14] + probs[:,15]
	x3 = probs[:,8] + probs[:,9] + probs[:,10] + probs[:,11] + probs[:,12] + probs[:,13] + probs[:,14] + probs[:,15]

	x = torch.stack((x0, x1, x2, x3), -1)
	return x


def convert_prob_to_exp(probs):
	"""
		Helper tool to get the expectation values
		of the single qubits from prob dict.
	"""
	x0 = probs[1] + probs[3] + probs[5] + probs[7] + probs[9] + probs[11] + probs[13] + probs[15]
	x1 = probs[2] + probs[3] + probs[6] + probs[7] + probs[10] + probs[11] + probs[14] + probs[15]
	x2 = probs[4] + probs[5] + probs[6] + probs[7] + probs[12] + probs[13] + probs[14] + probs[15]
	x3 = probs[8] + probs[9] + probs[10] + probs[11] + probs[12] + probs[13] + probs[14] + probs[15]

	x = torch.stack((x0, x1, x2, x3), -1)
	return x
