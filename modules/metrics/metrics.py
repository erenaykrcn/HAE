import numpy as np
from scipy import stats

def get_scores(predict, labels):
	predict = np.array(predict)
	labels = np.array(labels)

	num_anomalies_labeled_by_model = len(predict[np.where(predict==-1)])
	num_anomalies = len(labels[np.where(labels<4)])

	labels[np.where(labels<4)] = np.ones(num_anomalies) * (-1)
	correctly_discovered_anomalies = np.where(labels == predict, -1, 0)
	num_correctly_discovered_anomalies = len(correctly_discovered_anomalies[np.where(correctly_discovered_anomalies==-1)])

	precision = num_correctly_discovered_anomalies / num_anomalies_labeled_by_model
	recall = num_correctly_discovered_anomalies / num_anomalies
	f1 = stats.hmean([precision, recall])

	return (f1, precision, recall)


# TODO: Delete this part after testing is over
from predict import predict_HAE, predict_classical

predict, labels = predict_HAE(qc_index = 8, loss_value = 0.039, n_samples=600, offset=100)
f1, precision, recall = get_scores(predict, labels)

print(f"HYBRID: f1: {f1}; precision: {precision}; recall: {recall}")

predict, labels = predict_classical(n_samples=400, offset=0)
f1, precision, recall = get_scores(predict, labels)

print(f"CLASSICAL: f1: {f1}; precision: {precision}; recall: {recall}")