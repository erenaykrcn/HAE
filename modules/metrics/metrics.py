import numpy as np
from scipy import stats

def get_scores(predict, labels):
	predict = np.array(predict)
	labels = np.array(labels)

	num_anomalies_labeled_by_model = len(predict[np.where(predict==-1)])
	num_anomalies = len(labels[np.where(labels==-1)])

	predict = np.where(predict==-1, -1, 0)
	num_correctly_discovered_anomalies = np.sum(np.where(labels == predict, 1, 0))

	precision = num_correctly_discovered_anomalies / num_anomalies_labeled_by_model
	recall = num_correctly_discovered_anomalies / num_anomalies
	f1 = stats.hmean([precision, recall])

	return (f1, precision, recall)

