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


"""from predict import predict_HAE, predict_classical, predict_ADL, predict_QVC

predict, labels = predict_HAE(qc_index = 2, loss_value = 0.022, n_samples=500)
f1, precision, recall = get_scores(predict, labels)

print(f"HYBRID: f1: {f1}; precision: {precision}; recall: {recall}")

predict, labels = predict_classical(n_samples=500)
f1, precision, recall = get_scores(predict, labels)

print(f"CLASSICAL: f1: {f1}; precision: {precision}; recall: {recall}")

#predict, labels = predict_ADL(qc_index = 4, loss_value = 0.193, n_samples=100)
#f1, precision, recall = get_scores(predict, labels)

#print(f"AD_LOSS: f1: {f1}; precision: {precision}; recall: {recall}")

predict, labels = predict_QVC(qc_index = 9, loss_value = 0.394, n_samples=500, is_binary=True)
f1, precision, recall = get_scores(predict, labels)

print(f"QVC_LOSS: f1: {f1}; precision: {precision}; recall: {recall}")"""