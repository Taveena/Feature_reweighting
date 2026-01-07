from sklearn import metrics
import numpy as np


class LinearClassifier:
	"""
	Class represents the linear classification model
	"""

	def __init__(self, model):
		"""
		Initializes the linear classification model
		:param model: scikit-learn classification model
		"""
		self.model = model

	def fit(self, x_train, y_train, x_val, y_val):
		"""
		Fits the model
		:param x_train: training samples
		:param y_train: training labels
		:param x_val: validation samples
		:param y_val: validation labels
		:return: fitted model
		"""
		#self.model.fit(x_train, y_train[:, 0])
		if(self.model=='LinearDiscriminantAnalysis'):
			#x_train = np.split(x_train, [x_train.shape(0), ((x_train.shape(1))/1000), 1000])
			#x_val = np.split(x_val, [x_val.shape(0), ((x_val.shape(1))/1000), 1000])
			#print('shape of x_train and x_val', x_train.shape, x_val.shape)
			self.model.fit(x_train, y_train)
			return self.evaluate(x_val, y_val)
		else:
			self.model.fit(x_train, y_train)
			return self.evaluate(x_val, y_val)

	def evaluate(self, x_test, y_test):
		"""
		Evaluates the fitted model
		:param x_test: test samples
		:param y_test: test labels
		:return: required metrics
		"""
		predictions = []
		real_outputs = []
		prediction_probs = []
		#if(self.model == 'LinearDiscriminantAnalysis'):
			#x_test = np.split(x_test, [x_test.shape(0), ((x_test.shape(1))/1000), 1000])
			#print('shape of x_test', x_test.shape)
		for i in range(x_test.shape[0]):
			pattern = x_test[i, :].reshape(1, -1)
			prediction = self.model.predict(pattern)
			#print('prediction', prediction)
			predictions.append(prediction[0])
			#real_outputs.append(y_test[i, 0])
			real_outputs.append(y_test[i])
			prediction_prob = self.model.predict_proba(pattern)
			prediction_probs.append(prediction_prob[0])
		auc_roc = metrics.roc_auc_score(real_outputs, prediction_probs, multi_class='ovr')
		acc = metrics.accuracy_score(real_outputs, predictions)
		precision = metrics.precision_score(real_outputs, predictions, average='macro')
		recall = metrics.recall_score(real_outputs, predictions, average='macro')
		f1= metrics.f1_score(real_outputs, predictions, average='macro')
		kappa = metrics.cohen_kappa_score(real_outputs, predictions)
		#print('acc, precision, recall, f1, auc_roc, kappa', acc, precision, recall, f1, auc_roc, kappa)

		return [acc, precision, recall, f1, auc_roc, kappa] #


