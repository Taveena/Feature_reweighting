# https://github.com/stupiddogger/FBCSP/blob/master/FBCSP.py

# coding: utf-8
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from mne.decoding import CSP
import sklearn.feature_selection
from sklearn.feature_selection import mutual_info_classif
from scipy import signal
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import random 
random.seed(1)

class fbcsp:
	
	def __init__(self, X_train, y_train, X_val, y_val, X_test, y_test):
		super(fbcsp, self).__init__()

	# defination of bandpass filter 
	def butter_bandpass(self, lowcut, highcut, fs, order):
		nyq=0.5*fs
		low=lowcut/nyq
		high=highcut/nyq
		b,a=signal.butter(order,[low,high],'bandpass')
		return b,a

	def butter_bandpass_filter(self, data,lowcut, highcut, fs, order):
		b,a=self.butter_bandpass(lowcut,highcut,fs,order)
		#print('b and a', b, a, 'data', data)
		y=signal.filtfilt(b,a,data,axis=2)
		return y

	def frequency_bands(self, X_train, X_test, y_train):
		csp=CSP(n_components=4, reg=None, log=True, norm_trace=False)
		X_train=self.butter_bandpass_filter(X_train,lowcut=4,highcut=40,fs=250,order=2)
		X_test=self.butter_bandpass_filter(X_test,lowcut=4,highcut=40,fs=250,order=2)
		#acquire and combine features of different fequency bands
		features_train=[]
		features_test=[]
		freq=[4,8,12,16,20,24,28,32,36,40]
		for freq_count in range(len(freq)):
		#loop for frequency
			lower=freq[freq_count]
			if lower==freq[-1]:
				break
			higher=freq[freq_count+1]
			X_train_filt=self.butter_bandpass_filter(X_train,lowcut=lower,highcut=higher,fs=250,order=2)
			X_test_filt=self.butter_bandpass_filter(X_test,lowcut=lower,highcut=higher,fs=250,order=2)
			tmp_train=csp.fit_transform(X_train_filt,y_train)
			tmp_test=csp.transform(X_test_filt)
			if freq_count==0:
				features_train=tmp_train
				features_test=tmp_test
			else:
				features_train=np.concatenate((features_train,tmp_train),axis=1)
				features_test=np.concatenate((features_test,tmp_test),axis=1)
		#print('shape of train features',features_train.shape)
		#print('shape of test features',features_test.shape) 

		return features_train, features_test

	def feature_selection(self, features_train, y_train, features_test):
		#get the best k features base on MIBIF algorithm
		select_K=sklearn.feature_selection.SelectKBest(mutual_info_classif, k='all').fit(features_train, y_train) # k=10 for fbcsp, k='all' for fbcsp_all
		New_train=select_K.transform(features_train)
		#np.random.shuffle(New_train)
		New_test=select_K.transform(features_test)
		#np.random.shuffle(New_test)
		#print(New_train.shape)
		#print(New_test.shape)
		ss = preprocessing.StandardScaler()
		X_select_train = ss.fit_transform(New_train, y_train)
		X_select_test = ss.fit_transform(New_test)

		return X_select_train, X_select_test

	def classification(self, X_train, y_train, X_val, y_val, X_test, y_test):
		features_train, features_val = self.frequency_bands(X_train, X_val, y_train)
		features_train, features_test = self.frequency_bands(X_train, X_test, y_train)
		X_select_train, X_select_val = self.feature_selection(features_train, y_train, features_val)
		X_select_train, X_select_test = self.feature_selection(features_train, y_train, features_test)
		#y_test, y_pred, y_probs = self.classifier(X_select_train,y_train,X_select_test,y_test)
		return X_select_train, y_train, X_select_val, y_val, X_select_test, y_test

if __name__=="__main__":

	model = fbcsp(X_train, y_train, X_val, y_val, X_test, y_test)
	#X_train, y_train, X_test, y_test = model.load_data(train_data, train_label, val_data, val_label, test_data, test_label)
	#features_train, features_test = model.frequency_bands(X_train,X_test,y_train)
	#X_select_train, X_select_test = model.feature_selection(features_train,y_train,features_test)
	X_select_train, y_train, X_select_val, y_val, X_test, y_test = model.classification()
	#print(acc)
		
'''
		X_train = self.train_data
		X_val = self.val_data
		X_test = self.test_data
		y_train = self.train_label
		y_val = self.val_label
		y_test = self.test_label

	train_data= './data_non_deep/A01/train_data.npy'
	train_label= './data_non_deep/A01/train_label.npy'
	val_data= './data_non_deep/A01/val_data.npy'
	val_label= './data_non_deep/A01/val_label.npy'
	test_data= './data_non_deep/A01/test_data.npy'
	test_label= './data_non_deep/A01/test_label.npy'
	def classifier(self,X_select_train,y_train,X_select_test,y_test):
		#classify
		from sklearn.svm import SVC
		clf=svm.SVC(C=0.8,kernel='rbf', probability=True)
		clf.fit(X_select_train,y_train)
		y_pred=clf.predict(X_select_test)
		y_probs = clf.predict_proba(X_select_test)
		#print(y_test)
		#print(y_pred)
		#print(acc)

		return y_test, y_pred, y_probs
'''
		
	




