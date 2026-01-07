import random
import os
import scipy.io
import fnmatch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils import data
import torch
import hdf5storage
import csv
import numpy as np
from scipy.signal import savgol_filter
from scipy import signal as sg


class custom_dataset(data.Dataset):
	def __init__(self, path, viz=False, window_size=-1, source='list'):
		super(custom_dataset, self).__init__()
		self.dataset = []
		self.window_size=window_size
		if source=='list':
			#print('list')
			self.load_dataset_list(path)
		else:
			#print('directory')
			self.load_dataset_dir(path)
		self.viz = viz
		#self.mean = np.load('mean.npy')
		#self.std = np.load('std.npy')
		#self.mean = self.mean.astype('float32')
		#self.std = self.std.astype('float32')

	def load_dataset_list(self, path):
		reader = csv.reader(open(path, "r"))
		for row in reader:
			if int(row[1])==-1:
				pass
			else:
				self.dataset.append(row)
		#print('shuffling data')
		for i in range(5):
			random.shuffle(self.dataset);
		print('number of sample : '+str(len(self.dataset)));

	def load_data(self, instance, repeat, data_type='txt'):
		signal_file = instance[0]
		label = int(instance[1])

		if data_type=='mat':
			signal = hdf5storage.loadmat(signal_file)
			signal = signal['data']	

		if data_type=='txt':
			signal=[]
			with open(signal_file,'r') as fp:
				content = fp.readlines()
			for channel in content:
				signal.append(channel.strip().split(',')[1:])
			signal = np.asarray(signal)

		if data_type=='csv':
			signal=[]
			reader = csv.reader(open(signal_file, "r"))
			for row in reader:
				signal.append(row[1:])
				#signal = list(map(lambda x:x, signal))   #[0]
			'''
			if repeat=="True":
				signal = np.tile(signal, (1,13))
				signal = signal[:,0:1000]
			'''
			#for i in range(len(signal)):
			#	signal[i] = ','.join(signal[i])
			signal = np.asarray(signal)  #, dtype=object
			#print('length of the signal',signal[63][:10])
			#input('halt')
		return signal, label

	def random_sample(self, signal):
		signal_length = signal.shape[1]

		if signal_length> self.window_size:
			if self.window_size>0:
				start_index = random.randint(0, signal_length-self.window_size)
				window_size = self.window_size
			else:
				start_index=0
				window_size=signal_length
			resized_signal = signal[:,start_index: start_index+window_size]
		else:
			resized_signal = np.zeros((signal.shape[0], self.window_size),dtype='float32')
			resized_signal[:,0:signal.shape[-1]] = self.preprocessing_persample(signal)
		return resized_signal

	def visualize(self, signal):
		for i in range(signal.shape[0]):
			plt.plot(signal[i], linestyle='-',linewidth=2.0, color= 'r')
		plt.show()
		return 0

	def __len__(self):
		return len(self.dataset)												

	def butter_bandpass(self, lowcut, highcut, fs=160, order=5, label=None):
		nyq = 0.5 * fs
		low = lowcut / nyq
		high = highcut / nyq
		sos = sg.butter(order, [low, high], btype='band', output='sos')
		return sos 

	def band_pass_filter(self, signal, label=None):
		order = 2
		fs = 160
		filter = self.butter_bandpass(0.5, 40, fs, order, label)
		filtered = sg.sosfilt(filter, signal)
		return filtered	

	def preprocessing_wholistic(self, signal):
		new_signal = np.zeros(signal.shape,dtype='float32')
		signal = self.band_pass_filter(signal)
		signal=savgol_filter(signal, 5, 2)
	
		for i in range(signal.shape[1]):
			temp=(signal[:,i]-self.mean) /self.std						
			new_signal[:,i]=temp

		min_vector = np.min(new_signal,axis=1).reshape(-1,1)
		max_vector = np.max(new_signal,axis=1).reshape(-1,1)
		signal = new_signal - np.tile(min_vector,new_signal.shape[1])
		rem = max_vector-min_vector 
		signal = signal/rem
		signal = (signal -0.5)/0.5
		return signal
	

	def preprocessing_persample(self, signal):
		#signal = self.band_pass_filter(signal)
		#signal=savgol_filter(signal, 5, 2)
		#print(signal.shape)

		# z-score 
		mean = np.mean(signal, axis=1).reshape(-1,1)
		std = np.std(signal, axis=1).reshape(-1,1)
		new_signal = (signal-np.tile(mean,signal.shape[1]))/np.tile(std,signal.shape[1])
		'''
		#print('mean-----------', mean, 'std++++++++++++++',std)
		if std.any()==0:
			print('mean-----------', mean, 'std++++++++++++++',std)
			new_signal = (signal-np.tile(mean,signal.shape[1]))#/np.tile(std,signal.shape[1])
		else:
			new_signal = (signal-np.tile(mean,signal.shape[1]))/np.tile(std,signal.shape[1])
		'''
		# min-max 
		#min = np.min(signal, axis=1).reshape(-1,1)
		#max = np.max(signal, axis=1).reshape(-1,1)
		
		#new_signal = (signal-np.tile(min,signal.shape[1]))/np.tile((max-min),signal.shape[1])

		'''
		for i in range(signal.shape[1]):
			temp=(signal[:,i]-mean) /std						
			new_signal[:,i]=temp
		'''
		return new_signal

	def __getitem__(self, index):
		instance = self.dataset[index]
		signal, label = self.load_data(instance, repeat = "False", data_type='csv')   ##### repeat variable to repeat the data to increase data length
		#print('length of the signal', signal.shape)
		signal = signal.astype('float32')
		signal = self.preprocessing_persample(signal)
		#signal = self.random_sample(signal)
		if self.viz:
			self.visualize(signal)
		return signal, label  

if __name__=='__main__':
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	trainDataset = custom_dataset(path = 'kfold_4sec/fold1train.txt', source='list', window_size=640, viz=True)
	trainDataloader = DataLoader(trainDataset,batch_size=10, shuffle=False, num_workers=1)
	for i, (signal, label) in enumerate(trainDataloader):
		print(signal.shape, label)

