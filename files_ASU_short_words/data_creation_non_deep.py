import numpy as np
import argparse
import csv
import random
import os
import hdf5storage

parser = argparse.ArgumentParser(description='Requirment of the progarm to get executed')
parser.add_argument('--data_list', type=str, default='split_bci/A01/A01_1_train.txt', 
                    help='the log file storage path')
parser.add_argument('--data_save_path', type=str, default='data_non_deep/A01/train_data.npy', 
                    help='path to save concatenated data into npy file')
parser.add_argument('--data_label_save_path', type=str, default='data_non_deep/A01/train_label.npy', 
                    help='path to save concatenated data labels into npy file')

args=parser.parse_args()
data_list=args.data_list
data_save_path=args.data_save_path 
data_label_save_path=args.data_label_save_path 

data_path = data_save_path.split('/')
#print('data_path parts', data_path[0], data_path[1], data_path[2])

if not os.path.exists(data_path[0]):
	os.mkdir(data_path[0])
if not os.path.exists(os.path.join(data_path[0], data_path[1])):
	os.mkdir(os.path.join(data_path[0], data_path[1]))
if not os.path.exists(os.path.join(data_path[0], data_path[1], data_path[2])):
	os.mkdir(os.path.join(data_path[0], data_path[1], data_path[2]))

# read datapaths from the text file, then open the data and append it to a list, then convert the list to a numpy array
def load_dataset_list(path):
	dataset = []
	reader = csv.reader(open(path, "r"))
	for row in reader:
		if int(row[1])==-1:
			pass
		else:
			dataset.append(row)
	#print('shuffling data')
	for i in range(5):
		random.shuffle(dataset);
	#print('number of sample : '+str(len(self.dataset)));
	return dataset


def load_data(instance, type='mat'):
	signal_file = instance[0]
	label = int(instance[1])

	if type=='mat':
		signal = hdf5storage.loadmat(signal_file)
		#signal = signal['data']
		g1 = signal_file.split('/')[-1]
		g2 = g1.split('.')[0]
		g3 = g2[0:4] + '1' + g2[-1]
		#print(g3)
		signal = signal[g3]

	if type=='txt':
		signal=[]
		with open(signal_file,'r') as fp:
			content = fp.readlines()
		for channel in content:
			signal.append(channel.strip().split(',')[1:])
		signal = np.asarray(signal)

	if type=='csv':
		signal=[]
		reader = csv.reader(open(signal_file, "r"))
		for row in reader:
			signal.append(row[0:])
		signal = np.asarray(signal)

	return signal, label

# crawl over all data subject specific and then create a single npy file for complete data of each subject
#def complete_data():
	

dataset = load_dataset_list(data_list)
data, data_label=[], []
for index in range(len(dataset)):
	instance = dataset[index]
	signal, label = load_data(instance, type='mat')
	data.append(signal)
	data_label.append(label)

data = np.asarray(data)
data_label = np.asarray(data_label)
with open(data_save_path, 'wb') as f:
	np.save(f, data)
f.close()
with open(data_label_save_path, 'wb') as g:
	np.save(g, data_label)
g.close()

