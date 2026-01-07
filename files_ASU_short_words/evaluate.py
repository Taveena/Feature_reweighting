from torch.utils.data import DataLoader
from torch.utils import data

#import models
from eeg_model import *
from ShallowConvNet import *
from eeg_deepconvnet import *
from eeg_tsseffnet import *
from eeg_fgnet import *
from LMDA import LMDA
#from eeg_channel_gated_final import *
#from eeg_temporal_gated_final import *
#from eeg_fgnet_msfnot import *
#from eeg_fgnet import *

from data_loader_eegnet_ASU import custom_dataset
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, cohen_kappa_score, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score
import os 
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import csv
import argparse
import sys
import random

classes=['up', 'in', 'out']
n_classes = len(classes)

seed_n=0

print('seed is ' + str(seed_n))
random.seed(seed_n)
np.random.seed(seed_n)
torch.manual_seed(seed_n)
torch.cuda.manual_seed(seed_n)
torch.cuda.manual_seed_all(seed_n)

parser = argparse.ArgumentParser(description='Requirment of the progarm to get executed')
parser.add_argument('--model_name', type=str, default='fgnet', 
                    help='name of the model')

args = parser.parse_args()
model_name = args.model_name #100

def indices_to_one_hot(data, nb_classes):
	"""Convert an iterable of indices to one-hot encoded labels."""
	targets = np.array(data).reshape(-1)
	return np.eye(nb_classes)[targets]

def evaluate(model, loader):
	accuracy=0.0
	running_loss = 0.0
	predictions=[]
	pred_score_vector=[]
	pred_score=[]
	labels_all=[]
	model.eval()
	features=[]
	with torch.no_grad():
		for i, (inputs, labels) in enumerate(loader): 
			#if model_name in  ['shallowconvnet_repeated', 'deepconvnet_repeated', 'DeepConvNet_orig']:
				#inputs = inputs.tile((1,1,13)).cuda()
				#inputs = inputs[:,:,0:1000].cuda()
			if model_name in ['tsseffnet', 'TS_SEFFNet']:
				inputs = inputs.unsqueeze(3)
			else:
				inputs = inputs.unsqueeze(1)	
			inputs = inputs.type(torch.FloatTensor).cuda()
			labels = labels.cuda()
			'''
			if embedding_flag:
				output,embedding,_,_ = model(inputs,embedding=True)
				#print(embedding.shape)
				features.append(embedding)
			'''
			output = model(inputs)
			output = F.softmax(output)
			pred_score_vector = pred_score_vector + output.cpu().numpy().tolist()

			pred = torch.argmax(output, dim=1)
			pred = pred.reshape(-1)

			pred_temp = torch.max(output, dim=1)[0]
			pred_temp = pred_temp.reshape(-1)


			pred_score=pred_score +pred_temp.tolist()


			predictions = predictions + pred.cpu().numpy().tolist()
			labels_all = labels_all + labels.cpu().numpy().tolist()
			accuracy += (pred.eq(labels.data.view_as(pred)).sum())
		acc = accuracy.cpu().numpy()/(len(loader.dataset)*1.0)*100
	'''
	if embedding_flag:
		features=torch.cat(features,dim=0)
		print(features.shape)
		features= features.cpu().numpy()
		numpy_save_features_name =os.path.join(numpy_save_path,'features',model_path.split(os.sep)[-2]+'_'+str(n_classes)+'.npy')
		np.save(numpy_save_features_name,features)
	'''


	loss = running_loss*1.0/len(loader)

	kappa = cohen_kappa_score(labels_all, predictions)
	f1= f1_score(labels_all, predictions,average='macro')
	precision = precision_score(labels_all, predictions,average='macro')
	recall = recall_score(labels_all, predictions,average='macro')
	'''
	cm = confusion_matrix(labels_all, predictions,normalize='all')
	df_cm = pd.DataFrame(cm, index = [i for i in classes],columns = [i for i in classes])
	plt.figure(figsize = (fig_size, fig_size))
	#sn.heatmap(df_cm, annot=True, cmap="YlGnBu",linewidths=.5)
	palette = sn.color_palette("ch:start=.2,rot=-.3", as_cmap=True)
	sn.set(font_scale=8)    #15 for 2 class, 11 for 3 class, 8 for 4 class
	sn.heatmap(df_cm, annot=True, linewidths=.5, cmap=palette, linecolor='black', annot_kws={"fontsize":90}, cbar=False)   #fontsize: 170 for 2 class, 130 for 3 class, 90 for 4 class
	plt.title(model_name, fontsize=100)   #fontsize: 170 for 2 class, 130 for 3 class, 100 for 4 class
	#sn.heatmap(df_cm, annot=True, cmap="crest",linewidths=.5)
	saving_file_name = model_path.split(os.sep)[-2]+'_'+str(n_classes)+'.png'
	print(saving_file_name)

	if not os.path.exists(os.path.join(cm_save_path,str(n_classes))):
		os.mkdir(os.path.join(cm_save_path,str(n_classes)))

	plt.savefig(os.path.join(cm_save_path,str(n_classes),saving_file_name), dpi=400)
	print(os.path.join(cm_save_path,str(n_classes),saving_file_name))
	#plt.show()

	#numpy_save_prediction_name =os.path.join(numpy_save_path,'prediction',model_path.split(os.sep)[-2]+'_'+str(n_classes)+'.npy')
	#numpy_save_gt_name =os.path.join(numpy_save_path,'gt',model_path.split(os.sep)[-2]+'_'+str(n_classes)+'.npy')
	#np.save(numpy_save_prediction_name,pred_score_vector)
	#np.save(numpy_save_gt_name,labels_all_one_hot)
	'''
	pred_score_vector=np.asarray(pred_score_vector)
	labels_all_one_hot = indices_to_one_hot(labels_all, n_classes)
	#print(labels_all_one_hot.shape, pred_score_vector.shape)
	auc_roc = roc_auc_score(labels_all_one_hot, pred_score_vector)

	return [acc, precision, recall, f1, auc_roc, kappa],[pred_score, predictions, labels_all]

batch_size = 64
n_classes = 3
channel = 60
window=512

#model_name = 'fgnet'   # eegnet, eegnetfusion, mieegnet, deep, shallow, TS_SEFFNet, temporal_gated_final, channel_gated_final, fgnet, fgnet_msfnot

if not os.path.exists('./metrics'):
	os.mkdir('./metrics')

if not os.path.exists('./metrics/'+model_name+'/'):
	os.path.join('./metrics/',model_name+'/')
	os.mkdir('./metrics/'+model_name+'/')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

accuracy_list=[]
precision=[]
recall=[]
f1=[]
auc_roc=[]
kappa=[]
for j in [1,3,5,6,8,12]:
	print('Subject '+str(j)+ '-----------------------')
	for i in range(1,6):
		model_path='weights_short_words/'+model_name+'/'+str(j)+'/'+str(i)+'/'
		data_path='./../ASU_dataset/Short_words_dataset_sampled/split_asu_5fold_train_test_val_sub'+str(j)+'/'+str(i)+'_test.txt'

		# data loaders for train , validation, test 
		testDataset = custom_dataset(path = data_path, source='list', window_size=window)
		testDataloader = DataLoader(testDataset, batch_size=batch_size, shuffle=False, num_workers=4)

		# creating a model 
		if model_name=='eegnet':
			net = EEGNet_v2(n_classes,Chans=channel, Samples=window).cuda()
		elif model_name=='eegnetfusion':
			net = EEGNet_fusion(n_classes,Chans=channel, Samples=window, n_size=1792).cuda()
		elif model_name=='mieegnet':
			net = MI_EEGNet(n_classes,Chans=channel, Samples=window).cuda()
		elif model_name=='tsseffnet':
			net = TS_SEFFNet(in_chans=channel, n_classes = n_classes).cuda()#, Samples=window
		elif model_name=='fgnet':
			net = EEGNet_fgnet(n_classes, Chans=channel, Samples=window).cuda()
		elif model_name=='shallow':
			net = ShallowConvNet(n_classes=n_classes,input_shape=[batch_size,1,channel,window], F1=40, T1=25, F2=40, P1_T=75, P1_S=5, drop_out=0.5,last_dim=3320).cuda()
		elif model_name=='deep':
			net = DeepConvNet(n_classes=n_classes,input_shape=[batch_size,1,channel,window],first_conv_length=10,block_out_channels=[25, 25, 50, 100, 200], pool_size=2, last_dim=4600).cuda()
		elif model_name=='lmda':
			net = LMDA(num_classes=n_classes, chans=channel, samples=window).cuda()
		else:
			print('Wrong model name')
		#net = EEGNet_proposed_temporal_gated_final(n_classes,Chans=channel, Samples=window).cuda()
		#net = EEGNet_proposed_channel_gated_final(n_classes, Chans=channel, Samples=window).cuda()
		#net = EEGNet_fgnet_msfnot(n_classes, Chans=channel, Samples=window).cuda()
		#net = EEGNet_fgnet_v1(n_classes, Chans=channel, Samples=window).cuda()
		#net.load_state_dict(torch.load(os.path.join(model_path,'best_val_loss_weight.pth')))
		checkpoint = torch.load(os.path.join(model_path,'current.pth'))
		net.load_state_dict(checkpoint['model_state_dict'])

		test_metric = evaluate(net, testDataloader) #, meta_data

		#print('Accuracy of fold '+str(i))
		#print(round(test_metric[0],4), '\t', end="")
		#print(round(test_metric[0][1],4), '\t', end="")
		accuracy_list.append(str(round(test_metric[0][0],4)))
		precision.append(str(round(test_metric[0][1],4)))
		recall.append(str(round(test_metric[0][2],4)))
		f1.append(str(round(test_metric[0][3],4)))
		auc_roc.append(str(round(test_metric[0][4],4)))
		kappa.append(str(round(test_metric[0][5],4)))
	#precision.insert((j*10),'\n')
	with open('./metrics/'+model_name+'/accuracy.csv', 'a') as e:
	# create the csv writer
		writer = csv.writer(e)
	# write a row to the csv file
		writer.writerow(accuracy_list)
	e.close()
	accuracy_list.clear()
	with open('./metrics/'+model_name+'/precision.csv', 'a') as f:
	# create the csv writer
		writer = csv.writer(f)
	# write a row to the csv file
		writer.writerow(precision)
	f.close()
	precision.clear()
	with open('./metrics/'+model_name+'/recall.csv', 'a') as g:
	# create the csv writer
		writer = csv.writer(g)
	# write a row to the csv file
		writer.writerow(recall)
	g.close()
	recall.clear()
	with open('./metrics/'+model_name+'/f1.csv', 'a') as h:
	# create the csv writer
		writer = csv.writer(h)
	# write a row to the csv file
		writer.writerow(f1)
	h.close()
	f1.clear()
	with open('./metrics/'+model_name+'/auc_roc.csv', 'a') as k:
	# create the csv writer
		writer = csv.writer(k)
	# write a row to the csv file
		writer.writerow(auc_roc)
	k.close()
	auc_roc.clear()
	with open('./metrics/'+model_name+'/kappa.csv', 'a') as l:
	# create the csv writer
		writer = csv.writer(l)
	# write a row to the csv file
		writer.writerow(kappa)
	l.close()
	kappa.clear()
	#print('\n')




'''
print('Precision', test_metric[1])
print('Recall', test_metric[2])
print('F1', test_metric[3])
print('auc', test_metric[4])
print('kappa', test_metric[5])
'''




