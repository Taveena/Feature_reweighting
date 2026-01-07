import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils import data
import torch.optim as optim
from torch.autograd import Variable
#from eeg_proposed_aux import *
#from eeg_proposed_aux_hierarchical import *
#from eeg_temporal_gated_aux import *
#from eeg_proposed_aux_final1 import *
#from eeg_fgnet import EEGNet_fgnet
#from eeg_fgnet_v7 import EEGNet_fgnet_v7
#from eeg_tsseffnet import TS_SEFFNet
#from ShallowConvNet import ShallowConvNet
#from DeepConvNet import DeepConvNet
#from eeg_channel_gated_final import EEGNet_proposed_channel_gated_final
#from eeg_fgnet_msfnot import EEGNet_fgnet_msfnot
from eeg_model import * 
#from eeg_temporal_gated_final import EEGNet_proposed_temporal_gated_final

from LMDA import LMDA

from data_loader_eegnet import custom_dataset
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import os 
import argparse
import sys
from datetime import datetime
import random
import numpy as np

parser = argparse.ArgumentParser(description='Requirment of the progarm to get executed')
parser.add_argument('--batch_size', type=int, default=64, 
                    help='input batch size for training (default: 128)')
parser.add_argument('--no_epochs', type=int, default=200,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--n_classes', type=int, default=4,
                    help='number of classes in the dataset')
parser.add_argument('--window', type=int, default=1000,
                    help='the temporal length of the signal')
parser.add_argument('--channel', type=int, default=22,
                    help='number of channels of the EEG signal or the temporal width of the signal')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--model_save_path', type=str, default='weights_5fold', 
                    help='the saving path of the model')
parser.add_argument('--log_file', type=str, default='log_files_hgd_5fold', 
                    help='the log file storage path')
parser.add_argument('--config_file', type=str, default='config', 
                    help='the configuration file storage path')
parser.add_argument('--train_list', type=str, default='split_bci/A01/A01_1_train.txt', 
                    help='the log file storage path')
parser.add_argument('--test_list', type=str, default='split_bci/A01/A01_1_test.txt', 
                    help='the log file storage path')
parser.add_argument('--val_list', type=str, default='split_bci/A01/A01_1_val.txt', 
                    help='the log file storage path')
parser.add_argument('--model_name', type=str, default='fgnet', 
                    help='model name for training')
parser.add_argument('--save_all_weights', action='store_true', default=False,
                    help='disable the functionality of saving weight after each epoch')
parser.add_argument('--resume_train', action='store_true', default=False,
                    help='disable the functionality of saving weight after each epoch')
parser.add_argument('--fold', type=int, default=1,
                    help='fold')
parser.add_argument('--subject', type=int, default=1,
                    help='subject number')
parser.add_argument('--weight_decay', type=int, default=1e-4,
                    help='weight decay of optimizer')
parser.add_argument('--beta1', type=int, default=0.5,
                    help='beta1 value of optimizer')
parser.add_argument('--beta2', type=int, default=0.999,
                    help='beta1 value of optimizer')
parser.add_argument('--factor', type=int, default=0.1,
                    help='factor of learning rate scheduler')
parser.add_argument('--patience', type=int, default=25,
                    help='patience of learning rate scheduler')
                    
args = parser.parse_args()
no_epoch=args.no_epochs #100
batch_size = args.batch_size #64
n_classes = args.n_classes #4
window= args.window #1125
channel =  args.channel #22
n_channels = channel
model_save_path=args.model_save_path #'weight'
log_file=args.log_file # 'log_files/log_imagined_eegnet_v2.txt'
save_all_weights=args.save_all_weights #False
resume_train=args.resume_train #False
model_name = args.model_name
train_list_split= args.train_list
test_list_split= args.test_list
val_list_split= args.val_list
lr = args.lr
fold = args.fold
subject = args.subject
weight_decay = args.weight_decay
beta1 = args.beta1
beta2 = args.beta2
factor = args.factor
patience = args.patience

config_file=args.config_file
configuration_file = os.path.join(config_file, model_name+'_'+str(n_classes)+'_'+'configuration.txt')


#if os.path.exists(configuration_file):
#	os.remove(configuration_file)

# create a directory to the save the models 
if not os.path.exists(model_save_path):
	os.mkdir(model_save_path)

if not os.path.exists(os.path.join(model_save_path,model_name)):
	os.mkdir(os.path.join(model_save_path,model_name))

if not os.path.exists(os.path.join(model_save_path,model_name,str(subject))):
	os.mkdir(os.path.join(model_save_path,model_name,str(subject)))

if not os.path.exists(os.path.join(model_save_path,model_name,str(subject),str(fold))):
	os.mkdir(os.path.join(model_save_path,model_name,str(subject),str(fold)))

# create a directory to the save the log file
if not os.path.exists(log_file):
	os.mkdir(log_file)

if not os.path.exists(os.path.join(log_file,model_name)):
	os.mkdir(os.path.join(log_file,model_name))

if not os.path.exists(os.path.join(log_file,model_name,str(subject))):
	os.mkdir(os.path.join(log_file,model_name,str(subject)))
	
log_file_path = os.path.join(log_file, model_name, str(subject), 'log_'+str(fold)+'.txt')


# datetime object containing current date and time
now = datetime.now()
 
#print("now =", now)

# dd/mm/YY H:M:S
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
#print("date and time =", dt_string)

seed_n = 0
print('seed is ' + str(seed_n))
random.seed(seed_n)
np.random.seed(seed_n)
torch.manual_seed(seed_n)
torch.cuda.manual_seed(seed_n)
torch.cuda.manual_seed_all(seed_n)

configuration = '==========================================\n'+\
                ' No of epoch      :' + str(no_epoch)+'\n'+\
                ' fold             :'+str(fold)+'\n'+\
                ' Batch Size       :' + str(batch_size)+'\n'+\
                ' No of classes    :'+str(n_classes)+'\n'+\
                ' window           :'+str(window)+'\n'+\
                ' learning rate    :'+str(lr)+'\n'+\
                ' seed value       :'+str(seed_n)+'\n'+\
                ' model save path  :' +  model_save_path+'\n'+\
                ' log file path    :'+ log_file+'\n'+\
                ' save all weights :'+str(save_all_weights)+'\n'+\
                ' Resume train     :'+ str(resume_train)+'\n'+\
                ' model name       :'+ model_name +'\n'+\
                ' train list       :'+ train_list_split +'\n'+\
                ' test list        :'+ test_list_split +'\n'+\
                ' val list         :'+ val_list_split +'\n'+\
                ' Date and time    :'+ dt_string +'\n'

print('Used configuration =='+'\n'+configuration)
with open(configuration_file, 'a') as file:
	file.write(configuration) 

def evaluate(model, loader, criterion):
	accuracy=0.0
	running_loss = 0.0
	predictions=[]
	labels_all=[]
	model.eval()
	with torch.no_grad():
		for i, (inputs, labels) in enumerate(loader): 
			inputs = inputs.unsqueeze(1)		
			inputs = inputs.type(torch.FloatTensor).cuda()
			labels = labels.cuda()#-1
			output = model(inputs) #
			loss = criterion(output, labels)# + 0.1*criterion(aux, labels)
			running_loss += loss.data
			output = F.softmax(output)
			pred = torch.argmax(output, dim=1)
			pred = pred.reshape(-1)
			predictions = predictions + pred.cpu().numpy().tolist()
			labels_all = labels_all + labels.cpu().numpy().tolist()
			accuracy += (pred.eq(labels.data.view_as(pred)).sum())
		acc = accuracy.cpu().numpy()/(len(loader.dataset)*1.0)*100

	loss = running_loss*1.0/len(loader)
	'''
	precision_left = precision_score(labels_all, predictions, average='binary', pos_label=0)
	recall_left = recall_score(labels_all, predictions, average='binary', pos_label=0)
	f1_left = f1_score(labels_all, predictions, pos_label=0, average='binary')
	precision_right = precision_score(labels_all, predictions, average='binary', pos_label=1)
	recall_right = recall_score(labels_all, predictions, average='binary', pos_label=1)
	f1_right = f1_score(labels_all, predictions, pos_label=1, average='binary')
	'''
	#cm = confusion_matrix(labels_all, predictions)
	#print(cm)

	return loss, [acc] #, precision_left, recall_left, f1_left, precision_right,  recall_right, f1_right]


def log_write(epoch, train_loss, train_metric, val_loss, val_metric, test_loss, test_metric, lr, counter):
	string='Validation update ->  '+str(counter)+\
	'| Epoch ->  '+str(epoch+1)+\
	'| Train ->  Loss: '+ str(train_loss.item())+\
	', Acc : '+ str(train_metric[0])+\
	'| Validation -> Loss : '+ str(val_loss.item())+\
	', Acc : '+ str(val_metric[0])+\
	'| Test -> Loss : '+ str(test_loss.item())+\
	', Acc : '+ str(test_metric[0])+\
	'| LR -> '+ str(lr)

	'''	
	+\
	', Precision left : '+ str(test_metric[1])+\
	', Recall left : '+ str(test_metric[2])+\
	', F1score left : '+ str(test_metric[3])+\
	', Precision right : '+ str(test_metric[4])+\
	', Recall right : '+ str(test_metric[5])+\
	', F1score right : '+ str(test_metric[6])
	'''

	print(string)

	with open(log_file_path, 'a') as file:
		file.write(string+'\n') 


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# data loaders for train , validation, test 
trainDataset = custom_dataset(path = train_list_split, source='list', window_size=window)
trainDataloader = DataLoader(trainDataset,batch_size=batch_size, shuffle=True, num_workers=4)

testDataset = custom_dataset(path = test_list_split, source='list', window_size=window)
testDataloader = DataLoader(testDataset,batch_size=batch_size, shuffle=False, num_workers=4)

valDataset = custom_dataset(path = val_list_split, source='list', window_size=window)
valDataloader = DataLoader(valDataset,batch_size=batch_size, shuffle=False, num_workers=4)


# creating a model 
if model_name=='fgnet':
	net = EEGNet_fgnet(n_classes,Chans=channel, Samples=window).cuda()
elif model_name=='fgnet_v7':
	net = EEGNet_fgnet_v7(n_classes,Chans=channel, Samples=window).cuda()
elif model_name=='channel_gated_final':
	net = EEGNet_proposed_channel_gated_final(n_classes,Chans=channel, Samples=window).cuda()
elif model_name=='temporal_gated_final':
	net = EEGNet_proposed_temporal_gated_final(n_classes,Chans=channel, Samples=window).cuda()
elif model_name=='fgnet_msfnot':
	net = EEGNet_fgnet_msfnot(n_classes,Chans=channel, Samples=window).cuda()
elif model_name=='eegnet':
	net = EEGNet_v2(n_classes,Chans=channel, Samples=window).cuda()
elif model_name=='eegnetfusion':
	net = EEGNet_fusion(n_classes,Chans=channel, Samples=window).cuda()
elif model_name=='mieegnet':
	net = MI_EEGNet(n_classes,Chans=channel, Samples=window).cuda()
elif model_name=='tsseffnet':
	net = TS_SEFFNet(n_classes,in_chans=channel).cuda()   # , Samples=window this is not a parameter in tss method
elif model_name=='deep': # repeated one
	net = DeepConvNet(n_classes=n_classes,input_shape=[batch_size,1,n_channels,window],first_conv_length=10,block_out_channels=[25, 25, 50, 100, 200], pool_size=3, last_dim=1400).cuda()
elif model_name=='shallow':  # repeated one
	net = ShallowConvNet(n_classes=n_classes,input_shape=[batch_size,1,n_channels,window], F1=40, T1=25, F2=40, P1_T=75, P1_S=15, drop_out=0.5,last_dim=58560).cuda()
elif model_name=='lmda':
	net = LMDA(num_classes=n_classes, chans=channel, samples=window).cuda()
else:
	print('wrong model name')
	sys.exit(exitCodeYouFindAppropriate)

'''
if model_name=='proposed_aux':
	net = EEGNet_proposed_aux(n_classes,Chans=channel, Samples=window).cuda()
elif model_name=='proposed_temporal_gated_aux':
	net = EEGNet_proposed_temporal_gated_aux(n_classes,Chans=channel, Samples=window).cuda()
'''

#net.weights_init()
#net.load_state_dict(torch.load('best_val_loss_weight.pth'))

#net = EEGNet(n_classes).cuda()
#net.weights_init()

#net = EEGNet_fusion(n_classes,Chans=64, Samples=window).cuda()
#net.weights_init()


# build the hyper parameters 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay, betas=(beta1, beta2))

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=factor, patience=patience)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1, last_epoch=- 1, verbose=False)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=no_epoch)

best_val_loss=9999
best_test_acc=0
counter = 0


# resume traning 
if resume_train:
	#checkpoint = torch.load(os.path.join(model_save_path,str(fold),model_name,'current.pth'))
	checkpoint = torch.load(os.path.join(model_save_path,model_name,str(subject),str(fold),'current.pth'))
	net.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	epoch_resume = checkpoint['epoch']
	loss = checkpoint['loss']
	best_val_loss=checkpoint['best_val_loss']
	best_test_acc = checkpoint['best_test_acc']

for epoch in range(no_epoch):  # loop over the dataset multiple times

	if resume_train and epoch<epoch_resume+1:
		print('epoch resume', epoch_resume)
		continue

	net.train()

	print("\nEpoch ", epoch+1,' ==>')

	running_loss = 0.0

	for i, (inputs, labels) in tqdm(enumerate(trainDataloader), total=len(trainDataloader)):

		#inputs = inputs.permute(0,2,1).unsqueeze(1)
		#print(inputs.shape)
		inputs = inputs.unsqueeze(1)		
		inputs = inputs.type(torch.FloatTensor).cuda()
		labels = labels.cuda()#-1
		inputs, labels = Variable(inputs.cuda(0)), Variable(labels.cuda(0))
		#print(torch.max(inputs), torch.min(inputs))

		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = net(inputs)  #, aux
		#print(outputs.shape,labels)
		loss = criterion(outputs, labels)
		#l1_lambda = 0.001
		#l1_norm = sum(torch.linalg.norm(p.squeeze(dim=-1).squeeze(dim=-1), 1) for p in net.parameters())
		#loss = loss + l1_lambda * l1_norm
		#aux_loss = criterion(aux, labels)# + criterion(aux2, labels)
		#loss= loss +0.1*aux_loss#+ 0.25*aux_loss
		loss.backward()
		optimizer.step()
		#print(F.softmax(outputs))
		running_loss += loss.data

	train_loss = running_loss*1.0/len(trainDataloader)
	_, train_metric = evaluate(net, trainDataloader,criterion)

	val_loss, val_metric = evaluate(net, valDataloader, criterion)

	# Note that step should be called after validate()
	scheduler.step(val_loss)
	#scheduler.step()

	if save_all_weights:
		torch.save(net.state_dict(), os.path.join(model_save_path,model_name,str(subject),str(fold),str(epoch+1)+'.pth'))

	# save checkpoint after each epoch
	torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss, 
            'LR': optimizer.param_groups[0]['lr'], 
            'best_val_loss':best_val_loss,
            #'best_test_acc' :best_test_acc }, os.path.join(model_save_path,str(fold),model_name,'current.pth'))
            'best_test_acc' :best_test_acc }, os.path.join(model_save_path,model_name,str(subject),str(fold),'current.pth'))

	# best weight on the basis of validation loss 
	if val_loss<best_val_loss:
		print('update val')
		best_val_loss= val_loss
		#torch.save(net.state_dict(), os.path.join(model_save_path,str(fold),model_name,'best_val_loss_weight.pth'))
		torch.save(net.state_dict(), os.path.join(model_save_path,model_name,str(subject),str(fold),'best_val_loss_weight.pth'))
		counter= counter+1

	test_loss, test_metric = evaluate(net, testDataloader,criterion)


	# best weight on the basis of test accuracy (cheating)
	if test_metric[0]>best_test_acc:
		print('Test accuracy improves from ', best_test_acc, ' to', test_metric[0])
		best_test_acc= test_metric[0]
		#torch.save(net.state_dict(), os.path.join(model_save_path,str(fold),model_name,'best_test_acc_weight.pth'))
		torch.save(net.state_dict(), os.path.join(model_save_path,model_name,str(subject),str(fold),'best_test_acc_weight.pth'))

	log_write(epoch, train_loss, train_metric, val_loss, val_metric, test_loss, test_metric, optimizer.param_groups[0]['lr'], counter)

