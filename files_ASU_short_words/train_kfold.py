from torch.utils.data import DataLoader
from torch.utils import data
#from modified_resnet_model import Resnet18, Resnet50
import torch.optim as optim

from eeg_model import *
from ShallowConvNet import *
from eeg_deepconvnet import *
from eeg_tsseffnet import *
from eeg_fgnet import *
from LMDA import *
#from eeg_channel_gated_final import *
#from eeg_temporal_gated_final import *
#from eeg_fgnet_msfnot import *
#from eeg_fgnet_3ablation import *
#from eeg_fgnet_5ablation import *
'''
from eeg_conformer import Conformer
from eeg_fgnet_transformer import EEGNet_fgnet_transformer
from ConvNeXt import ConvNeXt
from DeformConvNeXt import DeformConvNeXt

from DeformConvNeXt_orig_asu import DeformConvNeXt_orig
from DeformConvNeXt_asu import DeformConvNeXt
'''
from data_loader_eegnet_ASU import custom_dataset
import torch
import torch.nn as nn
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
#import torchvision

parser = argparse.ArgumentParser(description='Requirment of the progarm to get executed')
parser.add_argument('--batch_size', type=int, default=64, 
                    help='input batch size for training (default: 128)')
parser.add_argument('--no_epochs', type=int, default=200,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--n_classes', type=int, default=3,
                    help='number of classes in the dataset')
parser.add_argument('--channel', type=int, default=16,
                    help='number of channels of the EEG signal or the temporal width of the signal')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--model_save_path', type=str, default='weights_dans_whole_6fold', 
                    help='the saving path of the model')
parser.add_argument('--log_file', type=str, default='log_files_dans_whole_6fold', 
                    help='the log file storage path')
parser.add_argument('--config_file', type=str, default='config', 
                    help='the configuration file storage path')
parser.add_argument('--train_list', type=str, default='split_dans_img_whole_6fold/1_train.txt', 
                    help='the log file storage path')
parser.add_argument('--test_list', type=str, default='split_dans_img_whole_6fold/1_test.txt', 
                    help='the log file storage path')
parser.add_argument('--val_list', type=str, default='split_dans_img_whole_6fold/1_val.txt', 
                    help='the log file storage path')
parser.add_argument('--model_name', type=str, default='mieegnet', 
                    help='model name for training')
parser.add_argument('--save_all_weights', action='store_true', default=False,
                    help='disable the functionality of saving weight after each epoch')
parser.add_argument('--resume_train', action='store_true', default=False,
                    help='disable the functionality of saving weight after each epoch')
parser.add_argument('--fold', type=int, default=1,
                    help='fold')
parser.add_argument('--window', type=int, default=2048,
                    help='window samples')
parser.add_argument('--weight_decay', type=int, default=1e-4,
                    help='weight decay of optimizer')
parser.add_argument('--beta1', type=int, default=0.9,
                    help='beta1 value of optimizer')
parser.add_argument('--beta2', type=int, default=0.999,
                    help='beta1 value of optimizer')
parser.add_argument('--factor', type=int, default=0.05,
                    help='factor of learning rate scheduler')
parser.add_argument('--early_stop', action='store_true', default= False,
                    help='early stop true or false')
parser.add_argument('--early_stop_thresh', type=int, default=10,
                    help='early stop threshold')
parser.add_argument('--subject', type=int, default=1,
                    help='subject')


""" net config """
parser.add_argument('--width_stages', type=str, default='16,32,64') # 32,64,128
parser.add_argument('--n_cell_stages', type=str, default='3,3,1')
parser.add_argument('--stride_stages', type=tuple, default=[(2,4),(2,4),(1,1)])


args = parser.parse_args()
no_epoch=args.no_epochs #100
batch_size = args.batch_size #64
n_classes = args.n_classes #4
channel =  args.channel #22
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
window = args.window
weight_decay = args.weight_decay
beta1 = args.beta1
beta2 = args.beta2
factor = args.factor
early_stop = args.early_stop
early_stop_thresh = args.early_stop_thresh
width_stages = args.width_stages
n_cell_stages = args.n_cell_stages
stride_stages = args.stride_stages
subject = args.subject

config_file=args.config_file

if not os.path.exists(config_file):
	os.mkdir(config_file)

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

# datetime object containing current date and time
now = datetime.now()
 
#print("now =", now)

# dd/mm/YY H:M:S
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
#print("date and time =", dt_string)
date_time = now.strftime("%d_%m_%Y-%H:%M:%S")

log_file_path = os.path.join(log_file, model_name,str(subject), 'log_'+str(fold)+f'_{date_time}.txt') 

print(f"model  -------  {model_name}")
seed_n=0

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
                ' learning rate    :'+str(lr)+'\n'+\
                ' seed value       :'+str(seed_n)+'\n'+\
                ' model save path  :' +  model_save_path+'\n'+\
                ' log file path    :'+ log_file+'\n'+\
                ' save all weights :'+str(save_all_weights)+'\n'+\
                ' Resume train     :'+ str(resume_train)+'\n'+\
                ' Early stop       :'+ str(early_stop)+'\n'+\
                ' Early stop thres :'+ str(early_stop_thresh)+'\n'+\
                ' model name       :'+ model_name +'\n'+\
                ' train list       :'+ train_list_split +'\n'+\
                ' test list        :'+ test_list_split +'\n'+\
                ' val list         :'+ val_list_split +'\n'+\
                ' Date and time    :'+ dt_string +'\n'

print('Used configuration =='+'\n'+configuration)
with open(configuration_file, 'a') as file:
	file.write(configuration) 

def log_write(epoch, train_loss, train_metric, val_loss, val_metric, test_loss, test_metric, lr, counter):
#def log_write(epoch, train_loss, train_metric, test_loss, test_metric, lr):
	string='Validation update ->  '+str(counter)+\
	'| Epoch ->  '+str(epoch+1)+\
	'| Train ->  Loss: '+ str(train_loss.item())+\
	', Acc : '+ str(train_metric[0])+\
	'| Val -> Loss : '+ str(val_loss.item())+\
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


def evaluate(model, loader, criterion):
	accuracy=0.0
	running_loss = 0.0
	predictions=[]
	labels_all=[]
	model.eval()
	with torch.no_grad():
		for i, (inputs, labels) in enumerate(loader): 
			if model_name=='tsseffnet':
				inputs = inputs.unsqueeze(3)
			else:
				inputs = inputs.unsqueeze(1)
			inputs = inputs.type(torch.FloatTensor).cuda()
			labels = labels.cuda()
			output = model(inputs) #
			loss = criterion(output, labels)# + 0.1*criterion(aux, labels)
			#l1_lambda = 0.001
			#l1_norm = sum(torch.linalg.norm(p.squeeze(dim=-1).squeeze(dim=-1), 1) for p in model.parameters())
			#loss = loss + l1_lambda * l1_norm
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



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# data loaders for train , validation, test 
trainDataset = custom_dataset(path = train_list_split, source='list')
trainDataloader = DataLoader(trainDataset,batch_size=batch_size, shuffle=True, num_workers=8)

testDataset = custom_dataset(path = test_list_split, source='list')
testDataloader = DataLoader(testDataset,batch_size=batch_size, shuffle=False, num_workers=8)

valDataset = custom_dataset(path = val_list_split, source='list')
valDataloader = DataLoader(valDataset,batch_size=batch_size, shuffle=False, num_workers=8)


'''
# creating a model 

if model_name=='ResNet18':
	net = Resnet18().cuda()
#elif model_name=='ResNet50':
#	net = Resnet50().cuda()
elif model_name=='ResNet50_2': # ['Org_ResNet_seed_0''ResNet50_seed_99', 'ResNet50_seed_1234'] - wrong seq
	net = torchvision.models.resnet50(pretrained=True)
	net.fc = torch.nn.Linear(net.fc.in_features, n_classes)
	net.cuda()
elif model_name in ['Org_alexnet_seed_0','Org_alexnet_seed_99','Org_alexnet_seed_1234']:
	net = torchvision.models.alexnet(pretrained=True)
	net.classifier[6] = nn.Linear(net.classifier[6].in_features, n_classes)
	net.cuda()
	print("here")
elif model_name in ['Org_googlenet_seed_0', 'Org_googlenet_seed_99', 'Org_googlenet_seed_1234']:
	net = torchvision.models.inception_v3(pretrained=True)
	net.fc = torch.nn.Linear(net.fc.in_features, n_classes)
	net.cuda()
elif model_name in ['Org_resnet18_seed_0', 'Org_resnet18_seed_99', 'Org_resnet18_seed_1234']:
	net = torchvision.models.resnet18(pretrained=True)
	net.fc = torch.nn.Linear(net.fc.in_features, n_classes)
	net.cuda()
elif model_name in ['Org_vgg18_seed_0', 'Org_vgg18_seed_99', 'Org_vgg18_seed_1234']:
	net = torchvision.models.vgg11(pretrained=True)
	net.classifier[6] = nn.Linear(net.classifier[6].in_features, n_classes)
	net.cuda()
else:
	print('wrong model name')
	sys.exit(exitCodeYouFindAppropriate)
'''


# creating a model 
if model_name=='fgnet':
	net = EEGNet_fgnet(n_classes,Chans=channel, Samples=window).cuda()
	net.weights_init()
elif model_name=='eegnet':
	net = EEGNet_v2(n_classes,Chans=channel, Samples=window).cuda()
	net.weights_init()
elif model_name=='eegnetfusion':
	net = EEGNet_fusion(n_classes,Chans=channel, Samples=window).cuda()
	net.weights_init()
elif model_name=='mieegnet':
	net = MI_EEGNet(n_classes,Chans=channel, Samples=window).cuda()
	net.weights_init()
elif model_name=='tsseffnet':
	net = TS_SEFFNet(in_chans=channel,n_classes=n_classes).cuda()   # , Samples=window this is not a parameter in tss method
	#net.weights_init()
elif model_name=='deep': # repeated one
	net = DeepConvNet(n_classes=n_classes,input_shape=[batch_size,1,n_channels,window],first_conv_length=10,block_out_channels=[25, 25, 50, 100, 200], pool_size=2, last_dim=1400).cuda()
	net.weights_init()
elif model_name=='shallow':  # repeated one
	n_channels=channel
	net = ShallowConvNet(n_classes=n_classes,input_shape=[batch_size,1,n_channels,window], F1=40, T1=25, F2=40, P1_T=75, P1_S=5, drop_out=0.5,last_dim=3320).cuda()
	net.weights_init()
elif model_name=='lmda':
	net = LMDA(num_classes=n_classes, chans=channel, samples=window).cuda()
else:
	print('wrong model name')
	sys.exit(exitCodeYouFindAppropriate)

'''
elif model_name=='fgnet_transformer':
	net = EEGNet_fgnet_transformer(n_classes,Chans=channel, Samples=window).cuda()
	net.weights_init()
elif model_name=='ConvNeXt':
	net = ConvNeXt(num_classes=n_classes,in_chans=1).cuda()
	#net.weights_init()
elif model_name=='DeformConvNeXt_orig':
	net = DeformConvNeXt_orig(n_classes=n_classes).cuda()
elif model_name == 'DeformConvNeXt':
	net = DeformConvNeXt(n_classes=n_classes).cuda()

else:
	print('wrong model name')
	sys.exit(exitCodeYouFindAppropriate)
'''

# build the hyper parameters 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4, betas=(0.5, 0.999))

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=25)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5, last_epoch = -1)
best_val_loss=9999
best_test_acc=0
counter = 0
best_epoch = 0

# resume traning 
if resume_train:
	#checkpoint = torch.load(os.path.join(model_save_path,str(fold),model_name,'current.pth'))
	checkpoint = torch.load(os.path.join(model_save_path,model_name,str(subject),str(fold),'current.pth'))
	net.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	epoch_resume = checkpoint['epoch']
	loss = checkpoint['loss']
	best_test_acc = checkpoint['best_test_acc']

for epoch in range(no_epoch):  # loop over the dataset multiple times

	if resume_train and epoch<epoch_resume+1:
		print('epoch resume', epoch_resume)
		continue

	net.train()

	print("\nEpoch ", epoch+1,' ==>')

	running_loss = 0.0

	for i, (inputs, labels) in tqdm(enumerate(trainDataloader), total=len(trainDataloader)):
		if model_name=='tsseffnet':
			inputs = inputs.unsqueeze(3)
		else:
			inputs = inputs.unsqueeze(1)		
		inputs = inputs.type(torch.FloatTensor).cuda()
		labels = labels.cuda()
		inputs, labels = inputs.cuda(0), labels.cuda(0)
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
		torch.save(net.state_dict(), os.path.join(model_save_path,model_name,str(subject),str(fold),str(epoch)+'.pth')) #str(subject),

	#if epoch == 99:
	#	torch.save(net.state_dict(), os.path.join(model_save_path,model_name,str(fold),str(epoch+1)+'.pth'))

	# save checkpoint after each epoch
	torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss, 
            'LR': optimizer.param_groups[0]['lr'], 
            #'best_val_loss':best_val_loss,
            #'best_test_acc' :best_test_acc }, os.path.join(model_save_path,str(fold),model_name,'current.pth'))
            'best_test_acc' :best_test_acc }, os.path.join(model_save_path,model_name,str(subject),str(fold),'current.pth'))#str(subject),

	# best weight on the basis of validation loss 
	
	if val_loss<best_val_loss:
		print('update val')
		best_val_loss= val_loss
		best_epoch = epoch
		#torch.save(net.state_dict(), os.path.join(model_save_path,str(fold),model_name,'best_val_loss_weight.pth'))
		torch.save(net.state_dict(), os.path.join(model_save_path,model_name,str(subject),str(fold),'best_val_loss_weight.pth'))#str(subject),
		counter= counter+1
	
	test_loss, test_metric = evaluate(net, testDataloader,criterion)


	# best weight on the basis of test accuracy (cheating)
	if test_metric[0]>best_test_acc:
		print('Test accuracy improves from ', best_test_acc, ' to', test_metric[0])
		best_test_acc= test_metric[0]
		#torch.save(net.state_dict(), os.path.join(model_save_path,str(fold),model_name,'best_test_acc_weight.pth'))
		torch.save(net.state_dict(), os.path.join(model_save_path,model_name,str(subject),str(fold),'best_test_acc_weight.pth'))#str(subject),

	#log_write(epoch, train_loss, train_metric, test_loss, test_metric, j)
	log_write(epoch, train_loss, train_metric, val_loss, val_metric, test_loss, test_metric, optimizer.param_groups[0]['lr'], counter)
	'''
	if early_stop:	 
		if epoch - best_epoch > early_stop_thresh:
			print("Early stopped training at epoch %d" % epoch)
			break  # terminate the training loop
	'''
'''
	val_loss = ''
	val_metric = ''
	print(f'lr - {optimizer.param_groups[0]["lr"]}')
	j = optimizer.param_groups[0]["lr"]
'''
