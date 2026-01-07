from fbcsp import *
import linear_classifier 
import os 
import argparse
import sys
import csv
from datetime import datetime
from sklearn.metrics import confusion_matrix, cohen_kappa_score, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
#from sklearn.lda import LDA

parser = argparse.ArgumentParser(description='Requirment of the progarm to get executed')
parser.add_argument('--train_data_path', type=str, default='data_non_deep/A01/1/train_data.npy', 
                    help='the log file storage path')
parser.add_argument('--train_label_path', type=str, default='data_non_deep/A01/1/train_label.npy', 
                    help='path to save concatenated data into npy file')
parser.add_argument('--val_data_path', type=str, default='data_non_deep/A01/1/val_data.npy', 
                    help='path to save concatenated data labels into npy file')
parser.add_argument('--val_label_path', type=str, default='data_non_deep/A01/1/val_label.npy', 
                    help='the log file storage path')
parser.add_argument('--test_data_path', type=str, default='data_non_deep/A01/1/test_data.npy', 
                    help='path to save concatenated data into npy file')
parser.add_argument('--test_label_path', type=str, default='data_non_deep/A01/1/test_label.npy', 
                    help='path to save concatenated data labels into npy file')
parser.add_argument('--model_name', type=str, default='lda', 
                    help='model name for training')
parser.add_argument('--n_classes', type=int, default=4,
                    help='number of classes in the dataset')
parser.add_argument('--fold', type=int, default=1,
                    help='fold')
parser.add_argument('--subject', type=int, default=1,
                    help='subject number')
parser.add_argument('--log_file', type=str, default='log_files_short_words/', 
                    help='the log file storage path')
parser.add_argument('--config_file', type=str, default='config', 
                    help='the configuration file storage path')

args=parser.parse_args()
train_data_path=args.train_data_path
train_label_path=args.train_label_path 
val_data_path=args.val_data_path 
val_label_path=args.val_label_path
test_data_path=args.test_data_path
test_label_path=args.test_label_path 
model_name=args.model_name
n_classes = args.n_classes
fold = args.fold
subject = args.subject
log_file=args.log_file

config_file=args.config_file
configuration_file = os.path.join(config_file, model_name+'_'+str(n_classes)+'_'+'configuration.txt')

#if os.path.exists(configuration_file):
#	os.remove(configuration_file)

# create a directory to the save the log file
if not os.path.exists(log_file):
	os.mkdir(log_file)

if not os.path.exists(os.path.join(log_file,model_name)):
	os.mkdir(os.path.join(log_file,model_name))

if not os.path.exists(os.path.join(log_file,model_name,str(subject))):
	os.mkdir(os.path.join(log_file,model_name,str(subject)))
	
log_file_path = os.path.join(log_file, model_name, str(subject), 'log_5fold.txt')



# datetime object containing current date and time
now = datetime.now()
 
#print("now =", now)

# dd/mm/YY H:M:S
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
#print("date and time =", dt_string)

configuration = '==========================================\n'+\
                ' No of classes:    '+str(n_classes)+'\n'+\
                ' log file path:    '+ log_file+'\n'+\
                ' model name:       '+ model_name +'\n'+\
                ' No of folds:      '+str(fold)+'\n'+\
                ' subject number:   '+str(subject)+'\n'+\
                ' train_data_path:  '+ train_data_path +'\n'+\
                ' train_label_path: '+ train_label_path +'\n'+\
                ' test_data_path:   '+ test_data_path +'\n'+\
                ' test_label_path:  '+ test_label_path +'\n'+\
                ' val_data_path:    '+ val_data_path +'\n'+\
                ' val_label_path:   '+ val_label_path +'\n'+\
                ' Date and time:    '+ dt_string +'\n'

print('Used configuration =='+'\n'+configuration)
with open(configuration_file, 'a') as file:
	file.write(configuration) 

def log_write(model_name, subject, fold, acc, precision, recall, f1, auc_roc, kappa):
	#fields = ['Subject', 'Fold', 'Accuracy', 'Precision', 'Recall', 'F1 score', 'AUC_ROC', 'Kappa'] 
	row = [model_name, str(subject), str(fold), str(round(acc,4)*100), str(round(precision,2)), str(round(recall,2)), str(round(f1,2)), str(round(auc_roc,2)), str(round(kappa,2))] 

	with open(log_file_path, 'a') as e:
	# create the csv writer
		writer = csv.writer(e)
	# write a row to the csv file
		#writer.writerow(fields)
		writer.writerow(row)
	e.close()

def load_data(train_data_path, train_label_path, val_data_path, val_label_path, test_data_path, test_label_path):
	X_train = np.load(train_data_path).astype('float32')
	y_train = np.load(train_label_path).astype('float32')
	X_test = np.load(test_data_path).astype('float32')
	y_test = np.load(test_label_path).astype('float32')
	X_val = np.load(val_data_path).astype('float32')
	y_val = np.load(val_label_path).astype('float32')
	#print(train_data.shape)
	#train_data = np.concatenate((train_data, val_data))
	#train_label = np.concatenate((train_label, val_label))

	return X_train, y_train, X_val, y_val, X_test, y_test

# loading data
X_train, y_train, X_val, y_val, X_test, y_test = load_data(train_data_path, train_label_path, val_data_path, val_label_path, test_data_path, test_label_path)

# creating a model 
if model_name=='fbcsp_all': # fbcsp
	net = fbcsp(X_train, y_train, X_val, y_val, X_test, y_test)
	X_select_train, y_train, X_select_val, y_val, X_select_test, y_test = net.classification(X_train, y_train, X_val, y_val, X_test, y_test)
	model = linear_classifier.LinearClassifier(SVC(C=0.8,kernel='rbf', probability=True))
	validation_metrics = model.fit(X_select_train, y_train, X_select_val, y_val)
	#val_results.append(validation_metrics)

	test_metrics = model.evaluate(X_select_test, y_test)
	#test_results.append(test_metrics)
	#x_label, pred_label, y_probs = net.classification()
	#metrics = metrics(x_label, pred_label, pred_probs)

elif model_name=='lda':
	net = fbcsp(X_train, y_train, X_val, y_val, X_test, y_test)
	X_train, y_train, X_val, y_val, X_test, y_test = net.classification(X_train, y_train, X_val, y_val, X_test, y_test)
	model = linear_classifier.LinearClassifier(LinearDiscriminantAnalysis(solver='svd')) #, shrinkage='auto'
	#model = linear_classifier.LinearClassifier(LDA())
	#X_train = X_train.reshape(1,-1)
	#print('X_train shape', X_train.shape, 'after reshape', X_train.reshape(X_train.shape[:-2] + (-1,)).shape)
	validation_metrics = model.fit(X_train, y_train, X_val, y_val)
	test_metrics = model.evaluate(X_test, y_test)
else:
	print('wrong model name')
	sys.exit(exitCodeYouFindAppropriate)



log_write(model_name, subject, fold, test_metrics[0], test_metrics[1], test_metrics[2], test_metrics[3], test_metrics[4], test_metrics[5])

'''
def indices_to_one_hot(data, n_classes=4):
	"""Convert an iterable of indices to one-hot encoded labels."""
	#print('data', data)
	#data = data.astype('int32')
	targets = np.array(data).reshape(-1)
	#print('targets', targets, 'targets type', type(targets), 'targets[0]', type(targets[0]), 'n_classes', n_classes)
	one_hot_targets=np.eye(n_classes)[targets]
	#print('one_hot_targets', one_hot_targets, 'type(one_hot_targets)', type(one_hot_targets), 'shape of one_hot_targets', one_hot_targets.shape)
	return one_hot_targets

def metrics(x_label, pred_label,pred_probs):
	#print('x labels',x_label, 'datatype of x labels', type(x_label), 'shape of x label', x_label.shape)
	#print('predicted labels',pred_label, 'datatype of predicted labels', type(pred_label), 'shape of predicted  label', pred_label.shape)
	acc=accuracy_score(x_label,pred_label)
	kappa = cohen_kappa_score(x_label, pred_label)
	f1= f1_score(x_label, pred_label,average='macro')
	precision = precision_score(x_label, pred_label,average='macro')
	recall = recall_score(x_label, pred_label,average='macro')
	x_label = x_label.astype('int32').tolist()
	#x_label = [x - 1 for x in x_label]
	x_label = list(map(lambda x: x - 1, x_label))
	#print('x label after subtracting 1', x_label, 'y_label', y_label)
	x_label=np.array(x_label).reshape(-1,1)
	auc_roc = roc_auc_score(x_label, pred_probs, multi_class='ovr')
	#print('acc', acc, 'kappa', kappa, 'f1', f1, 'precision', precision, 'recall', recall, 'auc_roc', auc_roc)
	return [acc, precision, recall, f1, auc_roc, kappa]
'''

