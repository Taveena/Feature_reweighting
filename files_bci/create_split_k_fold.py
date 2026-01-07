import os 
import random 
import argparse

parser = argparse.ArgumentParser(description='Requirment of the progarm to get executed')
parser.add_argument('--subject', type=str, default='A01', 
                    help='subject name')

args = parser.parse_args()
subject = args.subject
#write_split='./split_physionet'
#read_data='./../physionet_data_4s'
write_split='./split_bci_5fold/'+subject
read_data='./bci_iv_2a_data/'+subject

if not os.path.exists(write_split):
	os.mkdir(write_split)

# def dump_file(data,type_,sub_name, per=0.9):
# 	print(os.path.join(write_split,sub_name))
# 	if not os.path.exists(os.path.join(write_split,sub_name)):
# 		os.mkdir(os.path.join(write_split,sub_name))
	
# 	if type_=='test':
# 		write_name =os.path.join(write_split,sub_name,sub_name+'_'+type_+'.txt')	
# 		for instance in data:
# 			string = ','.join(instance)
# 			with open(write_name, 'a') as fp:
# 				fp.write(string+'\n')
# 	if type_=='train':
# 		for i in range(10):
# 			random.shuffle(data)
# 		number_of_instance = len(data)
# 		break_point=int(number_of_instance*per)
# 		train_data=data[0:break_point]
# 		val_data =data[break_point:]
# 		write_name =os.path.join(write_split,sub_name,sub_name+'_'+'train'+'.txt')	
# 		for instance in train_data:
# 			string = ','.join(instance)
# 			with open(write_name, 'a') as fp:
# 				fp.write(string+'\n')
# 		write_name =os.path.join(write_split,sub_name,sub_name+'_'+'val'+'.txt')	
# 		for instance in val_data:
# 			string = ','.join(instance)
# 			with open(write_name, 'a') as fp:
# 				fp.write(string+'\n')
		
# from glob import glob
# directory = glob(os.path.join(read_data,"*/"))

# for dir in directory:
# 	for subdir in ['train','test']:	
# 		data=[]
# 		crawl_dir=os.path.join(dir, subdir)
# 		#print(dir, crawl_dir)
# 		sub_name = dir.split(os.sep)[2]
# 		print(dir, sub_name)
# 		for dirName, subdirList, fileList in os.walk(crawl_dir):
# 			for fname in fileList:
# 				data.append([os.path.join(dirName,fname),dirName.split(os.sep)[-1]])
# 				#print('\t%s' % os.path.join(dirName,fname),dirName.split(os.sep)[-1])
# 		dump_file(data,subdir,sub_name)



# per=[0.9,0.85] means
# (train+val):(test) = 0.9
# train:val = 0.85

def dump_file_new_5fold(data, per=[0.9,0.875]):
	if not os.path.exists(write_split):
		os.mkdir(write_split)

	number_of_instance = len(data)
	# break_pt_1 = int(number_of_instance*per[0])
	# break_pt_2 = int(number_of_instance*per[0]*per[1])

	for i in range(10):
		random.shuffle(data)

	folds = []
	fold_len = int(number_of_instance/5)
	for i in range(0, 5):
		if i==4:
			fold = data[i*fold_len:]
		else:
			fold = data[i*fold_len:(i+1)*fold_len]
		folds.append(fold)

	for i, fold in enumerate(folds):
		test_data = fold
		temp = []
		for j in range(0,5):
			if i!=j:
				temp.append(folds[j])
		train_val_data = [item for sublist in temp for item in sublist]
		break_pt = int(len(train_val_data)*per[1])
		train_data = train_val_data[0:break_pt]
		val_data = train_val_data[break_pt:]

		test_write_name = os.path.join(write_split, str(i+1)+'_'+'test'+'.txt')
		for instance in test_data:
			string = ','.join(instance)
			with open(test_write_name, 'a') as fp:
				fp.write(string+'\n')

		train_write_name =os.path.join(write_split, str(i+1)+'_'+'train'+'.txt')	
		for instance in train_data:
			string = ','.join(instance)
			with open(train_write_name, 'a') as fp:
				fp.write(string+'\n')

		val_write_name =os.path.join(write_split, str(i+1)+'_'+'val'+'.txt')
		for instance in val_data:
			string = ','.join(instance)
			with open(val_write_name, 'a') as fp:
				fp.write(string+'\n')


def dump_file_new_10fold(data, per=[0.9,0.85]):
	if not os.path.exists(write_split):
		os.mkdir(write_split)

	number_of_instance = len(data)
	# break_pt_1 = int(number_of_instance*per[0])
	# break_pt_2 = int(number_of_instance*per[0]*per[1])

	for i in range(10):
		random.shuffle(data)

	folds = []
	fold_len = int(number_of_instance/10)
	for i in range(0, 10):
		if i==9:
			fold = data[i*fold_len:]
		else:
			fold = data[i*fold_len:(i+1)*fold_len]
		folds.append(fold)

	for i, fold in enumerate(folds):
		test_data = fold
		temp = []
		for j in range(0,10):
			if i!=j:
				temp.append(folds[j])
		train_val_data = [item for sublist in temp for item in sublist]
		break_pt = int(len(train_val_data)*per[1])
		train_data = train_val_data[0:break_pt]
		val_data = train_val_data[break_pt:]

		test_write_name = os.path.join(write_split, str(i+1)+'_'+'test'+'.txt')
		for instance in test_data:
			string = ','.join(instance)
			with open(test_write_name, 'a') as fp:
				fp.write(string+'\n')

		train_write_name =os.path.join(write_split, str(i+1)+'_'+'train'+'.txt')	
		for instance in train_data:
			string = ','.join(instance)
			with open(train_write_name, 'a') as fp:
				fp.write(string+'\n')

		val_write_name =os.path.join(write_split, str(i+1)+'_'+'val'+'.txt')
		for instance in val_data:
			string = ','.join(instance)
			with open(val_write_name, 'a') as fp:
				fp.write(string+'\n')

	# train_data = data[0:break_pt_2]
	# val_data = data[break_pt_2:break_pt_1]
	# test_data = data[break_pt_1:]

	# test_write_name = os.path.join(write_split,sub_name,sub_name+'_'+'test'+'.txt')
	# for instance in test_data:
	# 	string = ','.join(instance)
	# 	with open(test_write_name, 'a') as fp:
	# 		fp.write(string+'\n')

	# train_write_name =os.path.join(write_split,sub_name,sub_name+'_'+'train'+'.txt')	
	# for instance in train_data:
	# 	string = ','.join(instance)
	# 	with open(train_write_name, 'a') as fp:
	# 		fp.write(string+'\n')

	# val_write_name =os.path.join(write_split,sub_name,sub_name+'_'+'val'+'.txt')	
	# for instance in val_data:
	# 	string = ','.join(instance)
	# 	with open(val_write_name, 'a') as fp:
	# 		fp.write(string+'\n')
		
from glob import glob
directory = glob(os.path.join(read_data,"*/"))
complete_data = []
for dir in directory:

	#for subdir in ['train','test']:	
	# data=[]
	#crawl_dir=os.path.join(dir, subdir)
	#print(dir) #, crawl_dir)
	#input('halt')
	#label = dir.split(os.sep)[-1]
	#print(dir, label)
	#input('halt')
	for dirName, subdirList, fileList in os.walk(dir):
		#print('dirName', dirName)
		#input('halt')
		for fname in fileList:
			complete_data.append([os.path.join(dirName,fname),dirName.split(os.sep)[-1]])
			#print(fname)
			#print('\t%s' % os.path.join(dirName,fname),dirName.split(os.sep)[-2])
			#input('halt')
dump_file_new_5fold(complete_data)
