#!/usr/bin/env bash

NO_EPOCHS=200
BATCH_SIZE=64
WINDOW=1000
CHANNEL=22
N_CLASSES=4
PATIENCE=25

declare -a arr=("fgnet_msfnot") # shallow deep mieegnet

# loop over models
for i in "${arr[@]}"
do
	MODEL_NAME=$i
	# loop over subjects
	for j in {1..9}
	do
		#loop over folds
		for k in {1..5}
		do
			#LOG_FILE='log_files_bci/'$MODEL_NAME'/A0'$j'/log_'$k'.txt'
			#MODEL_SAVE_PATH='weights/'$MODEL_NAME'/A0'$j'/'$k
			TRAIN_LIST='split_bci_5fold/A0'$j'/'$k'_train.txt'
			TEST_LIST='split_bci_5fold/A0'$j'/'$k'_test.txt'
			VAL_LIST='split_bci_5fold/A0'$j'/'$k'_val.txt'

			python train_kfold.py --no_epochs $NO_EPOCHS --n_classes $N_CLASSES --batch_size $BATCH_SIZE --window $WINDOW --channel $CHANNEL --train_list $TRAIN_LIST --test_list $TEST_LIST --val_list $VAL_LIST --model_name $MODEL_NAME --patience $PATIENCE --fold $k --subject $j
		done
	done
done
#  --model_save_path $MODEL_SAVE_PATH --log_file $LOG_FILE
