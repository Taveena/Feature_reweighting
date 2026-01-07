#!/usr/bin/env bash

NO_EPOCHS=200
BATCH_SIZE=64
WINDOW=80
CHANNEL=64
N_CLASSES=4
PATIENCE=25

declare -a arr=("fgnet_transformer") # fgnet conformer

# loop over models
for i in "${arr[@]}"
do
	MODEL_NAME=$i
	#loop over folds
	for k in {1..5}
	do
		#LOG_FILE='log_files_physionet_5fold/'$MODEL_NAME'/log_'$k'.txt'
		#MODEL_SAVE_PATH='weights_5fold/'$MODEL_NAME'/'$k
		TRAIN_LIST='split_physionet_5fold/'$k'_train.txt'
		TEST_LIST='split_physionet_5fold/'$k'_test.txt'
		VAL_LIST='split_physionet_5fold/'$k'_val.txt'

		python train_fgnet_kfold.py --no_epochs $NO_EPOCHS --n_classes $N_CLASSES --batch_size $BATCH_SIZE --window $WINDOW --channel $CHANNEL --train_list $TRAIN_LIST --test_list $TEST_LIST --val_list $VAL_LIST --model_name $MODEL_NAME --patience $PATIENCE --fold $k

	done
done
# --model_save_path $MODEL_SAVE_PATH --log_file $LOG_FILE
