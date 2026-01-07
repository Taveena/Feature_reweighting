#!/usr/bin/env bash

NO_EPOCHS=200
BATCH_SIZE=64
WINDOW=1000
CHANNEL=22
N_CLASSES=4

declare -a arr=("fgnet")

# loop over models
for i in "${arr[@]}"
do
	MODEL_NAME=$i
	# loop over subjects
	for j in {1..9}
	do
		#loop over folds
		for k in {1..10}
		do
			LOG_FILE='log_files_bci/'$MODEL_NAME'/A0'$j'/log_'$k'.txt'
			MODEL_SAVE_PATH='weights/'$MODEL_NAME'/A0'$j'/'$k
			TRAIN_LIST='split_bci/A0'$j'/A0'$j'_'$k'_train.txt'
			TEST_LIST='split_bci/A0'$j'/A0'$j'_'$k'_test.txt'
			VAL_LIST='split_bci/A0'$j'/A0'$j'_'$k'_val.txt'

			python train_fgnet.py --no_epochs $NO_EPOCHS --n_classes $N_CLASSES --batch_size $BATCH_SIZE --window $WINDOW --channel $CHANNEL --train_list $TRAIN_LIST --test_list $TEST_LIST --val_list $VAL_LIST --model_save_path $MODEL_SAVE_PATH --log_file $LOG_FILE --model_name $MODEL_NAME
		done
	done
done
