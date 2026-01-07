#!/usr/bin/env bash

declare -a arr=("fbcsp_all") # "fbcsp"
declare -a partition=("train" "test" "val")

# loop over models
for i in "${arr[@]}"
do
	MODEL_NAME=$i
	# loop over subjects
	for j in 1 3 5 6 8 12
	do
		#loop over folds
		for k in {1..5}
		do
			TRAIN_DATA_PATH='./data_asu_non_deep/'$j'/'$k'/train_data.npy'
			TRAIN_LABEL_PATH='./data_asu_non_deep/'$j'/'$k'/train_label.npy'
			TEST_DATA_PATH='./data_asu_non_deep/'$j'/'$k'/test_data.npy'
			TEST_LABEL_PATH='./data_asu_non_deep/'$j'/'$k'/test_label.npy'
			VAL_DATA_PATH='./data_asu_non_deep/'$j'/'$k'/val_data.npy'
			VAL_LABEL_PATH='./data_asu_non_deep/'$j'/'$k'/val_label.npy'
			#LOG_FILE_PATH='log_files_bci_5fold/'$MODEL_NAME'/'$j'/log_'$k'.csv'

			python train_linear.py --train_data_path $TRAIN_DATA_PATH --train_label_path $TRAIN_LABEL_PATH --test_data_path $TEST_DATA_PATH --test_label_path $TEST_LABEL_PATH --val_data_path $VAL_DATA_PATH --val_label_path $VAL_LABEL_PATH --model_name $MODEL_NAME --fold $k --subject $j
		done
	done
done
#  --log_file $LOG_FILE_PATH
