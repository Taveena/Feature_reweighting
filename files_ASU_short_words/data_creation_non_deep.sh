#!/usr/bin/env bash

declare -a arr=("fbcsp")
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
			for l in ${partition[@]};
			do
				DATA_LIST='./../ASU_dataset/Short_words_dataset_sampled/split_asu_5fold_train_test_val_sub'$j'/'$k'_'$l'.txt'
				DATA_SAVE_PATH='data_asu_non_deep/'$j'/'$k'/'$l'_data.npy'
				DATA_LABEL_SAVE_PATH='data_asu_non_deep/'$j'/'$k'/'$l'_label.npy'

				python data_creation_non_deep.py --data_list $DATA_LIST --data_save_path $DATA_SAVE_PATH --data_label_save_path $DATA_LABEL_SAVE_PATH
			done
		done
	done
done
