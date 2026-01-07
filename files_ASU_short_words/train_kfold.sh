#!/usr/bin/env bash

NO_EPOCHS=200
BATCH_SIZE=64
WINDOW=512
CHANNEL=60
N_CLASSES=3
LR=0.001

declare -a arr=("tsseffnet" "lmda") # eegnet, eegnetfusion, mieegnet, shallow, deep, tsseffnet, fgnet, channel_gated_final temporal_gated_final, temporal_gated_final, fgnet_msfnot. fgnet_3abl, fgnet_5abl, conformer, fgnet_transformer, DeformConvNeXt_orig

#declare -a arr=("ResNet50_2")

# loop over models
for i in "${arr[@]}"
do
	#loop over 
	for j in 1 3 5 6 8 12
	do	
		MODEL_NAME=$i
		for k in 1 2 3 4 5 #6 7 8 9 10
		do
			LOG_FILE='log_files_short_words/'
			MODEL_SAVE_PATH='weights_short_words/'
			TRAIN_LIST='./../ASU_dataset/Short_words_dataset_sampled/split_asu_5fold_train_test_val_sub'$j'/'$k'_train.txt'
			TEST_LIST='./../ASU_dataset/Short_words_dataset_sampled/split_asu_5fold_train_test_val_sub'$j'/'$k'_test.txt'
			VAL_LIST='./../ASU_dataset/Short_words_dataset_sampled/split_asu_5fold_train_test_val_sub'$j'/'$k'_val.txt'

			CUDA_VISIBLE_DEVICES=0 python train_kfold.py --no_epochs $NO_EPOCHS --n_classes $N_CLASSES --batch_size $BATCH_SIZE --train_list $TRAIN_LIST --test_list $TEST_LIST --val_list $VAL_LIST --model_name $MODEL_NAME --fold $k --lr $LR --channel $CHANNEL --window $WINDOW --log_file $LOG_FILE --model_save_path $MODEL_SAVE_PATH --subject $j

		done
	done
done
#  --model_save_path $MODEL_SAVE_PATH --log_file $LOG_FILE--subject $j  --channel $CHANNEL  --val_list $VAL_LIST   --early_stop --early_stop_thresh $EARLY_STOP_THRESH
