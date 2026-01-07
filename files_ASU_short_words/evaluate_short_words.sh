#!/usr/bin/env bash

declare -a arr=( "lmda" ) # "fgnet" "shallow" "eegnet" "mieegnet" "deep" "tsseffnet"  "eegnetfusion" 

#declare -a arr=("ResNet50_2")

# loop over models
for i in "${arr[@]}"
do

	python evaluate.py --model_name $i

done
