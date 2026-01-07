#!/usr/bin/env bash



for i in {1..9}
do
	SUBJECT='A0'$i
	python create_split_k_fold.py --subject $SUBJECT
done

