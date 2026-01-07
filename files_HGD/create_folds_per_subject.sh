#!/usr/bin/env bash

for i in {1..14}
do
	SUBJECT='subject'$i
	python create_split_k_fold.py --subject $SUBJECT
done

