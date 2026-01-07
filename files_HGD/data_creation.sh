#!/usr/bin/env bash


for SUBJECT_ID in {2..14}
do
	python evaluate_channel.py --subject_id $SUBJECT_ID

done
