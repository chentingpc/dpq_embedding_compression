#!/bin/bash

gpu=$1
cd ../..

data=iwslt15
for data_suffix in envi vien; do
	if [ $data_suffix == vien ]; then
	  srcdst="--src=vi --tgt=en"
	else
	  srcdst="--src=en --tgt=vi"
	fi
	data_path=nmt/data/$data
	timestamp=$(date +"%m%d_%H%M%S")
	save_root=./nmt/results/

	# Batched run
	save_path=$save_root/baselines/$data\_$data_suffix/m$model\_$(date +"%m%d_%H%M%S")

	# Temp run
	# save_path=$save_root/baselines/tmp/$(date +"%m%d_%H%M%S")\_m$model

	echo $save_path
	if [ -e $save_path ]; then
		rm -r $save_path
	fi
	mkdir -p $save_path

	CUDA_VISIBLE_DEVICES=$gpu stdbuf -oL -eL python -m nmt.nmt \
		$srcdst \
		--hparams_path=nmt/standard_hparams/iwslt15.json \
		--vocab_prefix=$data_path/vocab  \
		--train_prefix=$data_path/train \
		--dev_prefix=$data_path/tst2012  \
		--test_prefix=$data_path/tst2013 \
		--out_dir=$save_path \
		>$save_path/log 2>&1 #&
done
