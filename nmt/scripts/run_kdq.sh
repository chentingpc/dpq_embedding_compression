#!/bin/bash

gpu=$1
cd ../..

data=iwslt15
data_path=nmt/data/$data
timestamp=$(date +"%m%d_%H%M%S")
save_root=./nmt/results/

for data_suffix in envi vien; do
for K in 2 8 32 128;do
for D in 8 32 128 ;do
for kdq_type in smx vq ; do
for kdq_share_subspace in True False; do
for additive_quantization in False; do
	if [ $data_suffix == vien ]; then
	  srcdst="--src=vi --tgt=en"
	else
	  srcdst="--src=en --tgt=vi"
	fi

	# Batched run
	save_path=$save_root/kd_encoding/$data\_$data_suffix/m$model\_kdq$kdq_type\_sharec$kdq_share_subspace\_addi$additive_quantization/K$K\_D$D\_$(date +"%m%d_%H%M%S")

	# Temp run
	# save_path=$save_root/kd_encoding/tmp/$(date +"%m%d_%H%M%S")\_m$model\_kdq$kdq_type\_sharec$kdq_share_subspace\_addi$additive_quantization/K$K\_D$D

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
		--kdq_type=$kdq_type \
		--K=$K \
		--D=$D \
		--kdq_share_subspace=$kdq_share_subspace \
		--additive_quantization=$additive_quantization \
		>$save_path/log 2>&1 #&
done
done
done
done
done
done
