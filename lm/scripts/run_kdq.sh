#!/bin/bash
cd ..
gpu=$1
echo $gpu

data_path=data/
save_root=results/

for dataset in ptb wikitext-2; do
for model in small medium large; do
for K in 2 8 32 128;do
for D in 10 25 50 ;do
for kdq_type in smx vq ; do
for kdq_share_subspace in True False; do
for additive_quantization in False; do
	if [ $model == small ]; then
		max_max_epoch=15
		max_grad_norm=3
	else
		max_max_epoch=35
		max_grad_norm=5
	fi

	# Batched run
	save_path=$save_root/kd_encoding/$dataset/m$model\_kdq$kdq_type\_sharec$kdq_share_subspace\_addi$additive_quantization/K$K\_D$D\_$(date +"%m%d_%H%M%S")

	# Temp run
	# save_path=$save_root/kd_encoding/tmp/$(date +"%m%d_%H%M%S")\_m$model\_kdq$kdq_type\_sharec$kdq_share_subspace\_addi$additive_quantization$ad/K$K\_D$D

	echo $save_path
	if [ -e $save_path ]; then
		rm -r $save_path
	fi
	mkdir -p $save_path
	CUDA_VISIBLE_DEVICES=$gpu stdbuf -oL -eL python ptb_word_lm.py --dataset=$dataset --data_path=$data_path --model=$model --save_path=$save_path --max_max_epoch=$max_max_epoch --max_grad_norm=$max_grad_norm --kdq_type=$kdq_type --K=$K --D=$D --kdq_share_subspace=$kdq_share_subspace --additive_quantization=$additive_quantization >$save_path/log 2>&1 #&
done
done
done
done
done
done
done
