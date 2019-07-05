#!/bin/bash
cd ..
gpu=$1
echo $gpu

data_path=data/
save_root=results/

for dataset in ptb ; do
for model in large; do
for K in 8 16 32 ;do
for D in 25 50 ;do
for kdq_type in smx vq ; do
for kdq_share_subspace in True; do
for additive_quantization in False; do
	max_max_epoch=25  # should be enough
	max_grad_norm=5
	lr_decay=0.7

	# Batched run
	save_path=$save_root/kd_encoding/$dataset/m$model\_kdq$kdq_type\_sharec$kdq_share_subspace\_addi$additive_quantization/K$K\_D$D\_$(date +"%m%d_%H%M%S")

	# Temp run
	# save_path=$save_root/kd_encoding/tmp/$(date +"%m%d_%H%M%S")\_m$model\_kdq$kdq_type\_sharec$kdq_share_subspace\_addi$additive_quantization$ad/K$K\_D$D

	echo $save_path
	if [ -e $save_path ]; then
		rm -r $save_path
	fi
	mkdir -p $save_path
	CUDA_VISIBLE_DEVICES=$gpu stdbuf -oL -eL python ptb_word_lm.py --dataset=$dataset --data_path=$data_path --model=$model --save_path=$save_path --max_max_epoch=$max_max_epoch --max_grad_norm=$max_grad_norm --lr_decay=$lr_decay --kdq_type=$kdq_type --K=$K --D=$D --kdq_share_subspace=$kdq_share_subspace --additive_quantization=$additive_quantization >$save_path/log 2>&1 #&
done
done
done
done
done
done
done
