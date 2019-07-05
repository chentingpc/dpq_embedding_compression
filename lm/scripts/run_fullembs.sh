#!/bin/bash
cd ..
gpu=$1
echo GPU=$gpu

data_path=data/
save_root=results/

for dataset in ptb wikitext-2; do
for model in small medium large; do
	if [ $model == small ]; then
		max_max_epoch=15
		max_grad_norm=3
	elif [ $model == medium ]; then
		max_max_epoch=35
		max_grad_norm=5
	else
		max_max_epoch=35
		max_grad_norm=10
	fi

	save_path=$save_root/baselines/model$model\_epoch$max_max_epoch\_norm$max_grad_norm
	echo $save_path
	if [ -e $save_path ]; then
		rm -r $save_path
	fi
	mkdir -p $save_path
	CUDA_VISIBLE_DEVICES=$gpu stdbuf -oL -eL python ptb_word_lm.py --dataset=$dataset --data_path=$data_path --model=$model --save_path=$save_path --max_max_epoch=$max_max_epoch --max_grad_norm=$max_grad_norm >$save_path/log 2>&1 #&
done
done
