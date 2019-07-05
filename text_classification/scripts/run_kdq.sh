#!/bin/bash
cd ..
gpu=$1
data_root=data/
save_root=results/

max_iter=5000
optimizer=lazy_adam
#optimizer=adam
learning_rate=1e-3
batch_size=64
dims=300
hidden_layers=1
reg_weight=0
concat_maxpooling=False

for dataset in ag_news yahoo_answers dbpedia yelp_review_full yelp_review_polarity amazon_review_full amazon_review_polarity; do
for K in 16 32; do
for D in 50 100 ;do
for kdq_type in smx vq ; do
for kdq_share_subspace in True False; do
for additive_quantization in False; do
	echo $dataset
	data_dir=$data_root/$dataset/

	# Batched run
	save_path=$save_root/kd_encoding/$dataset/m$model\_kdq$kdq_type\_sharec$kdq_share_subspace\_addi$additive_quantization/K$K\_D$D\_$(date +"%m%d_%H%M%S")

	# Temp run
	# save_path=$save_root/kd_encoding/tmp/$(date +"%m%d_%H%M%S")\_m$model\_kdq$kdq_type\_sharec$kdq_share_subspace\_addi$additive_quantization/K$K\_D$D

	echo $save_path
	if [ -e $save_path ]; then
		rm -r $save_path
	fi
	mkdir -p $save_path
	CUDA_VISIBLE_DEVICES=$gpu stdbuf -oL -eL python main.py --dataset=$dataset --data_dir=$data_dir --optimizer=$optimizer --save_path=$save_path --max_iter=$max_iter --learning_rate=$learning_rate --batch_size=$batch_size --dims=$dims --reg_weight=$reg_weight --concat_maxpooling=$concat_maxpooling --hidden_layers=$hidden_layers --kdq_type=$kdq_type --K=$K --D=$D --kdq_share_subspace=$kdq_share_subspace --additive_quantization=$additive_quantization >$save_path/log 2>&1 #&
done
done
done
done
done
done
