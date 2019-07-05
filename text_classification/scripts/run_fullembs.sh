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
	echo $dataset
	data_dir=$data_root/$dataset/

<<comment
	if [ $dataset == ag_news ]; then
		max_iter=1000
	elif [ $dataset == yahoo_answers ]; then
		max_iter=3500
	elif [ $dataset == dbpedia ]; then
		max_iter=1500
	elif [ $dataset == yelp_review_full ]; then
		max_iter=2000
	elif [ $dataset == yelp_review_polarity ]; then
		max_iter=1500
	fi
comment

	# Batched run
	save_path=$save_root/baselines/$(date +"%m%d_%H%M%S")\_$dataset\_bsize$batch_size\_dims$dims\_maxiter$max_iter

	# Temp run
	# save_path=$save_root/baselines/tmp/$(date +"%m%d_%H%M%S")\_$dataset\_bsize$batch_size\_dims$dims\_maxiter$max_iter

	if [ ! -e $save_path ]; then
		mkdir -p $save_path
	fi
	CUDA_VISIBLE_DEVICES=$gpu stdbuf -oL -eL python main.py --dataset=$dataset --data_dir=$data_dir --save_dir=$save_path --optimizer=$optimizer --batch_size=$batch_size --learning_rate=$learning_rate --reg_weight=$reg_weight --dims=$dims --max_iter=$max_iter --concat_maxpooling=$concat_maxpooling --hidden_layers=$hidden_layers >$save_path/log 2>&1
done
