#!/bin/bash

for dataset in iwslt15_envi iwslt15_vien; do
  echo dataset=$dataset
  plot_name=$dataset
  result_folder=../results/kd_encoding/$dataset/
  if [ $dataset == iwslt15_envi ]; then
    vocab_size=17191
    emb_size=512
    tradeoff_perfcut=11
    tradeoff_baseperf=25.4
  elif [ $dataset == iwslt15_vien ]; then
    vocab_size=7709
    emb_size=512
    tradeoff_perfcut=11
    tradeoff_baseperf=23.0
  fi

  python ../../core/parse_result_varying_kd.py \
    --task=nmt \
    --root=$result_folder/ \
    --vocab_size=$vocab_size \
    --emb_size=$emb_size \
    --plot_name=$plot_name \
    --heatmap \
    --tradeoff \
    --tradeoff_perfcut=$tradeoff_perfcut \
    --tradeoff_baseperf=$tradeoff_baseperf \
    # --tradeoff_logscale
done
