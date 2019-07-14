Differentiable embedding compression with KD codes
================================================================================

This is code for [our paper](https://arxiv.org/abs/1908.xxxx) on compressing the embedding table with end-to-end learned KD codes via differentiable product quantization.

## Requirements

* *[tensorflow](https://github.com/rusty1s/pytorch_geometric/releases)*

This code was developed under tensorflow version 1.12.0 and python 2. So if it doesn't work for you, you may want to install the right version.


## Run the experiments

Cd to scripts/ subfolder in specific task folders (i.e. one of lm, nmt, text_classification).

Run the original full embedding baseline using the following command:
```
./run_fullembs.sh
```
Or run the kd code based method using the other command:
```
./run_kdq.sh
```

For text classification datasets (other than ag_news), please download them from [this link](https://www.dropbox.com/s/8k7whejju4a8w7d/text_classification_kdq.zip?dl=0), and put all subfolders of datasets in the text_classification/data folder.

## Cite

Please cite [our paper](https://arxiv.org/abs/1908.xxxx) if you find it helpful in your own work:

```
@article{kdq2019,
  title={Differentiable Product Quantization for End-to-End Embedding Compression,
  author={Ting Chen, Yizhou Sun}
  journal={CoRR},
  volume={abs/1908.xxxx},
  year={2019},
}
```

## Acknowledgement

The language model is modified from [tenorflow's ptb tutorial](https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb), and NMT model is modified from [tensorflow/nmt](https://github.com/tensorflow/nmt). We would like to thank the original creators of these models.
