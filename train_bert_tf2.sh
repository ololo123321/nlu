#!/usr/bin/env bash

data_dir="/home/datascientist/rured/json_data/"
pretrained_dir="/home/datascientist/lfs/bert_multi_cased_L-12_H-768_A-12"
model_dir="/home/datascientist/relation-extraction/model"


python train_bert_tf2.py \
    --data_dir=${data_dir} \
    --pretrained_dir=${pretrained_dir} \
    --model_dir=${model_dir} \

