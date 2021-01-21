#!/usr/bin/env bash

#data_dir=/home/datascientist/relation-extraction/data/rured/rured_fixed
#data_dir=/home/datascientist/relation-extraction/data/bankruptcy/v1
data_dir=/home/datascientist/relation-extraction/data/bankruptcy/v2

elmo_dir=/home/datascientist/my_elmo_warmed_up

#model_dir=/home/datascientist/relation-extraction/models/elmo_2bilstm128_with_custom_lr_schedule_v2
#model_dir=/home/datascientist/relation-extraction/models/elmo_nm_bankruptcy_event_v2
#model_dir=/home/datascientist/relation-extraction/models/elmo_nm_bankruptcy_event_v3
#model_dir=/home/datascientist/relation-extraction/models/elmo_nm_bankruptcy_event_v4
model_dir=/home/datascientist/relation-extraction/models/elmo_nm_bankruptcy_event_with_coref_v2_doc_level

rm -r ${model_dir}
mkdir ${model_dir}

data_dir_mnt=/data
elmo_dir_mnt=/elmo
model_dir_mnt=/output

elmo_dropout=0.3
#elmo_dropout=0.0

#use_ner_emb=0
use_ner_emb=1

#span_emb_type=0  # здесь не пробовал
span_emb_type=1

#split=0  # уровень документа
split=1  # урвоень предложений

#window=1
window=3

#num_recurrent_layers=2
#cell_name=lstm
#cell_dim=128
#rnn_dropout=0.5

#num_recurrent_layers=2
#cell_name=lstm
#cell_dim=16
#rnn_dropout=0.5

# пока это лучшие параметры
num_recurrent_layers=1
cell_name=lstm
cell_dim=32
rnn_dropout=0.5

#num_recurrent_layers=1
#cell_name=gru
#cell_dim=8
#rnn_dropout=0.5

#batch_size=32  # если на уровне предложений
batch_size=4  # если на уровне докуента

time docker run -it \
    -v $(pwd):/work \
    -v ${data_dir}:${data_dir_mnt} \
    -v ${elmo_dir}:${elmo_dir_mnt} \
    -v ${model_dir}:${model_dir_mnt} \
    -w /work \
    --runtime=nvidia \
    tf1_for_elmo_training:0.0.1 python train_elmo_tf1.py \
        --train_data_dir="${data_dir_mnt}/train" \
        --valid_data_dir="${data_dir_mnt}/valid" \
        --elmo_dir=${elmo_dir_mnt} \
        --model_dir=${model_dir_mnt} \
        --elmo_dropout=${elmo_dropout} \
        --use_ner_emb=${use_ner_emb} \
        --span_emb_type=${span_emb_type} \
        --ner_emb_dropout=0.2 \
        --merged_emb_dropout=0.1 \
        --num_recurrent_layers=${num_recurrent_layers} \
        --cell_name=${cell_name} \
        --cell_dim=${cell_dim} \
        --rnn_dropout=${rnn_dropout} \
        --epochs=300 \
        --batch_size=${batch_size} \
        --split=${split} \
        --window=${window} |& tee "${model_dir}/training.log"
