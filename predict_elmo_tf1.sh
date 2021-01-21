#!/usr/bin/env bash

#model_dir=/home/datascientist/relation-extraction/models/elmo_2bilstm128_with_custom_lr_schedule_v2
#model_dir=/home/datascientist/relation-extraction/models/elmo_nm_bankruptcy_event_v3
model_dir=/home/datascientist/relation-extraction/models/elmo_nm_bankruptcy_event_v4

elmo_dir=/home/datascientist/my_elmo_warmed_up
data_dir=/home/datascientist/relation-extraction/data/bankruptcy/test
output_dir=/home/datascientist/relation-extraction/data/bankruptcy/predictions

rm -r ${output_dir}
mkdir ${output_dir}

model_dir_mnt=/model
elmo_dir_mnt=/elmo
data_dir_mnt=/data
output_dir_mnt=/output

batch_size=32

time docker run -it \
    -v $(pwd):/work \
    -v ${model_dir}:${model_dir_mnt} \
    -v ${elmo_dir}:${elmo_dir_mnt} \
    -v ${data_dir}:${data_dir_mnt} \
    -v ${output_dir}:${output_dir_mnt} \
    -w /work \
    --runtime=nvidia \
    tf1_for_elmo_training:0.0.1 python predict_elmo_tf1.py \
        --model_dir=${model_dir_mnt} \
        --data_dir=${data_dir_mnt} \
        --output_dir=${output_dir_mnt} \
        --batch_size=${batch_size}
