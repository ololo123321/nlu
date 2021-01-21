#!/usr/bin/env bash

answers_dir=/home/datascientist/relation-extraction/data/bankruptcy/test
predictions_dir=/home/datascientist/relation-extraction/data/bankruptcy/predictions
#predictions_dir=/home/datascientist/relation-extraction/data/bankruptcy/predictions_rule_based

python3 evaluate.py \
    --answers_dir=${answers_dir} \
    --predictions_dir=${predictions_dir}
