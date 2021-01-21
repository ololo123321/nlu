#!/usr/bin/env bash

data_dir=/home/datascientist/relation-extraction/data/bankruptcy/v2/collection

bad_files=(
    "0006"
    "0014"
    "0023"
    "0024"
    "0044"
    "0053"
    "0063"
    "0066"
    "0067"
    "0083"
    "0097"
    "0102"
    "0109"
    "0119"
)

for f in "${bad_files[@]}"; do
    rm "${data_dir}/${f}.txt"
    rm "${data_dir}/${f}.ann"
done
