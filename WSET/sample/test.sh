#!/usr/bin/env bash

input=../../data/preprocessed/2021-10-13_16-57-51.json

python -m sample.main --input $input

dot -T png result/tree.dot -o result/tree.png