#!/bin/bash

MODEL_PATH="/home/spider/wangyunfei01/workspace/Chinese-BERT-wwm/chinese_roberta_wwm_large_ext"

python3 ./run_export.py \
    --vocab_file=$MODEL_PATH/vocab.txt \
    --bert_config_file=$MODEL_PATH/bert_config.json \
    --init_checkpoint=$MODEL_PATH/bert_model.ckpt \
    --use_tpu=False \
    --output_dir=./output_dir \
    --export_dir=./saved_model

