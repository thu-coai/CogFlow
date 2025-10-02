#!/bin/bash

# prepare SFT data
DATASET_PATH=../../dataset/train.json
DATASET_INFO_PATH=../data/dataset_info.json

echo "Preparing standard CogFlow SFT data..."
python prepare_cogflow_sft_data.py \
    --dataset_path $DATASET_PATH \
    --output_path ../data/cog_flow/cog_flow.json \
	--dataset_info_path $DATASET_INFO_PATH \
    --dataset_name cog_flow \
    --dataset_type cogflow 

echo "Preparing direct(no-reasoning) SFT data..."
python prepare_cogflow_sft_data.py \
    --dataset_path $DATASET_PATH \
    --output_path ../data/cog_flow/cog_flow_direct.json \
	--dataset_info_path $DATASET_INFO_PATH \
    --dataset_name cog_flow_direct \
    --dataset_type direct 

echo "Preparing distillation SFT data..."
python prepare_cogflow_sft_data.py \
    --dataset_path $DATASET_PATH \
    --output_path ../data/cog_flow/cog_flow_distillr1.json \
	--dataset_info_path $DATASET_INFO_PATH \
    --dataset_name cog_flow_distillr1 \
    --dataset_type distillr1 

echo "SFT data preparation finished."
echo "================================================================="