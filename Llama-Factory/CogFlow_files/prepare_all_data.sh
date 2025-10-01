#!/bin/bash

# prepare SFT data
DATASET_PATH=../../dataset/train.json
DATASET_INFO_PATH=../data/dataset_info.json
START_NUM=0
END_NUM=999

echo "Preparing standard CogFlow SFT data..."
python prepare_cogflow_sft_data.py \
    --dataset_path $DATASET_PATH \
    --output_path ../data/cog_flow/cog_flow.json \
	--dataset_info_path $DATASET_INFO_PATH \
    --dataset_name cog_flow \
    --dataset_type cogflow \
    --start_num $START_NUM \
    --end_num $END_NUM

echo "Preparing direct(no-reasoning) SFT data..."
python prepare_cogflow_sft_data.py \
    --dataset_path $DATASET_PATH \
    --output_path ../data/cog_flow/cog_flow_direct.json \
	--dataset_info_path $DATASET_INFO_PATH \
    --dataset_name cog_flow_direct \
    --dataset_type direct \
    --start_num $START_NUM \
    --end_num $END_NUM

echo "Preparing distillation SFT data..."
python prepare_cogflow_sft_data.py \
    --dataset_path $DATASET_PATH \
    --output_path ../data/cog_flow/cog_flow_distillr1.json \
	--dataset_info_path $DATASET_INFO_PATH \
    --dataset_name cog_flow_distillr1 \
    --dataset_type distillr1 \
    --start_num $START_NUM \
    --end_num $END_NUM

echo "SFT data preparation finished."
echo "================================================================="


RM_DATASET_PATH="../../dataset/train.json"
RM_OUTPUT_PATH="../data/cog_flow_rm"
RM_DATASET_NAME="cog_flow_rm"
RM_DATASET_INFO_PATH="../data/dataset_info.json"
RM_CACHE_PATH="tmp/rm_cache.jsonl"
RM_LOG_PATH="logs/rm_generation.log"
RM_START_NUM=0
RM_END_NUM=-1


echo "Starting Reward Model data generation..."
python prepare_cogflow_rm_data.py \
    --dataset_path $RM_DATASET_PATH \
    --output_path $RM_OUTPUT_PATH \
    --dataset_name $RM_DATASET_NAME \
    --dataset_info_path $RM_DATASET_INFO_PATH \
    --cache_path $RM_CACHE_PATH \
    --log_path $RM_LOG_PATH \
    --start_num $RM_START_NUM \
    --end_num $RM_END_NUM \
	--model ds-v3 \
	--platform custom

echo "Reward Model data generation finished."
echo "================================================================="
