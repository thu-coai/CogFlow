#!/bin/bash


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
