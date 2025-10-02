#!/bin/bash


RM_TRAIN_PATH="../../dataset/rm_train.json"
RM_EVAL_PATH="../../dataset/rm_eval.json"
RM_TEST_PATH="../../dataset/rm_test.json"
RM_OUTPUT_PATH="../data/cog_flow_rm"
RM_DATASET_NAME="cog_flow_rm"
RM_DATASET_INFO_PATH="../data/dataset_info.json"
RM_LOG_PATH="logs/rm_generation.log"


echo "Starting Reward Model data generation..."
python prepare_cogflow_rm_data_offtheshelf.py \
    --train_path $RM_TRAIN_PATH \
    --eval_path $RM_EVAL_PATH \
    --test_path $RM_TEST_PATH \
    --output_path $RM_OUTPUT_PATH \
    --dataset_name $RM_DATASET_NAME \
    --dataset_info_path $RM_DATASET_INFO_PATH \
    --log_path $RM_LOG_PATH \

echo "Reward Model data generation finished."
echo "================================================================="
