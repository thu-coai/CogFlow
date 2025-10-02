STORE_DATA_FOLDER="./data_cogflow"
COGFLOW_TOKENIZER_PATH="/path/to/tokenizer"
DIRECT_TOKENIZER_PATH="/path/to/tokenizer"

python dataset_all_prepare.py \
	--dataset_path "../../dataset/train.json" \
	--save_path_train "$STORE_DATA_FOLDER/rl_cog_flow_train/rl_cog_flow_train.json" \
	--save_path_eval "$STORE_DATA_FOLDER/rl_cog_flow_eval/rl_cog_flow_eval.json" \
	--data_source "cog_flow" \
	--split_ratio "8:1"

python dataset_preprocess_CogFlow.py \
	--local_dir $STORE_DATA_FOLDER \
	--tokenizer $COGFLOW_TOKENIZER_PATH

python dataset_preprocess_CogFlow_direct.py \
	--local_dir $STORE_DATA_FOLDER \
	--tokenizer $DIRECT_TOKENIZER_PATH