STORE_DATA_FOLDER="./data_cogflow"
COGFLOW_TOKENIZER_PATH="/path/to/tokenizer"
DIRECT_TOKENIZER_PATH="/path/to/tokenizer"

python dataset_all_prepare.py \
	--dataset_path "../../dataset/train.json" \
	--save_path "$STORE_DATA_FOLDER/rl_cog_flow_train/rl_cog_flow_train.json" \
	--data_source "cog_flow" \
	--start_num 1000 \
	--end_num 4199

python dataset_all_prepare.py \
	--dataset_path "../../dataset/train.json" \
	--save_path "$STORE_DATA_FOLDER/rl_cog_flow_eval/rl_cog_flow_eval.json" \
	--data_source "cog_flow" \
	--start_num 4200 \
	--end_num 4599

python dataset_preprocess_CogFlow.py \
	--local_dir $STORE_DATA_FOLDER \
	--tokenizer $COGFLOW_TOKENIZER_PATH

python dataset_preprocess_CogFlow_direct.py \
	--local_dir $STORE_DATA_FOLDER \
	--tokenizer $DIRECT_TOKENIZER_PATH