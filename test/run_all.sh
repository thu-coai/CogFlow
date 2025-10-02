data_path="../dataset/test.json" # full test dataset
GPU_DEVICES=1,2,3,5
NUM_ROLLOUTS=1

LLM_RESULT_FOLDER="results_rollout" # where to store the results of VLLM inference
EVA_RESULT_FOLDER="results_evaluated" # where to store the results of comparative preference evaluation
PROMPT_EVAL_MODEL="ds-r1"
PROMPT_EVAL_PLATFORM="custom"
RM_PATH="path/to/reward/model"

# Example: run local model

MODEL_NAME="/path/to/model"
MODEL_BRIEF_NAME=YourCustomizedNameExample:llama_sft_cogflow
TOKENIZER_PATH="path/to/tokenizer"

echo "--> Running VLLM inference..."
echo "    Model Path: $MODEL_NAME"
echo "    LLM Result Folder: $LLM_RESULT_FOLDER"

python run_rollout.py \
	--model_name "$MODEL_NAME" \
	--model_brief_name "$MODEL_BRIEF_NAME" \
	--data_path "$data_path" \
	--tokenizer_name "$TOKENIZER_PATH" \
	--result_folder "$LLM_RESULT_FOLDER" \
	--tp_per_instance 2 \
	--batch_size 128 \
	--gpu_devices $GPU_DEVICES \
	--num_rollouts $NUM_ROLLOUTS \
	--resume \
	--add_think_sys

# Example: call api

MODEL_NAME=o3-mini
MODEL_BRIEF_NAME=o3-mini
MODEL_PLATFORM=custom

echo "--> Running VLLM inference..."
echo "    Model Path: $MODEL_NAME"
echo "    LLM Result Folder: $LLM_RESULT_FOLDER"

python run_rollout.py \
	--model_name "$MODEL_NAME" \
	--model_brief_name "$MODEL_BRIEF_NAME" \
	--data_path "$data_path" \
	--result_folder "$LLM_RESULT_FOLDER" \
	--tp_per_instance 2 \
	--batch_size 128 \
	--gpu_devices $GPU_DEVICES \
	--num_rollouts $NUM_ROLLOUTS \
	--resume \
	--use_api \
	--api_model $MODEL_NAME \
	--api_platform $MODEL_PLATFORM \
	--num_api_workers 32

# Example: using two-stage comparative preference evaluation

echo "--> Running comparative preference evaluation..."
echo "    LLM Result Folder: $LLM_RESULT_FOLDER"
echo "    Evaluation Result Folder: $EVA_RESULT_FOLDER"
echo "    Prompt Evaluation Model: $PROMPT_EVAL_MODEL"
echo "    Prompt Evaluation Platform: $PROMPT_EVAL_PLATFORM"

CUDA_VISIBLE_DEVICES=$GPU_DEVICES \
python run_rm.py \
	--seed 42 \
	--llm_result "$LLM_RESULT_FOLDER" \
	--eval_result_folder "$EVA_RESULT_FOLDER" \
	\
	--eval_method prompt \
	--prompt_eval_model_name $PROMPT_EVAL_MODEL \
	--platform $PROMPT_EVAL_PLATFORM \
	--perform_analysis \
	--max_workers 32 \
	--resume

# Example: using model-based comparative preference evaluation

echo "--> Running comparative preference evaluation..."
echo "    LLM Result Folder: $LLM_RESULT_FOLDER"
echo "    Evaluation Result Folder: $EVA_RESULT_FOLDER"
echo "    Model Path: $RM_PATH"

CUDA_VISIBLE_DEVICES=$GPU_DEVICES \
python run_rm.py \
	--seed 42 \
	--llm_result "$LLM_RESULT_FOLDER" \
	--eval_result_folder "$EVA_RESULT_FOLDER" \
	\
	--eval_method model \
	--model_name $RM_PATH \
	--perform_analysis \
	--batch_size 32 \
	--resume