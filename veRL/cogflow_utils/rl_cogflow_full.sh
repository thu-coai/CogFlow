set -x

####################################
# You can change these environment variables to match your environment
export TMPDIR="/path/to/tmp" # absolute path to tmp file
MODEL_PATH=/path/to/sft_model # path to SFT model
REWARD_MODEL_PATH=/path/to/reward/model # path to preference reward model
MODEL_BRIEF_NAME=model_name # model name, used in log
####################################

export VLLM_ATTENTION_BACKEND=XFORMERS

TRAIN_DATASET_FILES=data_cogflow/cogflow_train.parquet
VAL_DATASET_FILES=data_cogflow/cogflow_test.parquet
REWARD_FUNCTION_FILE=custom_reward_full.py
PROJECT_NAME=verl_grpo_cog_flow
RUN_NAME=${MODEL_BRIEF_NAME}_cogflow_full
SRC_USE_CUSTOM_REWARD="[cog_flow]"

# You can tune these hyperparameters for better performance
N_GPUS_PER_NODE=4
N_NODES=1

ROLLOUT_TENSOR_PARALLEL_SIZE=4
TRAIN_ROLLOUT_N=6
TRAIN_BATCH_SIZE=24
ACTOR_PPO_MINI_BATCH_SIZE=4

ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU=3
ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=9
REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=9
RM_MICRO_BATCH_SIZE_PER_GPU=9

VAL_ROLLOUT_N=1

export TENSORBOARD_DIR=./tensorboard_log/${PROJECT_NAME}/${RUN_NAME}

RAY_TMP_DIR="./ray_tmp" \
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    \
    data.train_files=$TRAIN_DATASET_FILES \
    data.val_files=$VAL_DATASET_FILES \
    data.prompt_key=prompt \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=1024 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    data.return_raw_input_ids=True \
    data.shuffle=True \
    \
    data.custom_cls.path=null \
    data.custom_cls.name=null \
    \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ACTOR_PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ACTOR_PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.strategy=fsdp_3 \
    critic.strategy=fsdp_3 \
    \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$ROLLOUT_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TENSOR_PARALLEL_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=$TRAIN_ROLLOUT_N \
    actor_rollout_ref.rollout.max_num_batched_tokens=6144 \
    actor_rollout_ref.rollout.val_kwargs.n=$VAL_ROLLOUT_N \
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.9 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$REF_LOG_PROB_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    custom_reward_function.path=$REWARD_FUNCTION_FILE \
    custom_reward_function.name=compute_score \
    \
    reward_model.enable=True \
    reward_model.model.path=$REWARD_MODEL_PATH \
    reward_model.reward_manager=naive2 \
    reward_model.max_length=6144 \
    reward_model.micro_batch_size_per_gpu=$RM_MICRO_BATCH_SIZE_PER_GPU \
    reward_model.strategy=fsdp_3 \
    reward_model.model.input_tokenizer=$MODEL_PATH \
    +reward_model.src_use_custom_reward=$SRC_USE_CUSTOM_REWARD \
    +reward_model.remove_char=['*'] \
    \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$RUN_NAME \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=$N_NODES \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=4 $@ \
    trainer.resume_mode=disable \
    # trainer.val_before_train=False \
    # +trainer.val_only=True \
    # trainer.default_hdfs_dir=experiments/grpo 