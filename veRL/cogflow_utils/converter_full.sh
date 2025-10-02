MODEL_PATH=checkpoints/verl_grpo_cog_flow/some_run_name/global_step_xxx
SFT_MODEL=path/to/SFT/model
python scripts/model_merger.py --backend fsdp --tie-word-embedding --hf_model_path $SFT_MODEL --local_dir $MODEL_PATH/actor --target_dir $MODEL_PATH/model