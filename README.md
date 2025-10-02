# Think Socially via Cognitive Reasoning

[![arXiv](https://img.shields.io/badge/arXiv-2509.22546-b31b1b.svg)](https://arxiv.org/abs/2509.22546v1)

This repository contains the official implementation for the paper **"Think Socially via Cognitive Reasoning"**. We introduce **Cognitive Reasoning**, a new paradigm modeled on human social cognition, and **CogFlow**, a framework to instill this capability in LLMs.

CogFlow enables models to navigate complex social situations by generating a structured "cognitive flow" of interconnected cognitive units (e.g., *observation*, *attribution*). This approach moves beyond rigid logical deduction, which is often ill-suited for the ambiguous and interpretive nature of social interactions.

![CogFlow Framework Overview](https://storage.googleapis.com/static.deepmind.com/prompt_gallery/CogFlow_Framework_Overview.png)
> **Figure:** The CogFlow framework, from data collection via simulation to model optimization with SFT and RL.

## 📜 Table of Contents
- [✨ Features](#-features)
- [🚀 Getting Started](#-getting-started)
  - [Environment Setup](#environment-setup)
- [💾 Dataset Preparation](#-dataset-preparation)
  - [Option 1: Download Our Dataset (Recommended)](#option-1-download-our-dataset-recommended)
  - [Option 2: Generate Dataset via Cognitive Flow Simulation](#option-2-generate-dataset-via-cognitive-flow-simulation)
- [⚙️ Training Pipeline](#️-training-pipeline)
  - [Step 1: Supervised Fine-Tuning (SFT)](#step-1-supervised-fine-tuning-sft)
  - [Step 2: Train Preference Reward Model](#step-2-train-comparative-preference-reward-model)
  - [Step 3: Reinforcement Learning (RL)](#step-3-reinforcement-learning-rl)
- [📊 Analysis & Evaluation](#-analysis--evaluation)
  - [Automatic Evaluation](#automatic-evaluation)
  - [Attention Flow Visualization](#attention-flow-visualization)
- [🎓 Citation](#-citation)
- [🙏 Acknowledgements](#-acknowledgements)

## ✨ Features

* **Cognitive Reasoning Paradigm**: A structured approach that allows LLMs to interpret and respond to social situations more effectively.
* **Cognitive Flow Simulation**: A novel data generation process using tree-structured planning to simulate the associative nature of human thought.
* **SFT + RL Framework**: A complete training pipeline that first instills foundational skills via Supervised Fine-Tuning (SFT) and then refines them using multi-objective Reinforcement Learning (RL).
* **Analysis Tools**: Includes scripts for automated evaluation and visualizing the model's internal attention mechanisms to understand the cognitive flow.

## 🚀 Getting Started

```
git clone https://github.com/thu-coai/CogFlow
cd CogFlow
```

### Environment Setup

This project requires three separate Conda environments due to differing dependencies for data generation, SFT, and RL.

1.  **For Data Generation (`cogflow_data`)**
    ```bash
    conda create -n cogflow_data python=3.11
    conda activate cogflow_data
    pip install openai==1.109.1 zhipuai==2.1.5.20241204 numpy==2.2.0 filelock==3.16.1
    ```
2.  **For SFT & Reward Model (`cogflow_sft`)**
    ```bash
    conda create -n cogflow_sft python=3.11
    conda activate cogflow_sft
    # We have only tested on these versions; others may not work as expected.
    pip install transformers==4.50.0 torch==2.6.0 accelerate==1.5.2 tensorboard==2.19.0 deepspeed==0.16.5
    cd Llama-Factory
    pip install -e ".[torch,metrics]"
    pip install openai==1.109.1
    cd ..
    ```
3.  **For Reinforcement Learning (`cogflow_rl`)**
    ```bash
    conda create -n cogflow_rl python=3.10
    conda activate cogflow_rl
    pip install transformers==4.54.0 accelerate==1.9.0 torch==2.6.0 deepspeed==0.17.2 torchdata==0.11.0 vllm==0.8.3 ray==2.43.0 tensorboard==2.20.0
    # Download the correct flash-attention wheel from [https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.4.post1](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.4.post1)
    pip install path/to/flash_attn-xxx.whl
    cd veRL
    pip install -e .
    cd ..
    ```

## 💾 Dataset Preparation

You can either download our pre-generated dataset or run the cognitive flow simulation to create your own.

### Option 1: Download Our Dataset (Recommended)

You can directly download our dataset from [Huggingface](https://huggingface.co/datasets/thu-coai/CogFlow) and place the files at `dataset/train.json` and `dataset/test.json`. You can then proceed directly to the [Training Pipeline](#️-training-pipeline).

### Option 2: Generate Dataset via Cognitive Flow Simulation

The code for simulation is located in the `data_generation` directory.

#### Step 1: Prepare Raw Data

Place your raw data file at `data_generation/dataset/reddit_raw.json`. This should be a JSON file containing a list of dictionaries, each with `Post Text` and `Comments` keys.

#### Step 2: Configure API Keys

You need an API key from a supported LLM provider to generate the data.

1.  **Set API Credentials**: Add your API key in `data_generation/api_config.py` by modifying the `custom_client` arguments.
2.  **Select Platform**: Set the `platform` variable in `data_generation/run_all.py`. We recommend using the default (`custom`).
3.  **Verify Model Name**: Ensure the model name in `data_generation/api_config.py` matches your chosen platform.

#### Step 3: Run Simulation

Navigate to the `data_generation` directory and run the script. This will generate the complete dataset and split it into training and testing sets.

```bash
cd data_generation
bash generate_and_collect.sh
```

The final datasets will be saved as `dataset/train.json` and `dataset/test.json`.

## ⚙️ Training Pipeline

The training process uses **Llama-Factory** for SFT and Reward Model training, and **veRL** for Reinforcement Learning. In our practice, we use `Qwen-2.5-7B-Instruct` and `Llama-3.1-8B-Instruct` as our base models.

### Step 1: Supervised Fine-Tuning (SFT)

This step teaches the base model the fundamental capability of cognitive reasoning. 

- **Preprocess Training Data**: Run the following script to preprocess the training data in `dataset/train.json`. It will convert and register the dataset to `Llama-Factory/data`. 
    ```bash
    cd Llama-Factory/CogFlow_files
    bash prepare_data_sft.sh
    ```
- **Prepare Config**:
  - Modify the SFT configuration file, `Llama-Factory/CogFlow_files/qwen2.5-7b_full_sft_cogflow.yaml` for `Qwen-2.5-7B-Instruct` base model and `Llama-Factory/CogFlow_files/llama3.1-8b_full_sft_cogflow.yaml` for `Llama-3.1-8B-Instruct` base model.
  - update the corresponding `model_name_or_path` to your base models' path.
- **Run SFT Training**: Execute the training command:
    ```bash
    cd Llama-Factory
    llamafactory-cli train CogFlow_files/qwen2.5-7b_full_sft_cogflow.yaml
    ```

### Step 2: Train Comparative Preference Reward Model

The reward model learns to predict human preferences, guiding the RL process.

- **Prepare Training Data**: You can prepare it in two ways: 
    - Download Our Dataset (Recommended): downloading our preprocessed reward model data from [Huggingface](https://huggingface.co/datasets/thu-coai/CogFlow) and place it in the `dataset` folder, with names `rm_train.json`, `rm_eval.json`, and `rm_test.json`. Then, run the following script to register the dataset to `Llama-Factory/data`.
        ```bash
        cd Llama-Factory/CogFlow_files
        bash prepare_data_rm_offtheshelf.sh
        ```
    - Generate Reward Model Data: Construct reward model data from the `dataset/train.json`. Before running, please configure the API key in `Llama-Factory/CogFlow_files/prompt_utils/api_config.py` (the same procedure in [Step 2](#step-2-prepare-api-keys)). Then, run the following script to register the dataset to `Llama-Factory/data`.: 
        ```bash
        cd Llama-Factory/CogFlow_files
        bash prepare_data_rm.sh
        ```
- **Prepare Config**: Use `Llama-Factory/CogFlow_files/qwen2.5-7b_full_rm_class2.yaml` as a template and update `model_name_or_path`.
- **Run RM Training**:
    ```bash
    cd Llama-Factory
    llamafactory-cli train CogFlow_files/qwen2.5-7b_full_rm_class2.yaml
    ```

### Step 3: Reinforcement Learning (RL)

RL optimizes the SFT model's ability to generate high-quality and efficient cognitive flows.

- **Prepare RL Data**: Ensure `dataset/train.json` is in the root `dataset` folder. Run the preprocessing script to prepare the data for veRL.
    ```bash
    cd veRL/cogflow_utils
    bash prepare_all_data.sh
    ```
- **Configure Training Script**: In `veRL/cogflow_utils/llama31_8b_rm_cogflow_full.sh`, set the paths for `TMPDIR`, `MODEL_PATH` (your SFT model), and `REWARD_MODEL_PATH`. Also, set `TOKENIZER_MODEL` in `veRL/cogflow_utils/custom_reward_full.py` (the tokenizer of your SFT model). 
    - Scripts with suffixs `_direct` or `_distillr1` is used to training the ablations `Direct-GRPO` and `Distilled-R1`.

- **Run RL Training**, Checkpoints will be saved in the `checkpoints` directory.:
    ```bash
    cd veRL/cogflow_utils
    bash rl_cogflow_full.sh
    ```
- **Convert Checkpoint**: After training, convert the RL checkpoint into a standard checkpoint. Set the SFT model path and choose a checkpoint in `converter_full.sh`, then run the following script. The final model will be saved in the checkpoint's `model` subdirectory.
    ```bash
    bash converter_full.sh
    ``` 

## 📊 Analysis & Evaluation

### Automatic Evaluation

The code in the `test` directory is used for automated evaluation.

1.  **Environment**: You can use the `cogflow_rl` environment.
2.  **Configuration**: Modify the parameters in `run_all.sh`, including `TOKENIZER_PATH`, `RM_PATH`, and `MODEL_NAME`.
3.  **Run Evaluation**:
    ```bash
    cd test
    bash run_all.sh
    ```

### Attention Flow Visualization

Reproduce the Sankey diagram attention visualizations from our paper. The code is in `attention_visualizer`.

1.  **Calculate Attention**:
    -   Configure `config_cog.json` with your model path and the example input/output you wish to analyze.
    -   Run the calculation script:
        ```bash
        cd attention_visualizer
        python attention_calculater.py --config config_cog.json
        ```
2.  **Generate Visualization Data**: Process the saved attention weights to create the visualization file.
    ```bash
    python attention_visualizer.py --agg_method topk_topk_mean --agg_inner_top_k 10 --agg_top_k 10 --norm_method none --viz_norm_method power --viz_power 0.2 --run_folder attention_map_cog attention_data.npz
    ```
3.  **View and Edit**: Open `sankey_visualizer.html` in a web browser. Load the generated `attention_sankey_*.json` file to view, interactively edit, and export the Sankey diagram as an SVG.

## 🎓 Citation

If you use CogFlow in your research, please cite our paper:

```bibtex
@misc{cogflow,
      title={Think Socially via Cognitive Reasoning}, 
      author={Jinfeng Zhou and Zheyu Chen and Shuai Wang and Quanyu Dai and Zhenhua Dong and Hongning Wang and Minlie Huang},
      year={2025},
      eprint={2509.22546},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.22546}, 
}
```

## 🙏 Acknowledgements

Our SFT and RL implementations are built upon the excellent [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and [veRL](https://github.com/volcengine/verl) frameworks. We thank their developers for making their work public.
