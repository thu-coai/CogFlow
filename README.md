# CogFlow

## Abstract

LLMs trained for logical reasoning excel at step-by-step deduction to reach verifiable answers. However, this paradigm is ill-suited for navigating social situations, which induce an interpretive process of analyzing ambiguous cues that rarely yield a definitive outcome. To bridge this gap, we introduce Cognitive Reasoning, a paradigm modeled on human social cognition. It formulates the interpretive process into a structured cognitive flow of interconnected cognitive units (e.g., *observation* or *attribution*), which combine adaptively to enable effective social thinking and responses. We then propose CogFlow, a complete framework that instills this capability in LLMs. CogFlow first curates a dataset of cognitive flows by simulating the associative and progressive nature of human thought via tree-structured planning. After instilling the basic cognitive reasoning capability via supervised fine-tuning, CogFlow adopts reinforcement learning to enable the model to improve itself via trial and error, guided by a multi-objective reward that optimizes both cognitive flow and response quality. Extensive experiments show that CogFlow effectively enhances the social cognitive capabilities of LLMs, and even humans, leading to more effective social decision-making. 

## Dataset Preparation

You should prepare the dataset first. It can be done in two ways: (1) Download our dataset, or (2) run cognitive flow simulation to generate it. 

### Download our dataset
You can directly download our dataset from (release later), and place them at `dataset/train.json`, `dataset/text.json`, then you can run the following steps of SFT， RL and test.

### Cognitive flow simulation

Related code are in `data_generation`

#### Environment Installation

```bash
conda create -n cogflow_data python=3.11
conda activate cogflow_data
pip install openai==1.109.1 zhipuai==2.1.5.20241204
pip install numpy==2.2.0
pip install filelock==3.16.1
```

#### Data Generation

##### Step 1: Prepare the dataset

Replace the placeholder file at `data_generation/dataset/reddit_raw.json` with your complete dataset. The dataset must be a JSON file containing a list of dictionaries, where each dictionary follows this structure:

- `Post Text` (str): The main text of the post.
- `Comments` (list): A list of comments associated with the post.
- `Subreddit` (str, Optional): The subreddit where the post originated. This field is used to balance the data sources.

##### Step 2: Prepare the API keys

You need to use your own API key and choose the target platform.

1. Set API Credentials: Modify the arguments for the `custom_client` object within the `data_generation/api_config.py` file to add your API key.
2. Select Platform: While other platforms are supported, you must specify your choice by changing the `platform` variable in `data_generation/run_all.py`. However, we recommend using the default platform `custom`. 
3. Verify Model Name: Important: Based on your chosen platform, you must also verify and update the corresponding model name in `data_generation/api_config.py`

##### Step 3: Run the script

Navigate to the `data_generation` directory and execute the script.
```bash
cd data_generation
bash generate_and_collect.sh
```

The full data of every social situations will be saved in the subfolders within `data_generation/result/CogFlow_ds-r1_6_added`, and the final dataset will be saved in `dataset/generated_data.json`. Our script will also automatically split the dataset into training and testing sets, with the training set saved in `dataset/train.json` and the testing set saved in `dataset/test.json`, with a 90% to 10% split ratio.

## CogFlow SFT
The code are in `Llama-Factory`. 

More details of this framework can be found in https://llamafactory.readthedocs.io/en/latest/ or https://github.com/hiyouga/LLaMA-Factory

### Environment Installation

```bash
conda create -n cogflow_sft python=3.11
conda activate cogflow_sft
pip install transformers==4.50.0 torch==2.6.0 accelerate==1.5.2 tensorboard==2.19.0 deepspeed==0.16.5 # we only tested on this version, other versions are not guaranteed to work
pip install -e ".[torch,metrics]"
pip install openai==1.109.1
```

### Prepare Dataset

Make sure the train dataset is stored in the `dataset/train.json`. 

1. Prepare the api key: api key are needed for reward model data generation. You should set it in `Llama-Factory/CogFlow_files/prompt_utils/api_config.py`.
2. run the script. Note: if you generate the dataset by yourself, you should set `START_NUM` and `END_NUM` based on your planning of the dataset. 
```bash
cd Llama-Factory/CogFlow_files
bash prepare_all_data.sh # this will prepare the cogflow SFT, distilled-r1 SFT,  direct(no reasoning) STF, and reward model tuning data at the same time
```

### SFT

Supervised fine-tuning (SFT) can be performed on base models, such as `Qwen-2.5-7B-Instruct`, using the standard Llama-Factory SFT method. This process is also used to train both the Distilled-R1 and Direct models, though they utilize different added tokens and reasoning processes.

1. Configure SFT: Prepare an SFT configuration file. You can use `Llama-Factory/CogFlow_files/qwen2.5-7b_full_sft_cogflow.yaml` as a template. You SHOULD change the `model_name_or_path` to the real path. 
2. Run SFT: Execute the SFT command using the configuration file: 
```bash
cd Llama-Factory
llamafactory-cli train CogFlow_files/qwen2.5-7b_full_sft_cogflow.yaml
```

### Preference Reward Model

For preference reward modeling, we implemented a method called `rm_class`, which attaches a classification head to the base model. This approach is designed for use in veRL and creates a `PreTrainedModelForTokenClassification`.

You can perform RM training from base models like this:
1. prepare the RM config file (example: `Llama-Factory/CogFlow_files/qwen2.5-7b_full_rm_class2.yaml` ). You SHOULD change the `model_name_or_path` to the real path. 
2. Run RM tuning: Execute the RM tuning command using the configuration file: 
```bash
cd Llama-Factory
llamafactory-cli train CogFlow_files/qwen2.5-7b_full_rm_class2.yaml
```

## CogFlow RL
The code are in `veRL`.

### Environment Installation

First, locate the correct version of flash-attention from the official repository: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.7.4.post1. We will refer to the downloaded file path as `path/to/flash_attn-xxx.whl`.

```bash
cd veRL
conda create -n cogflow_rl python=3.10
conda activate cogflow_rl
pip install transformers==4.54.0 accelerate==1.9.0 torch==2.6.0 deepspeed==0.17.2
pip install path/to/flash_attn-xxx.whl
pip install torchdata==0.11.0
pip install vllm==0.8.3
pip install -e .
pip install ray==2.43.0
pip install tensorboard==2.20.0
```

### Data

#### Quick Prepare

1. Download the full training set into `dataset/train.json` (see the README of this repository for details)
2. run the preprocessing scripts. Note: if you generate the dataset by yourself, you should set `start_num` and `end_num` based on your planning of the dataset. We recommend you use completely different data points for SFT and RL.
```bash
cd veRL/cogflow_utils
bash prepare_all_data.sh
```

#### Detailed Explaination

Before training, the raw data must be preprocessed. The raw data is expected to be in `veRL/cogflow_utils/data_cogflow/rl_cog_graph_v12_2_train` (for training) and `veRL/cogflow_utils/data_cogflow/rl_cog_graph_v12_2_eval` (for validation). The `dataset_all_prepare.py` will do this automatically. 

Different preprocessing scripts are required for different models:

- For CogFlow and Distilled-R1:
  - Run: `veRL/cogflow_utils/data_preprocess_CogFlow.py`
  - This script will generate preprocessed data at the following locations:
    - Training set: `veRL/cogflow_utils/data_cogflow/cogflow_train.parquet`
    - Validation set: `veRL/cogflow_utils/data_cogflow/cogflow_test.parquet`
- For Direct-GRPO:
  - Run: `veRL/cogflow_utils/data_preprocess_CogFlow_noreason.py`
  - This script will generate preprocessed data at the following locations:
    - Training set: `veRL/cogflow_utils/data_cogflow/direct_train.parquet`
    - Validation set: `veRL/cogflow_utils/data_cogflow/direct_test.parquet`

The structure of one pre-processed data instance is as follows:
```python
{
    "data_source": data_source,
    "prompt": [
        {
            "role": "system", 
            "content": "You are a helpful assistant. You will always think before answer. Your thought should be wrapped in <think> and </think>. " 
        }, 
        {
            "role": "user",
            "content": question,
        }
    ],
    "ability": "cog",
    "reward_model": {
        "style": "rule",
        "ground_truth": rm_instruction
    },
    "extra_info": {
        'split': split,
        'index': idx, 
        'reference': {
            'responses': [reference_responses], 
            'len_reason_short': min(curr_think_len),
            'len_reason_long': max(curr_think_len),
            'epoch': 0,
        },
        'user_input': question,
        'answer': 'no_answer',
    }
}
```

### Training

#### Quick Start

We take the full version (CogFlow) as an example. The Distilled-R1 and Direct-GRPO are similar, using scripts with suffixs `_direct` or  `_distillr1` instead of `_full`, respectively. 

- Step 1: configure the `TMPDIR`, `MODEL_PATH`, and `REWARD_MODEL_PATH` repectively at `veRL/cogflow_utils/llama31_8b_rm_cogflow.sh`. `TMPDIR` is the temporary directory for storing the training data and model checkpoints. `MODEL_PATH` is the path to the SFT model. `REWARD_MODEL_PATH` is the path to the preference reward model. 

- Step 2: configure the `TOKENIZER_MODEL` at `veRL/cogflow_utils/custom_reward_full.py`. `TOKENIZER_MODEL` is the path to the tokenizer of the SFT model. 

- Step 3: run the command:
```bash
cd veRL/cogflow_utils
bash llama31_8b_rm_cogflow_full.sh
```
Now the checkpoints are stored in `veRL/cogflow_utils/checkpoints/verl_grpo_cog_flow/llama31_8b_rm_cogflow`

- Step 4: set the path to SFT model in `converter_full.sh`, choose the target checkpoint `xxx`, and run it to convert the RL trained model (using tools provided by veRL): 
```bash
bash converter_full.sh
```
Now the model is stored in `veRL/cogflow_utils/checkpoints/verl_grpo_cog_flow/llama31_8b_rm_cogflow/global_step_xxx/model`

#### Hyperparameters Explaination

For hyperparameters that control the reward function, please modify the list `omega` in `veRL/cogflow_utils/custom_reward_full.py`. 

Control which data_source(s) use custom_reward: 
- First, set: `SRC_USE_CUSTOM_REWARD="[cog_flow]"`
- Then, add:  `+reward_model.src_use_custom_reward=$SRC_USE_CUSTOM_REWARD`

For reward model: 

* using AutoModelForTokenClassification as Reward Model:
  - `reward_model.enable=True`
  - `reward_model.reward_manager=naive2` 
  - `reward_model.strategy=fsdp_3`
  - `custom_reward_function.path=...`
  - `custom_reward_function.name=compute_score`

For Rollout method：

* Using the modified default method: 
  - `actor_rollout_ref.actor.strategy=fsdp_3`
  - `critic.strategy=fsdp_3`

## Analysis

### Automatic Test
The code for automatic test is in `test`, including two parts: (1) rollout (2) run evaluator. 

#### Environment Installation

You can simply use the `cogflow_rl` environment (See `veRL/README.md` in this respository) to run the rollout (install the `zhipuai` package if you want to use the Zhipu API), or the following simplified environment: 

```bash
cd test
conda create -n cogflow_test python=3.10
conda activate cogflow_test
pip install transformers==4.54.0 torch==2.6.0
pip install vllm==0.8.3
pip install zhipuai==2.1.5.20241204
pip install google-genai==1.34.0 # for google token counting, not necessary
pip install openai==1.109.1
```

#### Run Evaluator

1. Configure the api (Optional) : `prompt_utils/api_config.py`
2. Configure the tokenizer (Optional) : `config_tokenizer.py`, fill the tokenizer path of the model you want to use (not necessary, it will only be used when analyze the length of the output)
3. Configure the script: change the parameters in  `run_all.sh`. You SHOULD modify the `TOKENIZER_PATH`, `RM_PATH`, `MODEL_NAME`, `MODEL_BRIEF_NAME` based on your needs.
4. Run the script: `bash run_all.sh`. 

### Attention Flow Visualization
In our paper, we visualized the attention flow between the parts of the cognitive flow. We provided the pipeline to generate the attention flow visualization, see `attention_visualizer`. 

This module is used to visualize the attention weights of the CogFlow model.

#### Environment Installation

You can use the same environment `cogflow_test` modules.

#### Part 1: Attention Calculation

First, We calculate the token-token attention weights of the CogFlow model on some given example and store them in one `.npz` file. 

1. configure the `config_cog.json`. The `user_input` and `generated_response` in it is the example displayed in the paper(https://arxiv.org/abs/2509.22546). You can change them based on your need. The `model_name` and `tokenizer_name` should be the same as the model you want to use. 

2. run the following command to calculate the attention weights and store them in `some_output_directory/attention.npz` file.

```shell
conda activate cogflow_test
cd attention_visualizer
python attention_calculater.py --config config_cog.json
```

#### Part 2: Attention Visualization

Then, we visualize the attention weights based on the `attention.npz` file. 

```bash
python attention_visualizer.py --agg_method topk_topk_mean --agg_inner_top_k 10 --agg_top_k 10 --norm_method none --viz_norm_method power --viz_power 0.2  --run_folder attention_map_cog attention_data.npz
```

#### Part 3: Visualization Modification and Export

We developed a lightweight web application to interactively modify and export the visualization.
1. Open the Web Application: Simply open the `sankey_visualizer.html` file in your web browser (e.g., Google Chrome, Firefox).
2. Load Data: Click on the `Select Data File` button and choose one of the `attention_sankey_*.json` files generated in the previous step. The Sankey diagram will be rendered on the screen.
3. Edit the Visualization: You can customize the appearance of the diagram in several ways:
  - Global Chart Settings: Use the sliders and dropdowns in the `Chart Settings` panel to adjust properties like side margins, font size, and font family for the entire diagram. You can also swap the left and right columns or toggle the visibility of labels.
  - Edit Link Styles: Click on any link (the flow paths between nodes). An `Edit Link Style` panel will appear on the right. Here, you can change the link's color, opacity, width, and add a colored border.
  - Edit Node Labels: Click directly on any node's text label. An input box will appear, allowing you to rename the node. Press Enter to confirm or Escape to cancel.
4. Export Your Work: Once you are satisfied with your edits, you can export the result:
  - Export JSON: Click this button to save all your style and label changes into a new .json file. This file can be loaded back into the editor later.
  - Export SVG: Click this button to save the current view of the diagram as a high-quality, scalable SVG image file, which is ideal for publications and presentations.