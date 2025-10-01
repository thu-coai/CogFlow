import json
import os
import logging
import random
import copy
import hashlib
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import argparse

# 假设这些是你本地的工具和模板文件
from prompt_utils.utils_4 import call_api_repeat, call_api_json_repeat

# --- 全局参数解析 ---
parser = argparse.ArgumentParser(description="Generate Reward Model training data from a single JSON file.")
parser.add_argument("--dataset_path", type=str, required=True, help="Path to the input JSON file containing all scenes.")
parser.add_argument("--output_path", type=str, required=True, help="Path to the output directory where dataset splits will be saved.")
parser.add_argument("--dataset_name", type=str, required=True, help="Prefix for the output dataset files and dataset_info entries.")
parser.add_argument("--dataset_info_path", type=str, required=True, help="Path to the dataset_info.json file to be created or updated.")
parser.add_argument("--cache_path", type=str, default="tmp/rm_single_file_cache.jsonl", help="Path to the cache file for API call results.")
parser.add_argument("--log_path", type=str, default="logs/rm_generation.log", help="Path to the log file.")
parser.add_argument("--start_num", type=int, default=0, help="Start index of scenes to process (inclusive).")
parser.add_argument("--end_num", type=int, default=-1, help="End index of scenes to process (inclusive). Negative values count from the end.")
parser.add_argument("--model", type=str, default="ds-v3", help="Model name to use for API calls.")
parser.add_argument("--platform", type=str, default="custom", help="Platform for API calls.")
parser.add_argument("--check_generate_time", default="no")
args = parser.parse_args()

# --- 日志记录器设置 ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_dir = os.path.dirname(args.log_path)
if log_dir:
    os.makedirs(log_dir, exist_ok=True)
file_handler = logging.FileHandler(args.log_path)
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# --- Prompt 模板定义 ---
rm_instruction_template = \
"""[Task]
Given a user query ([Input]), multiple reference responses ([Reference Responses]), and a candidate response for evaluation ([Candidate Response]).

The reference responses are given in order, and the first reference response is the best one. You should determine whether the candidate response strictly outperforms all reference responses. Thus, 0 means the candidate response is the best one, 1 means the candidate response is worse than at least one reference responses.

[Input]
{user_input}

[Reference Responses]
{reference_responses}

[Candidate Response]
{candidate_response}

[Output]
The rank of the candidate response is: """

get_constraint_template = \
"""**[Task]**
Your task is to rewrite the constraint part of a user's request based on a set of [Required Constraints]. You will be given the complete [Original User Input], which already includes some form of constraint.
You must generate a new, concise, and natural-sounding instruction that integrates the [Required Constraints] while being clearly different from the original constraint present in the [Original User Input].
**[Original User Input]**
{user_input}
**[Required Constraints]**
These are the new rules the rewritten instruction must follow:
{required_constraints}
**[Rules for Generation]**
1.  **Integrate:** The new instruction must incorporate all points from [Required Constraints].
2.  **Differentiate:** The new instruction's phrasing and style must be noticeably different from the original constraint found in the [Original User Input]. For example, if the original asks for "a JSON object", you could rephrase it to "output the result in JSON format".
3.  **No Answers:** The instruction must not hint at the answer to the user's question.
4.  **Concise Output:** Your entire output should be ONLY the new instruction text. Do not add explanations or extra words. It should be ready to be used directly as a new constraint for the user.
**[New Instruction Output]**
"""

check_validity_template = \
"""## **[Task]**

Please check the scenario, question, and constraint given in the [User Input], determine in order whether the following conditions are met, and provide your response following the output requirements in [Check Output Format].

1. Please check if the question is relevant to the scenario. For example, the content involved must be mentioned in the scenario.
2. Please check if the constraint does not imply the answer to the question, but only provides formatting content or restates the content of the question.
3. Please check if the constraint does not contain confusing content. The constraint must be a reasonable format restriction for someone answering the question. For example, the act of "requiring in the constraint not to imply the answer" does not meet the requirement.

## **[User Input]**

{user_input}

## **[Check Output Format]**

Please use JSON format for the output, with only one key named 'result', and the value being a boolean type. true indicates that all requirements are met, and false indicates that at least one requirement is not met. Please follow the structure below:

```json
{{
	"result": true / false
}}
```"""


def check_validity(args, user_input: str) -> bool:
    extracted_data = {"user_input": user_input}
    prompt = check_validity_template.format(
        extracted=json.dumps(extracted_data, ensure_ascii=False, indent=2),
    )
    validity = call_api_json_repeat(prompt, args, format_dict={"result": True}, format_level=1, brace_type='{')
    return validity.get("result", False)

def generate_new_constraint_prompt(user_input: str, mutation_type: str) -> str:
    logger.info(f"Generating new constraint for mutation type: {mutation_type}")
    new_constraint_requirements = []
    if mutation_type == "structure":
        new_constraint_requirements.append(random.choice([
            "The output should be formatted in a structure that is different from the original response. For example, JSON->Markdown or YAML->HTML.",
            "The output should be formatted in the original structure but with WRONG key names. For example, if the original format is JSON, the new format should be JSON but with wrong key names.",
        ]))
    elif mutation_type == "verbosity":
        current_verbosity = "medium"
        if any(kw in user_input.lower() for kw in ["detailed", "long"]):
            current_verbosity = "high"
        elif any(kw in user_input.lower() for kw in ["brief", "short", "concise"]):
            current_verbosity = "low"
        verbosity_options = {
            "high": "The output should be of high verbosity, which means highly detailed and long.",
            "medium": "The output should be of medium verbosity, which means balancing between brief and detailed.",
            "low": "The output should be of low verbosity, which means brief, concise and super short.",
        }
        available_options = [v for k, v in verbosity_options.items() if k != current_verbosity]
        if available_options:
            new_constraint_requirements.append(random.choice(available_options))
    new_constraint_text = "\n* ".join(new_constraint_requirements)
    prompt = get_constraint_template.format(user_input=user_input, required_constraints=new_constraint_text)
    new_instruction = call_api_repeat(prompt, args, prepare_for_json=False)
    if isinstance(new_instruction, dict):
        new_instruction = new_instruction.get('content', user_input)
    return new_instruction

def mutate_quality_prompt(user_input: str) -> str:
    mutate_quality_template = "Please generate a response to the following input that is factually incorrect, logically flawed, or incomplete.\n\nInput:\n{user_input}"
    return mutate_quality_template.format(user_input=user_input)

def process_scenes_from_main_file(fp: str, start_num: int, end_num: int) -> list:
    """从单个主JSON文件中读取、处理并生成所有训练数据对"""
    try:
        with open(fp, "r", encoding="utf-8") as f:
            all_scenes_full = json.load(f)

        if not all_scenes_full:
            logger.info("No data in dataset")
            return []
        
        if end_num < 0:
            end_num += len(all_scenes_full)
        
        all_scenes = all_scenes_full[start_num : end_num + 1]
        logger.info(f"Processing scenes from index {start_num} to {end_num} (inclusive). Total scenes in slice: {len(all_scenes)}")
        
        all_datapoints = []
        for scene in all_scenes:
            user_input = scene.get("input", "")
            responses = scene.get("reference_responses_from_diverse_source", [])
            if not user_input or not responses:
                continue
            for item in responses:
                item['reasoning_and_response'] = item['reasoning_and_response'].split('</think>')[-1]
            responses.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            def format_data_point(ref_responses_list, candi_response_obj, label, need_mutate: str | None):
                ref_responses_formatted = [{'rank': i + 1, 'content': resp['reasoning_and_response']} for i, resp in enumerate(ref_responses_list)]
                candi_response_text = candi_response_obj['reasoning_and_response']
                if len(candi_response_text) < 5:
                    logger.warning(f"Candidate response too short, skipping.")
                    return
                get_mutate_prompt_func = None
                if need_mutate:
                    if need_mutate == "structure": get_mutate_prompt_func = lambda ui=user_input: generate_new_constraint_prompt(ui, "structure")
                    elif need_mutate == "quality": get_mutate_prompt_func = lambda ui=user_input: mutate_quality_prompt(ui)
                    elif need_mutate == "verbosity": get_mutate_prompt_func = lambda ui=user_input: generate_new_constraint_prompt(ui, "verbosity")
                all_datapoints.append({
                    "instruction": "", "label": label,
                    "addition": {
                        "user_input": user_input, "reference_responses": ref_responses_formatted,
                        "candidate_response": candi_response_text, "mutate_need": need_mutate,
                        "get_mutate_prompt_func": get_mutate_prompt_func, "instruction_template": rm_instruction_template,
                    }
                })

            if len(responses) >= 4:
                num_pairs_per_scene = 8
                scheduled = ["quality"]
                if any(kw in user_input.lower() for kw in ["json", "format", "yaml", "markdown"]): scheduled.append("structure")
                if any(kw in user_input.lower() for kw in ["verbosity", "detailed", "brief"]): scheduled.append("verbosity")
                for _ in range(num_pairs_per_scene):
                    indices = sorted(random.sample(range(len(responses)), 4))
                    if len(scheduled) > 0:
                        mutate_type = scheduled.pop(0)
                        candi_idx = random.choice(indices)
                        format_data_point([responses[i] for i in indices if i != candi_idx], responses[candi_idx], 1, mutate_type)
                    else:
                        if random.random() < 0.5:
                            format_data_point([responses[i] for i in indices[:-1]], responses[indices[-1]], 1, None)
                        else:
                            format_data_point([responses[i] for i in indices[1:]], responses[indices[0]], 0, None)
        
        logger.info(f"Successfully processed {len(all_scenes)} scenes, generated {len(all_datapoints)} data points.")
        return all_datapoints
    except Exception as e:
        logger.exception(f"An error occurred while processing {fp}: {e}")
        return []

def get_hash_name(ori_dict: dict) -> str:
    return hashlib.md5(json.dumps(ori_dict, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()

def fill_mutate_requests(all_data: list[dict], cache_file_path: str) -> list[dict]:
    cache_dir = os.path.dirname(cache_file_path)
    if cache_dir: os.makedirs(cache_dir, exist_ok=True)
    storage = {}
    if os.path.exists(cache_file_path):
        with open(cache_file_path, "r") as f:
            for line in f:
                try:
                    item = json.loads(line)
                    storage[item['hash']] = item['data']
                except json.JSONDecodeError:
                    logger.warning(f"Could not decode line in cache file: {line}")
    lock = multiprocessing.Lock()
    def process_data_point(data: dict):
        if not data['addition']['mutate_need']: return
        request_to_hash = {"user_input": data['addition']['user_input'], "old_response": data['addition']['candidate_response'], "mutate_type": data['addition']['mutate_need']}
        item_hash = get_hash_name(request_to_hash)
        with lock:
            if item_hash in storage:
                logger.info(f"Cache hit for {item_hash}")
                cached_data = storage[item_hash]
                data['addition']['candidate_response'] = cached_data['addition']['candidate_response']
                data['addition']['mutate_prompt'] = cached_data['addition']['mutate_prompt']
                return
        logger.info(f"Processing type {data['addition']['mutate_need']} with hash {item_hash}")
        get_prompt_func = data['addition']['get_mutate_prompt_func']
        prompt = get_prompt_func()
        new_candidate_response = call_api_repeat(prompt, args, prepare_for_json=False)
        if isinstance(new_candidate_response, dict):
            new_candidate_response = new_candidate_response.get('content', '')
        with lock:
            data['addition']['candidate_response'] = new_candidate_response
            data['addition']['mutate_prompt'] = prompt
            storage[item_hash] = copy.deepcopy(data)
            with open(cache_file_path, "a") as f:
                storage[item_hash]['addition'].pop('get_mutate_prompt_func', None)
                f.write(json.dumps({"hash": item_hash, "data": storage[item_hash]}, ensure_ascii=False) + "\n")
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(process_data_point, all_data))
    for data in all_data:
        instruction = data['addition']['instruction_template'].format(
            user_input=data['addition']['user_input'],
            reference_responses=json.dumps(data['addition']['reference_responses'], indent=2, ensure_ascii=False),
            candidate_response=data['addition']['candidate_response'],
        )
        data['instruction'] = instruction
        if 'get_mutate_prompt_func' in data['addition']:
            del data['addition']['get_mutate_prompt_func']
    return all_data

def post_process_and_balance_data(all_data: list, split_name: str) -> list:
    logger.info(f"Post-processing and balancing '{split_name}' split with {len(all_data)} samples.")
    mutated_data, original_worse_data, original_better_data = [], [], []
    for data in all_data:
        if data['addition'].get('mutate_need') is not None:
            assert data['label'] == 1, "Mutated data should have label 1"
            mutated_data.append(data)
        elif data['label'] == 1: original_worse_data.append(data)
        else:
            assert data['label'] == 0, "Original better data should have label 0"
            original_better_data.append(data)
    logger.info(f"[{split_name}] Original counts - Mutated(label 1): {len(mutated_data)}, Original_Worse(label 1): {len(original_worse_data)}, Original_Better(label 0): {len(original_better_data)}")
    random.shuffle(original_better_data)
    random.shuffle(original_worse_data)
    mutated_cnt = len(mutated_data)
    if mutated_cnt > 0:
        original_better_data = original_better_data[:mutated_cnt*2]
        original_worse_data = original_worse_data[:mutated_cnt]
    balanced_data = original_better_data + original_worse_data + mutated_data
    random.shuffle(balanced_data)
    logger.info(f"[{split_name}] Balanced counts - Total: {len(balanced_data)}, Mutated: {len(mutated_data)}, Original_Worse: {len(original_worse_data)}, Original_Better: {len(original_better_data)}")
    for data in balanced_data:
        if 'old_response' not in data['addition']:
             data['addition']['old_response'] = data['addition']['candidate_response']
        for key in ['mutate_need', 'mutate_prompt', 'instruction_template']:
            if key in data['addition']:
                data['addition'][key] = str(data['addition'][key])
    return balanced_data

def output_dataset(all_data: list, tar_folder: str, dataset_name: str, info_file_path: str):
    os.makedirs(tar_folder, exist_ok=True)
    full_file_path = os.path.join(tar_folder, f"{dataset_name}.json")
    info_dir = os.path.dirname(info_file_path)
    if info_dir: os.makedirs(info_dir, exist_ok=True)
    try:
        with open(info_file_path, "r") as f: config = json.load(f)
    except FileNotFoundError: config = {}
    relative_file_path = os.path.relpath(full_file_path, info_dir)
    columns = {"prompt": "instruction", "label": "label"}
    config[dataset_name] = {"file_name": relative_file_path.replace('\\', '/'), "columns": columns}
    with open(info_file_path, "w") as f:
        json.dump(config, f, indent=4)
    for data in all_data:
        data.pop('addition')
    with open(full_file_path, "w") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Successfully saved {len(all_data)} samples to {full_file_path}")
    logger.info(f"Updated dataset info in {info_file_path}")

def generate_dataset(args):
    logger.info(f"Starting dataset generation from a single file: {args.dataset_path}")
    all_initial_data = process_scenes_from_main_file(args.dataset_path, args.start_num, args.end_num)
    if not all_initial_data:
        logger.error("No data points were generated. Exiting.")
        return
    logger.info(f"Extracted {len(all_initial_data)} initial data points in total.")
    all_processed_data = fill_mutate_requests(all_initial_data, args.cache_path)
    logger.info("Filled all mutate requests.")
    random.shuffle(all_processed_data)
    train_size = int(0.8 * len(all_processed_data))
    eval_size = int(0.1 * len(all_processed_data))
    train_data = all_processed_data[:train_size]
    eval_data = all_processed_data[train_size : train_size + eval_size]
    test_data = all_processed_data[train_size + eval_size :]
    splits = {"train": train_data, "eval": eval_data, "test": test_data}
    for split_name, split_data in splits.items():
        if not split_data:
            logger.warning(f"No data for {split_name} split. Skipping.")
            continue
        logger.info(f"--- Processing {split_name} split ---")
        balanced_split_data = post_process_and_balance_data(split_data, split_name)
        
        target_folder = args.output_path
            
        output_dataset(balanced_split_data, target_folder, f"{args.dataset_name}_{split_name}", args.dataset_info_path)

if __name__ == "__main__":
    random.seed(42)
    logger.info(f"Arguments received: {args}")
    generate_dataset(args)

