import json
import logging
import argparse
import os

from dataset_rl_template import rl_rm_instruction_template

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler("dataset_all_prepare.log")
file_handler.setLevel(logging.DEBUG)

# 定义日志格式
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# 将 Handler 添加到 Logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="../../dataset/train.json")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--data_source", type=str, default="cog_flow")
    parser.add_argument("--start_num", type=int, required=True, help="include this number")
    parser.add_argument("--end_num", type=int, required=True, help="include this number, if negative, count from the end")
    args = parser.parse_args()

    with open(args.dataset_path, "r") as f:
        data = json.load(f)
    if len(data) == 0:
        logger.error("No data in dataset")
        return
    
    output_data = []
    args.end_num = (args.end_num+len(data)) % len(data)
    args.start_num = (args.start_num+len(data)) % len(data)
    for i in range(args.start_num, args.end_num+1):
        if i >= len(data):
            break
        reference_responses = [item['reasoning_and_response'] for item in data[i]["reference_responses_from_diverse_source"] if item['source'] != 'r1']
        candidate_response = ""
        user_input = data[i]['input']
        rm_instruction = rl_rm_instruction_template.format(
            user_input=user_input,
            candidate_response=candidate_response,
            reference_responses=json.dumps(reference_responses, ensure_ascii=False)
        )
        output_data.append({
            "rm_instruction": rm_instruction,
            "user_input": user_input,
            "reference_responses": reference_responses,
            "data_source": args.data_source,
        })
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, "w") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()