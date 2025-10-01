import os
import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--dataset_info_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--dataset_type", required=True, choices=['cogflow', 'distillr1', 'direct'])
    parser.add_argument("--start_num", type=int, default=0, help="include this number")
    parser.add_argument("--end_num", type=int, default=-1, help="include this number, if negative, count from the end")
    args = parser.parse_args()

    with open(args.dataset_path, "r") as f:
        data = json.load(f)
    if len(data) == 0:
        print("No data in dataset")
        return
    
    output_data = []
    args.end_num = (args.end_num+len(data)) % len(data)
    args.start_num = (args.start_num+len(data)) % len(data)
    for i in range(args.start_num, args.end_num+1):
        if i >= len(data):
            break
        for item in data[i]['reference_responses_from_diverse_source']:
            if args.dataset_type == 'cogflow':
                if item['source'] == 'fake_variations':
                    continue
                if item['source'] == 'r1':
                    break
                output_data.append({
                    "instruction": data[i]['input'],
                    "output": item['reasoning_and_response'],
                    "system": "You are a helpful assistant. You will always think before answer. Your thought should be wrapped in <think> and </think>. "
                })
                output_data.append({
                    "instruction": data[i]['input'],
                    "output": item['reasoning_and_response'].split('</think>')[-1],
                    "system": "You are a helpful assistant. "
                })
            elif args.dataset_type == 'distillr1':
                if item['source'] != 'r1':
                    continue
                output_data.append({
                    "instruction": data[i]['input'],
                    "output": item['reasoning_and_response'].replace('<ds-r1>', '').replace('</ds-r1>', ''),
                    "system": "You are a helpful assistant. You will always think before answer. Your thought should be wrapped in <think> and </think>. "
                })
                output_data.append({
                    "instruction": data[i]['input'],
                    "output": item['reasoning_and_response'].split('</think>')[-1],
                    "system": "You are a helpful assistant. "
                })
            elif args.dataset_type == 'direct':
                if item['source'] == 'fake_variations':
                    continue
                if item['source'] == 'r1':
                    break
                output_data.append({
                    "instruction": data[i]['input'],
                    "output": item['reasoning_and_response'].split('</think>')[-1],
                    "system": "You are a helpful assistant. "
                })
    output_path = args.output_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
        
    dataset_info_path = args.dataset_info_path
    with open(dataset_info_path, "r") as f:
        dataset_info = json.load(f)
    dataset_info[args.dataset_name] = {
        "file_name": os.path.relpath(output_path, os.path.dirname(dataset_info_path)),
        "columns": {
            "prompt": "instruction",
            "response": "output",
            "system": "system"
        }
    }
    with open(dataset_info_path, "w") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
        
if __name__ == "__main__":
    main()