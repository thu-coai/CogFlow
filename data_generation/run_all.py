import os
import time
from utils_4 import get_time_str

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--platform', type=str, default="custom")
parser.add_argument('--main_model', type=str, default="ds-r1")
parser.add_argument('--quick_model', type=str, default="ds-v3")
parser.add_argument('--reference_models', type=str, default='ds-r1,ds-v3')
parser.add_argument('--baseline_model', type=str, default="ds-r1")
parser.add_argument('--run_instances', type=int, default=10)
parser.add_argument('--max_workers', type=int, default=4)

args = parser.parse_args()

platform = args.platform
main_model = args.main_model
quick_model = args.quick_model
reference_models = args.reference_models
baseline_model = args.baseline_model
run_instances = args.run_instances
max_workers = args.max_workers


def do_all(name, start_num, end_num, dataset_src, check_time = "yes"):
    folder_path = f"result/CogFlow_{main_model}_6"
    folder_path_2 = f"result/CogFlow_{main_model}_6_added"
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(folder_path_2, exist_ok=True)
    
    app_output_info_name = f"{name}.tmp"
    dataset_folder = "dataset/output"
    dataset_type = "json"
    
    all_names = os.listdir(folder_path_2)
    for name in all_names:
        if not name.endswith("_eval"):
            continue
        if not name.startswith(f"reddit_raw_{start_num}-{end_num}"):
            continue
        print(f"file {name} found, skip {name}, {start_num}, {end_num}")
        return
    print(f"start {name}, {start_num}, {end_num}")

    os.makedirs(os.path.dirname(app_output_info_name), exist_ok=True)
    os.system(f"\
        python scene_gen.py \
            --model \"{main_model}\" \
            --model_constraint \"{quick_model}\" \
            --dataset_src \"{dataset_src}\" \
            --start_num {start_num} \
            --end_num {end_num} \
            --app_output_info \"{app_output_info_name}\" \
            --check_generate_time \"{check_time}\" \
            --platform \"{platform}\" \
    ")
    with open(app_output_info_name, "r") as f:
        dataset_name = f.read().strip()
    os.system(f"\
        python cogflow_simulate.py\
            --dataset_name \"{dataset_name}\" \
            --data_from {0} \
            --data_to {end_num-start_num} \
            --dataset_folder \"{dataset_folder}\" \
            --dataset_type \"{dataset_type}\" \
            --model \"{main_model}\" \
            --reference_models \"{reference_models}\" \
            --app_output_info \"{app_output_info_name}\" \
            --check_generate_time \"{check_time}\" \
            --platform \"{platform}\" \
    ")
    
    with open(app_output_info_name, "r") as f:
        file_name = f.read().strip()
    os.system(f"\
        python chain_adder.py\
            --input_folder \"{folder_path}\" \
            --output_folder \"{folder_path_2}\" \
            --seed 121 \
            --file_name \"{file_name}\" \
            --model \"{quick_model}\" \
            --reference_models \"{reference_models}\" \
            --check_generate_time \"{check_time}\" \
            --platform \"{platform}\" \
    ")
    os.system(f"\
        python chain_evaluater.py\
            --folder_path \"{folder_path_2}\" \
            --seed 121 \
            --file_name \"{file_name}\" \
            --model \"{main_model}\" \
            --baseline_model \"{baseline_model}\" \
            --check_generate_time \"{check_time}\" \
            --platform \"{platform}\" \
    ")
    while True:
        try:
            with open("run_all_report.txt", "a") as f:
                f.write(f"time: {get_time_str()} {name}, {start_num}-{end_num}, en, multi completed\n")
            break
        except:
            print("retry writing report...")
            time.sleep(2)

from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = []
    for src, start, num in [("reddit", 0, run_instances)]:
        for idx in range(start, start+num):
            name = f"app_output_info/{idx}"
            future = executor.submit(
                do_all,
                name,       # name
                idx,        # start_num
                idx + 1,    # end_num
                src,        # dataset_src
                "no"
            )
            futures.append(future)
    for future in futures:
        print(f"future result: {future.result()}")