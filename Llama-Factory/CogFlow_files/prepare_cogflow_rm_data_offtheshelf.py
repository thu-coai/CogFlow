import json
import os
import logging
import argparse

parser = argparse.ArgumentParser(description="Process and validate pre-formatted dataset splits (train, eval, test).")
parser.add_argument("--train_path", type=str, help="Path to the input training JSON file.")
parser.add_argument("--eval_path", type=str, help="Path to the input evaluation JSON file.")
parser.add_argument("--test_path", type=str, help="Path to the input testing JSON file.")
parser.add_argument("--output_path", type=str, required=True, help="Path to the output directory where dataset splits will be saved.")
parser.add_argument("--dataset_name", type=str, required=True, help="Prefix for the output dataset files and dataset_info entries.")
parser.add_argument("--dataset_info_path", type=str, required=True, help="Path to the dataset_info.json file to be created or updated.")
parser.add_argument("--log_path", type=str, default="logs/dataset_processing.log", help="Path to the log file.")
args = parser.parse_args()

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_dir = os.path.dirname(args.log_path)
if log_dir:
    os.makedirs(log_dir, exist_ok=True)
file_handler = logging.FileHandler(args.log_path, mode='w') # Use mode 'w' to overwrite log each run
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def output_dataset(all_data: list, tar_folder: str, dataset_name: str, info_file_path: str):
    os.makedirs(tar_folder, exist_ok=True)
    full_file_path = os.path.join(tar_folder, f"{dataset_name}.json")
    
    info_dir = os.path.dirname(info_file_path)
    if info_dir: 
        os.makedirs(info_dir, exist_ok=True)
        
    try:
        with open(info_file_path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        config = {}
        
    relative_file_path = os.path.relpath(full_file_path, info_dir)
    columns = {"prompt": "instruction", "label": "label"}
    config[dataset_name] = {"file_name": relative_file_path.replace('\\', '/'), "columns": columns}
    
    with open(info_file_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
        
    with open(full_file_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
        
    logger.info(f"Successfully saved {len(all_data)} samples to {full_file_path}")
    logger.info(f"Updated dataset info in {info_file_path}")

def validate_and_process_split(file_path: str, split_name: str, args):
    logger.info(f"--- Processing {split_name} split from {file_path} ---")
    
    if not os.path.exists(file_path):
        logger.error(f"File not found for {split_name} split: {file_path}. Skipping.")
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {file_path}: {e}. Skipping.")
        return
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading {file_path}: {e}. Skipping.")
        return

    if not isinstance(data, list):
        logger.error(f"Data in {file_path} is not a list. Skipping.")
        return

    is_valid = True
    if not data:
        logger.warning(f"File {file_path} is empty. Skipping.")
        return

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            logger.error(f"Item at index {i} in {file_path} is not a dictionary. Skipping split.")
            is_valid = False
            break
        if "instruction" not in item or "label" not in item:
            logger.error(f"Item at index {i} in {file_path} is missing 'instruction' or 'label' key. Skipping split.")
            is_valid = False
            break
    
    if not is_valid:
        return
    
    logger.info(f"Validation successful for {split_name} split. Found {len(data)} samples.")
    
    output_dataset(
        all_data=data,
        tar_folder=args.output_path,
        dataset_name=f"{args.dataset_name}_{split_name}",
        info_file_path=args.dataset_info_path
    )

def main(args):
    logger.info("Starting dataset processing and validation.")
    logger.info(f"Arguments received: {args}")
    
    splits_to_process = {
        "train": args.train_path,
        "eval": args.eval_path,
        "test": args.test_path
    }
    
    processed_count = 0
    for split_name, file_path in splits_to_process.items():
        if file_path:
            validate_and_process_split(file_path, split_name, args)
            processed_count += 1
        else:
            logger.info(f"No path provided for {split_name} split. Skipping.")
    
    if processed_count == 0:
        logger.warning("No dataset paths were provided. The script finished without processing any files.")
    else:
        logger.info("All provided datasets have been processed.")

if __name__ == "__main__":
    main(args)
