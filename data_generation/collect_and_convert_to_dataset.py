import json
import os
import logging
import random
import copy
import traceback

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler("collect_and_convert_to_dataset.log")
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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input_folders", type=str, required=True, help="Example: result/CogFlow_ds-r1_6_added")
parser.add_argument("--output_folder", type=str, required=True)
parser.add_argument("--reference_api", type=str, default="ds-r1")
parser.add_argument("--skip_apis", type=str, default="ds-v3,gpt")
args = parser.parse_args()
args.input_folders = args.input_folders.split(",")
args.skip_apis = args.skip_apis.split(",")

def get_all_tags():
    tags = ['<think>', '</think>', '<Observation>', '</Observation>', '<Attribution>', '</Attribution>', '<Behavior>', '</Behavior>', '<Motivation>', '</Motivation>', '<Regulation>', '</Regulation>', '<Efficacy>', '</Efficacy>']
    for api in args.skip_apis + [args.reference_api]:
        tags.append(f"<{api}>")
        tags.append(f"</{api}>")
    return tags

def get_chain_from_reason_nodes(nodes: list, tar_num) -> list:
    # input tree
    # output one chain
    result = []
    terminate_cnt = 0
    while tar_num >= 0:
        for node in nodes:
            if node["id"] == tar_num:
                node['content'] = str(node['content'])
                result.append(copy.deepcopy(node))
                tags = get_all_tags()
                for s in tags:
                    if s in node['content']:
                        logger.warning(f"{s} in node['content'] of {node['type']}, drop this chain")
                        return []
                if result[-1]['type'] == 'Terminate' and terminate_cnt >= 1:
                    # logger.info("changed")
                    result[-1]['type'] = 'Behavior'
                if result[-1]['type'] == 'Terminate':
                    terminate_cnt += 1
                tar_num = node["parent"]
                break
    if terminate_cnt > 1:
        logger.warning("should not occur")
        return []
    result.reverse()
    return result

def extract_user_input_chains(datum):
    # get scenario and question
    scenario = datum["extracted"]["scenario"]
    question = datum["extracted"]["question"]
    # get user_input
    constraint = datum['extracted'].get('constraint', "")
    assert scenario is not None
    assert question is not None
    assert constraint is not None
    user_input = "\n".join([scenario, question, constraint])
        
    real_len = len(datum["reason_nodes"])
    while real_len > 0 and datum["reason_nodes"][real_len-1]["content"] == "none": 
        real_len -= 1
    
    datum["reason_nodes"] = datum["reason_nodes"][:real_len]
    
    # get all chains
    all_chains = []
    is_fa = {i['id']: False for i in datum["reason_nodes"]}
    for node in datum["reason_nodes"]:
        if node['parent'] >= 0:
            is_fa[node['parent']] = True
    for idx, node in enumerate(datum["reason_nodes"]):
        if node["type"] == "Terminate" and len(str(node["content"])) >= 5 and 'output_score' in node['additional'] and not is_fa[node['id']]:
            curr_chain = get_chain_from_reason_nodes(datum["reason_nodes"], node["id"])
            if len(curr_chain) > 1:
                all_chains.append(curr_chain)
            
    return user_input, all_chains, constraint

def convert_chain_to_str(nodes: list, no_result = False) -> str:
    # input:  [{"id": 0, "parent": -1, "type": "6 types / Identification / Terminate", "content": "..."}]
    # output: "content1\n content2 \n ..."
    result = "<think>\n"
    result_cnt = 0
    for i in range(len(nodes)):
        cur_node_str = f"{nodes[i]['content']}\n"
        tags = get_all_tags()
        for s in tags:
            assert s not in nodes[i]['content'], f'{s} should not appear in the content'
        if nodes[i]['type'] == 'Terminate' and not 'output_score' in nodes[i]['additional']:
            nodes[i]['type'] = 'Behavior'
            logger.warning(f"change Terminate to Behavior")
        if nodes[i]['type'] == 'Terminate': 
            result += "</think>\n"
            result_cnt += 1
            if no_result:
                break
        else: 
            cur_node_str = f"\n<{nodes[i]['type']}>\n" + cur_node_str + f"\n</{nodes[i]['type']}>\n"
        result += cur_node_str
    if result_cnt != 1:
        logger.warning(f"result_cnt: {result_cnt}")
    assert result_cnt == 1, f"result_cnt: {result_cnt}"
    return result

def import_data_from_file(fp: str) -> list: 
    try:
        with open(fp, "r") as f:
            data = json.load(f)
        try:
            user_input, all_chains, constraint = extract_user_input_chains(data)
        except Exception as e:
            logger.exception(f"error in extract_user_input_chains: {fp}, {e}")
            exit()
        # user_input = user_input+"\n"+constraint
        logger.debug(f"{fp} extracted: {len(all_chains)} responses. ")
        all_score = [{
            'id': chain[-1]['id'], 
            'score': chain[-1]['additional']['output_score']
        } for chain in all_chains if len(chain) >= 2 and not chain[-2]['type'] in args.skip_apis]
        all_score = sorted(all_score, key=lambda x:x['score'], reverse=True)
        if len(all_score) <= 2:
            logger.warning(f"{fp} has less than 3 responses. ")
            return []
        # logger.debug(f"{fp} sorted: {len(all_score)} responses. ")
        
        id_to_chain = {chain[-1]['id']: idx for idx, chain in enumerate(all_chains)}
        
        result = []
        def format_data_point(ref_ids, ref_scores):
            # input(f"in format_data_point: {len(result)}")
            ref_responses = [all_chains[id_to_chain[ref_id]][-1]['content'] for ref_id in ref_ids]
            ref_chains = [convert_chain_to_str(all_chains[id_to_chain[ref_id]]) for ref_id in ref_ids]
            ref_chains_with_type = []
            for ref_id, ref_score in zip(ref_ids, ref_scores):
                curr_chain = all_chains[id_to_chain[ref_id]]
                chain = convert_chain_to_str(curr_chain)
                chain_type = curr_chain[-1]['additional'].get('chain_type', 'cog')
                type_map = {
                    'cog': 'cogflow', 
                    'api': 'r1', 
                    'fake': 'fake_variations'
                }
                ref_chains_with_type.append({
                    'reasoning_and_response': chain, 
                    'source': type_map[chain_type],
                    'score': ref_score,
                })
            for chain in ref_chains:
                if len(chain.split('</think>')) != 2:
                    logger.error(f"error in {fp}: {chain}")
            
            result.append({
                "input": user_input,  
                "reference_responses_from_diverse_source": ref_chains_with_type,
                "major_category": "null", 
                "sub_category": "null",
            })
        
        cnt = len(all_score)
        logger.debug(f"{fp} top-{cnt} responses. ")
        format_data_point(
            [item['id'] for item in all_score[0:cnt]],
            [item['score'] for item in all_score[0:cnt]],
        )
        
        return result
    except Exception as e:
        logger.exception(f"error in {fp}: {e}")
        return []

def main():
    all_fp = []
    for folder in args.input_folders:
        for subfolder in os.listdir(folder):
            if os.path.isdir(os.path.join(folder, subfolder)):
                valid_files = []
                for file in os.listdir(os.path.join(folder, subfolder)):
                    if file.endswith(".json"):
                        valid_files.append(file)
                tar_file = sorted(valid_files)[-1]
                all_fp.append(os.path.join(folder, subfolder, tar_file))
    logger.info(f"total files: {len(all_fp)}")
    all_data = []
    for fp in all_fp:
        data = import_data_from_file(fp)
        all_data.extend(data)
    logger.info(f"total data: {len(all_data)}")
    
    # Split and Output the dataset
    with open(os.path.join(args.output_folder, "generated_data.json"), "w") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    logger.info(f"output to {os.path.join(args.output_folder, "generated_data.json")}: {len(all_data)}")
    
    train_split_pt = int(len(all_data) * 0.9)
    train_data = all_data[:train_split_pt]
    test_data = all_data[train_split_pt:]
    
    with open(os.path.join(args.output_folder, "train.json"), "w") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    logger.info(f"output to {os.path.join(args.output_folder, 'train.json')}: {len(train_data)}")
    
    with open(os.path.join(args.output_folder, "test.json"), "w") as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)
    logger.info(f"output to {os.path.join(args.output_folder, 'test.json')}: {len(test_data)}")

if __name__ == "__main__":
    main()