import os
import json
import random
import argparse
import copy

from utils_4 import *
from chain_eval_utils import extract_user_input_chains
from prompt_template_json import cog_ordinary_NBN_result_full_template_en

import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__file__)

def get_new_chain(candidate_node: list[dict], candidate_chain: list[list[dict]], args) -> list[dict]:
    if random.choice([True, False]):
        # operate on existing chain
        old_tar_chain = random.choice(candidate_chain)
        tar_chain = [node for node in old_tar_chain if node['type'] not in ['Terminate']]
        if len(tar_chain) >= 3:
            remain_num = random.randint(1, len(tar_chain)-1)
            return copy.deepcopy(random.sample(tar_chain, remain_num))
    # select random node
    node_num = min(5, len(candidate_node))
    if node_num <= 0:
        return []
    return copy.deepcopy(random.sample(candidate_node, k=node_num))

def convert_chain_to_str(nodes: list) -> str:
    # input:  [{"id": 0, "parent": -1, "type": "6 types / basic / result", "content": "..."}]
    # output: node 1 (name):\n ... \n ...
    result = ""
    for i in range(len(nodes)):
        cur_node_str = f"[{nodes[i]['type']}]:{nodes[i]['content']}\n"
        result += cur_node_str
    return result

def chain_to_output(chain: list[dict], user_input: str, args) -> str:
    chain_str = convert_chain_to_str(chain)
    prompt = cog_ordinary_NBN_result_full_template_en.format(
        user_input = user_input, 
        previous_nodes = chain_str
    )
    result = call_api_repeat(prompt, args, prepare_for_json=False)
    if isinstance(result, dict):
        return result["content"]
    else:
        return result

def add_chain(datum: dict, args) -> tuple[dict, bool]: 
    user_input, all_chains, constraint = extract_user_input_chains(datum, args)
    user_input = user_input+'\n'+constraint
    
    node_id_to_idx = {}
    for idx, node in enumerate(datum['reason_nodes']):
        node_id_to_idx[node['id']] = idx
    
    candidate_chain = []
    
    for chain in all_chains:
        if chain[-2]['type'] in args.reference_models:
            datum['reason_nodes'][node_id_to_idx[chain[-1]['id']]]['additional']['chain_type'] = 'api'
        else: 
            datum['reason_nodes'][node_id_to_idx[chain[-1]['id']]]['additional']['chain_type'] = 'cog'
            candidate_chain.append(chain)
    
    candidate_node = []
    for node in datum['reason_nodes']:
        if node['type'] not in args.reference_models + ['Terminate']:
            candidate_node.append(node)
    
    print(f"len all_chains: {len(all_chains)}, len candidate_chain: {len(candidate_chain)}, len candidate_node: {len(candidate_node)}")
    
    if len(candidate_chain) == 0 or len(all_chains) >= 8:
        return datum, False
    
    for _ in range(8-len(all_chains)):
        for _ in range(5):
            try:
                new_chain = get_new_chain(candidate_node, candidate_chain, args)
                assert len(new_chain) > 0
                for node in new_chain:
                    assert node['type'] in ['Observation', 'Attribution', 'Motivation', 'Regulation', 'Efficacy', 'Behavior'], f"node type: {node['type']}"
            except:
                logger.exception("get_new_chain error")
                continue
            break
        new_chain_output = chain_to_output(new_chain, user_input, args)
        last_id = datum['reason_nodes'][0]['parent']
        for node in new_chain:
            node_id_to_idx[node['id']] = len(datum['reason_nodes'])
            node['id'] = len(datum['reason_nodes'])
            node['parent'] = last_id
            datum['reason_nodes'].append(node)
            last_id = node['id']
        datum['reason_nodes'].append({
            "id": len(datum['reason_nodes']),
            "parent": last_id,
            "type": "Terminate",
            "content": new_chain_output,
            "additional": {
                "chain_type": "fake",
            }
        })
        
    return datum, True

def process_file(args, file_name):
    logger.info(f"Processing file: {file_name}")
    set_seed(args.seed)
    input_file = os.path.join(args.input_folder, file_name+".json")
    output_file = os.path.join(args.output_folder, file_name+".json")
    
    if args.jump_exist and os.path.exists(output_file):
        logger.info(f"File {output_file} already exists, skip re-gen")
    else: 
        logger.info(f"File {output_file} not exists, start re-gen")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        try:
            with open(input_file, "r") as f:
                data = json.load(f)
        except:
            logger.warning(f"File {input_file} load_error, skip")
            return
            
        result = []
        for idx, datum in enumerate(data):
            new_datum, _ = add_chain(datum, args)
            result.append(new_datum)
        
        with open(output_file, "w") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)

def main(args):
    while not check_time_in_discount_period(datetime.now().hour, datetime.now().minute) and args.check_generate_time == "yes":
        logger.warning(f"Time not in discount period, sleep 10 minutes")
        time.sleep(600)
    
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = []
        if args.auto_search:
            for file_name in os.listdir(args.input_folder):
                if file_name[-5:] == ".json":
                    future = executor.submit(process_file, copy.deepcopy(args), file_name[:-5])
                    futures.append(future)
        else:
            future = executor.submit(process_file, args, args.file_name)
            futures.append(future)
        for future in futures:
            future.result()
            # print(f"future result: {future.result()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--output_folder", type=str, required=True)
    parser.add_argument("--file_name", type=str)
    
    # not in use
    parser.add_argument("--auto_search", default=False, action="store_true")
    parser.add_argument("--jump_exist", default=False, action="store_true")
    
    parser.add_argument("--model", type=str, default="ds-v3")
    parser.add_argument("--platform", type=str, default="custom")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check_generate_time", type=str, default="no", choices=["yes", "no"])
    parser.add_argument("--reference_models", type=str, default='ds-r1,ds-v3')
    args = parser.parse_args()
    # change reference_models to list
    args.reference_models = args.reference_models.split(",")

    main(args)