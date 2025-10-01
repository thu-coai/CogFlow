from chain_eval_utils import *
from evaluate_template import *
import os
import json
import csv
import numpy as np

import argparse

MAX_ONE_BATCH = 12

lock_file_path = "./tmp/result_lock"
result_lock = filelock.FileLock(lock_file_path, timeout=20)

def score_by_output(user_input, all_chains: list, args, type = "mid", top_k = None) -> list:
    
    # select all output
    all_output = []
    for chain in all_chains:
        all_output.append({
            "id": chain[-1]["id"], 
            "content": chain[-1]["content"]
        })
    random.shuffle(all_output)
    # all_output[1], all_output[0] = all_output[0], all_output[1]
    print([item["id"] for item in all_output])
    
    all_explain = []
    
    def call_result(select):
        print(select)
        select_output = []
        for item in select:
            for output in all_output:
                if item == output['id']:
                    select_output.append(output)
        if type == "mid":
            prompt = output_direct_score_template.format(
                user_input = user_input, 
                answers = json.dumps(select_output, indent=4, ensure_ascii=False)
            )
        else:
            prompt = output_direct_score_template.format(
                user_input = user_input, 
                answers = json.dumps(select_output, indent=4, ensure_ascii=False)
            )
        # call api and process result
        result = call_api_json_repeat(prompt, args)
        output_tmp(json.dumps(result, indent=4, ensure_ascii=False), "score_by_output got result")
        all_explain.append(result)
        result = result["result"]
        
        # get log
        res_str = [f"{item['id']}: {item['score']}" for item in result]
        print(res_str)
        result = sorted(result, key=lambda x: x['id'])
        res_str = [f"{item['id']}: {item['score']}" for item in result]
        print(res_str)
        result = sorted(result, key=lambda x: x['score'], reverse=True)
        res_str = [f"{item['id']}: {item['score']}" for item in result]
        print(res_str)
        with open("output_order.tmp", "a") as f:
            f.write(f"{res_str}")
        
        # store outcome
        for chain in all_chains:
            for item in result:
                if item["id"] == chain[-1]["id"]:
                    chain[-1]["additional"]["output_score"] = item["score"]
                    chain[-1]["additional"]["output_reason"] = item["reason"]
        return result
    
    def sort_group(need_sort: list[int]):
        if type == "mid": 
            select = [need_sort[i] for i in range(len(need_sort))]
            result = call_result(select)
            skip_id = [select[0]]
            result = sorted(result, key=lambda x: x['score'], reverse=True)
            tar_pos = len(result)//2
            while tar_pos < len(result)-1 and result[tar_pos]['id'] in skip_id: 
                tar_pos += 1
            select = [result[tar_pos]['id']]
            for item in need_sort:
                if item != result[tar_pos]['id']:
                    select.append(item)
            call_result(select)
            return
        elif type == "ori1":
            select = [need_sort[i] for i in range(len(need_sort))]
            result = call_result(select)
            return
        elif type == "ori2":
            select = [need_sort[i] for i in range(len(need_sort))]
            result = call_result(select)
            random.shuffle(select)
            call_result(select)
            return
    
    all_id = [item['id'] for item in all_output]
    
    # short enough, directly sort it
    if len(all_id) <= MAX_ONE_BATCH: 
        sort_group(all_id)
        return all_chains, all_explain

    id_to_chainid = {}
    for idx, chain in enumerate(all_chains):
        id_to_chainid[chain[-1]['id']] = idx

    top_id = []
    dsr1_id = []
    for chain in all_chains:
        if len(chain) <= 1: continue
        if chain[-2]['type'] == args.baseline_model: 
            dsr1_id.append(chain[-1]['id'])
    
    base_score = -100
    selected_id = copy.deepcopy(dsr1_id)
    for i, id in enumerate(all_id):
        if not id in selected_id: 
            selected_id.append(id)
        if len(selected_id) == MAX_ONE_BATCH or i == len(all_id)-1:
            sort_group(selected_id)
            if base_score == -100:
                r1_chain = all_chains[id_to_chainid[dsr1_id[0]]]
                base_score = r1_chain[-1]['additional']['output_score']
            else:
                r1_chain = all_chains[id_to_chainid[dsr1_id[0]]]
                curr_score = r1_chain[-1]['additional']['output_score']
                for id in selected_id:
                    all_chains[id_to_chainid[id]][-1]['additional']['output_score'] += base_score - curr_score
            for id in selected_id:
                if not id in top_id:
                    top_id.append(id)
            top_id = sorted(top_id, key=lambda x:all_chains[id_to_chainid[x]][-1]['additional']['output_score'], reverse=True)
            if len(top_id) > 3:
                top_id = top_id[:3]
            selected_id = copy.deepcopy(dsr1_id+top_id)
        
    return all_chains, all_explain

def get_output_result_from_chains(all_chains: list):
    result = []
    for chain in all_chains:
        result.append({
            'id': chain[-1]['id'], 
            'score': chain[-1]['additional']['output_score'], 
            'type': chain[-1]['additional'].get('chain_type', chain[-2]['type'] if len(chain) >= 2 else 'error')
        })
    return result
