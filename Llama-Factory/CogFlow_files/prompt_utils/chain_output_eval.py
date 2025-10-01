from .chain_eval_utils import *
from .evaluate_template import *
import os
import json
import csv
import numpy as np

import argparse

MAX_ONE_BATCH = 11

def score_by_output(user_input, all_chains: list, args, type = "mid", top_k = None) -> tuple[list, list]:
    # select all output
    all_output = []
    for chain in all_chains:
        all_output.append({
            "id": chain[-1]["id"], 
            "content": chain[-1]["content"]
        })
    random.shuffle(all_output)
    # all_output[1], all_output[0] = all_output[0], all_output[1]
    # print([item["id"] for item in all_output])
    
    all_explain = []
    
    def call_result(select):
        # print(select)
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
        # print(res_str)
        result = sorted(result, key=lambda x: x['id'])
        res_str = [f"{item['id']}: {item['score']}" for item in result]
        # print(res_str)
        result = sorted(result, key=lambda x: x['score'], reverse=True)
        res_str = [f"{item['id']}: {item['score']}" for item in result]
        # print(res_str)
        with open("output_order.tmp", "a") as f:
            f.write(f"{res_str}")
        
        # store outcome
        for chain in all_chains:
            for item in result:
                if item["id"] == chain[-1]["id"]:
                    chain[-1]["additional"]["output_score"] = item["score"]
                    chain[-1]["additional"]["output_reason"] = item.get('reason', "")
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
    
    sort_group(all_id)
    return all_chains, all_explain

def eval_responses(user_input: str, eval_responses: list[str], ref_responses: list[str], model: str = 'qwen3_32B', score_type = "mid", platform = "custom"):
    args = argparse.Namespace()
    args.model = model
    args.platform = platform
    args.check_generate_time = 'no'
    
    if score_type == "individual": 
        all_eval_score = []
        all_ref_score = []
        template = output_direct_score_no_ref_template
        for resp in eval_responses:
            prompt = template.format(
                user_input = user_input,
                answer = resp
            )
            result = call_api_json_repeat(prompt, args, {"score": 1}, "", 1, '{', False)
            all_eval_score.append(int(result['score']))
        all_score = [item/10 for item in all_eval_score]
        return all_score, all_ref_score, all_eval_score
    
    all_chains = []
    for idx, res in enumerate(ref_responses):
        real_res = res.split('</think>')[-1]
        all_chains.append([{
            'type': 'Terminate', 
            'id': idx+1, 
            'content': real_res,
            'additional': {
                'type': 'reference',
                'id': idx,
            },
        }])
    
    for idx, res in enumerate(eval_responses):
        real_res = res.split('</think>')[-1]
        all_chains.append([{
            'type': 'Terminate', 
            'id': idx + len(ref_responses)+1, 
            'content': real_res,
            'additional': {
                'type': 'eval',
                'id': idx, 
            },
        }])
    
    all_chains, all_explain = score_by_output(user_input, all_chains, args, type=score_type)
    
    all_ref_score = [0 for _ in range(len(ref_responses))]
    for chain in all_chains:
        if chain[-1]['additional']['type'] == 'reference':
            all_ref_score[chain[-1]['additional']['id']] = chain[-1]['additional']['output_score']
    
    all_eval_score = [0 for _ in range(len(eval_responses))]
    for chain in all_chains:
        if chain[-1]['additional']['type'] == 'eval':
            all_eval_score[chain[-1]['additional']['id']] = chain[-1]['additional']['output_score']
    
    all_score = []
    for score in all_eval_score:
        cnt = 0
        for ref_score in all_ref_score:
            if ref_score > score:
                cnt += 1
        all_score.append((len(all_ref_score)-cnt)/len(all_ref_score))
    
    return all_score, all_ref_score, all_eval_score