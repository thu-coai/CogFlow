from .utils_4 import *
from .evaluate_template import *
# from response_constraints import ResponseConstraint

import os
import json
import numpy as np


tmp_folder = "tmp"+datetime.now().strftime("%Y%m%d")
os.makedirs(tmp_folder, exist_ok=True)

def get_tmp_file_name(module_name) -> str:
    return os.path.join(tmp_folder, datetime.now().strftime("%Y%m%d_%H%M%S")+module_name+".txt")

def output_tmp(msg, module_name):
    file_name = get_tmp_file_name(module_name)
    with open(file_name, "w") as f:
        f.write(msg)
    print(f"[INFO] {module_name} message wrote in {file_name}")

def get_chain_from_reason_nodes(nodes: list, tar_num) -> list:
    # input tree
    # output chain
    result = []
    while tar_num >= 0:
        for node in nodes:
            if node["id"] == tar_num:
                result.append(node)
                tar_num = node["parent"]
                break
    result.reverse()
    return result

def convert_chain_to_str(nodes: list) -> str:
    # input:  [{"id": 0, "parent": -1, "type": "6 types / Identification / Terminate", "content": "..."}]
    # output: "content1\n content2 \n ..."
    result = ""
    for i in range(len(nodes)):
        cur_node_str = f"<{nodes[i]['type']}>{nodes[i]['content']}</{nodes[i]['type']}>\n"
        if nodes[i]['type'] == 'Terminate': 
            result += "\n\n"
        result += cur_node_str
    return result

def get_rank_from_score(score):
    answer = []
    score = sorted(score, key=lambda x:x['score'])
    answer.append({
        'id': score[0]['id'], 
        'score': 1
    })
    for i in range(1, len(score)):
        cur_rank = i
        if score[i]['score'] == score[i-1]['score']:
            cur_rank = answer[i-1]['score']
        answer.append({
            'id': score[i]['id'], 
            'score': cur_rank
        })
    return answer

def get_result_from_explains(all_explains: list, func = "last"):
    result_dict = {}
    score1 = all_explains[0]['result']
    if func == "rank": 
        score1 = get_rank_from_score(score1)
    for item in score1:
        result_dict[item['id']] = item['score']
    score2 = all_explains[1]['result']
    if func == "rank":
        score2 = get_rank_from_score(score2)
    for item in score2:
        if func == "last":
            result_dict[item['id']] = item['score']
        else:
            result_dict[item['id']] = (result_dict[item['id']] + item['score'])/2
    result = [{'id': i, 'score': result_dict[i]} for i in result_dict.keys()]
    return result

def compare_results(result1, result2) -> dict:
    result1 = sorted(result1, key=lambda x:x['id'])
    result2 = sorted(result2, key=lambda x:x['id'])
    
    print("results prepared: ")
    print(result1)
    print(result2)
    
    stats = {}
    
    # calculate Kendall-like score
    def cmp(num1, num2):
        if num1 == num2: return 0.5
        elif num1 < num2: return 0
        else: return 1
    tot = 0
    for i in range(len(result1)):
        for j in range(i+1, len(result1)):
            res1 = cmp(result1[i]['score'], result1[j]['score'])
            res2 = cmp(result2[i]['score'], result2[j]['score'])
            tot += abs(res1-res2)
    kendall = tot / ((len(result1)-1) * len(result1) // 2)
    print(kendall)
    stats["kendall"] = kendall
    
    # calculate top-k overlap
    result1 = sorted(result1, key=lambda x:x['score'], reverse=True)
    result2 = sorted(result2, key=lambda x:x['score'], reverse=True)
    prev_score = set()
    for i in range(len(result1)):
        prev_score.add(result1[i]['id'])
        inter_cnt = 0
        worst = i
        for j in range(i+1):
            if result2[j]['id'] in prev_score:
                inter_cnt += 1
            else:
                for k in range(len(result1)):
                    if result1[k]['id'] == result2[j]['id']:
                        worst = max(k, worst)
        print(f"top-{i+1}: {inter_cnt} / {i+1} = {inter_cnt / (i+1)}; worst: {worst+1}")
        stats[f"top-{i+1}"] = inter_cnt / (i+1)
        stats[f"top-{i+1}-worst"] = worst+1
    
    # calculate buttom-k overlap
    result1.reverse()
    result2.reverse()
    prev_score = set()
    for i in range(len(result1)):
        prev_score.add(result1[i]['id'])
        inter_cnt = 0
        worst = i
        for j in range(i+1):
            if result2[j]['id'] in prev_score:
                inter_cnt += 1
            else:
                for k in range(len(result1)):
                    if result1[k]['id'] == result2[j]['id']:
                        worst = max(k, worst)
        print(f"buttom-{i+1}: {inter_cnt} / {i+1} = {inter_cnt / (i+1)}; worst: {worst+1}")
        stats[f"buttom-{i+1}"] = inter_cnt / (i+1)
        stats[f"buttom-{i+1}-worst"] = worst+1
        
    return stats
        
def extract_user_input_chains(datum, args):
    # get scenario and question
    scenario = datum["extracted"]["scenario"]
    question = datum["extracted"]["question"]
    # get user_input
    user_input = scenario+" "+question
    # constraint = ResponseConstraint(datum['extracted'].get('constraint', {}))
    constraint = datum['extracted'].get('constraint', "")
    
    real_len = len(datum["reason_nodes"])
    while real_len > 0 and datum["reason_nodes"][real_len-1]["content"] == "none": 
        real_len -= 1
    
    datum["reason_nodes"] = datum["reason_nodes"][:real_len]
    
    # get all chains
    all_chains = []
    for node in datum["reason_nodes"]:
        # print(f"{node['id']}: {node['type']}")
        if node["type"] == "Terminate":
            all_chains.append(get_chain_from_reason_nodes(datum["reason_nodes"], node["id"]))
            
    return user_input, all_chains, constraint