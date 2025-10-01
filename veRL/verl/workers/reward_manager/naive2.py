# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
from collections import defaultdict

import math
import copy

class NaiveRewardManager2:
    """The reward manager.
    provide the output of reward model for customized reward function. 
    NOTICE: the response did not skip_special_token, bacause we need to check the format. 
    """

    def __init__(
        self, tokenizer, num_examine, compute_score=None, reward_fn_key='data_source', 
        src_use_custom_reward: list[str]=[],
        reward_merge_method: str='min',
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key
        self.src_use_custom_reward = src_use_custom_reward
        self.reward_merge_method = reward_merge_method
        
    def get_all_data_dict(self, data: DataProto):
        all_data_dict = {} # key is datasource + extra_info.split + extra_info.index
        # gather the response of every input
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False) #####

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info: dict = data_item.non_tensor_batch.get('extra_info', {})

            current_data_key = f"{data_source}_{extra_info.get('split','none')}_{extra_info.get('index', 'none')}"
            if current_data_key not in all_data_dict.keys():
                all_data_dict[current_data_key] = {
                    "user_input": extra_info['user_input'],
                    'reference': extra_info.get('reference', {}),
                    'current_responses': [],
                }
            all_data_dict[current_data_key]['current_responses'].append({
                'response_str': response_str,
                'id': i, 
            })
        return all_data_dict

    def get_entropy(self, data_len: int, all_data_dict: dict) -> tuple[list[int], list[int], list[int]]:
        # calculate the number of cognitive units and calculate the entropy score. 
        data_node_entropy_score_list = [0 for _ in range(data_len)]
        data_node_entropy_list = [0 for _ in range(data_len)]
        max_entropy_list = [0 for _ in range(data_len)]
        for key in all_data_dict.keys():
            curr_responses = all_data_dict[key]['current_responses']
            all_node_cnt = {'Observation': 0, 'Attribution': 0, 'Motivation': 0, 'Regulation': 0, 'Efficacy': 0, 'Behavior': 0}
            for response in curr_responses:
                for node in all_node_cnt.keys():
                    cnt = len(response['response_str'].split(f'<{node}>'))-1
                    all_node_cnt[node] += cnt
            node_tot = sum(all_node_cnt.values())
            # print(f"node_cnt = {all_node_cnt}, node_tot = {node_tot}")
            all_entropy = {}
            for response in curr_responses:
                entropy_list = []
                for node in all_node_cnt.keys():
                    cnt = len(response['response_str'].split(f'<{node}>'))-1
                    prob = all_node_cnt[node]/(node_tot+1e-4)
                    for _ in range(cnt):
                        entropy_list.append(-math.log2(prob))
                if len(entropy_list) > 0:
                    entropy_num = sum(entropy_list) / len(entropy_list)
                else:
                    entropy_num = 0
                all_entropy[response['id']] = entropy_num
            max_entropy = max(all_entropy.values()) + 1e-6
            for response in curr_responses:
                data_node_entropy_score_list[response['id']] = all_entropy[response['id']] / max_entropy
                data_node_entropy_list[response['id']] = all_entropy[response['id']]
                max_entropy_list[response['id']] = max_entropy
        # print("data node entropy list: ", data_node_entropy_list)
        return data_node_entropy_score_list, data_node_entropy_list, max_entropy_list

    def __call__(self, data: DataProto, return_dict=False, epoch=-1):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        # if 'rm_scores' in data.batch.keys():
        #     if return_dict:
        #         return {"reward_tensor": data.batch['rm_scores']}
        #     else:
        #         return data.batch['rm_scores']

        # print(data.batch.keys())
        # print(data.batch['rm_scores'].shape)

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}
        
        all_data_dict = self.get_all_data_dict(data) # key is datasource + extra_info.split + extra_info.index

        data_node_entropy_score_list, data_node_entropy_list, max_entropy_list = self.get_entropy(len(data), all_data_dict)

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            # print(data_item.batch.keys())
            # print(data_item.non_tensor_batch.keys())

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            
            # get RM output
            if 'rm_scores' in data_item.batch.keys():
                rm_scores = data_item.batch.get('rm_scores', 0)
                last_token_reward = rm_scores[valid_response_length - 1]  
                reward_number = last_token_reward.item()
            else:
                reward_number = 0

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False) #####
            response_str: str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=False) #####

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            data_source = data_item.non_tensor_batch[self.reward_fn_key]

            extra_info = data_item.non_tensor_batch.get('extra_info', None)
            
            entropy_info = {
                'node_entropy_score': data_node_entropy_score_list[i],
                'max_entropy': max_entropy_list[i],
                'node_entropy': data_node_entropy_list[i],
            }

            if data_source in self.src_use_custom_reward:
                score = self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    rm_scores=reward_number,
                    correctness=1.,
                    entropy_info=entropy_info,
                )
            else:
                score = _default_compute_score(
                    data_source=data_source,
                    solution_str=response_str.split('</think>')[-1],
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )
                score = self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    rm_scores=reward_number,
                    correctness=score,
                    entropy_info=entropy_info,
                )

            if isinstance(score, dict):
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print(f"[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
