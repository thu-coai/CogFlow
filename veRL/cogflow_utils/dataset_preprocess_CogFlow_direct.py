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
"""
Preprocess the CogGraph dataset to parquet format
"""

import re
import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer
from verl.utils.hdfs_io import copy, makedirs
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='./data_cogflow/')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--tokenizer', type=str)

    args = parser.parse_args()

    TOKENIZER = AutoTokenizer.from_pretrained(args.tokenizer)

    train_files = [
        os.path.join(args.local_dir, f"rl_cog_flow_train/rl_cog_flow_train.json"), 
    ]
    
    eval_files = [
        os.path.join(args.local_dir, f"rl_cog_flow_eval/rl_cog_flow_eval.json"), 
    ]
    
    dataset = load_dataset("json", data_files={
        "train": train_files,
        "test": eval_files
    })

    train_dataset = dataset['train']
    test_dataset = dataset['test']

    data_src_cnt = {}
    think_len = []
    response_len = []

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):
            data_source = example.pop('data_source')
            data_src_cnt[data_source] = data_src_cnt.get(data_source, 0) + 1
            question = example.pop('user_input')
            rm_instruction = example.pop('rm_instruction')
            reference_responses = example.pop('reference_responses')
            curr_think_len = []
            curr_response_len = []
            def get_len(s: str):
                return len(TOKENIZER.encode(s))
            for ref in reference_responses[:]:
                think_len.append(get_len(ref.split('</think>')[0]))
                curr_think_len.append(get_len(ref.split('</think>')[0]))
                response_len.append(get_len(ref.split('</think>')[-1]))
                curr_response_len.append(get_len(ref.split('</think>')[-1]))
            for ref in reference_responses:
                assert len(ref.split('</think>')) == 2
            if len(reference_responses) <= 2:
                print('no reference response')
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                       "role": "system", 
                       "content": "You are a helpful assistant. " 
                    }, 
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "cog",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": rm_instruction
                },
                "extra_info": {
                    'split': split,
                    'index': idx, 
                    'reference': {
                        'responses': [reference_responses], 
                        'len_reason_short': min(curr_think_len),
                        'len_reason_long': max(curr_think_len),
                        'epoch': 0,
                    },
                    'user_input': question,
                    'answer': 'no_answer',
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'direct_train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'direct_test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)
        
    print(data_src_cnt)
    print('think:', sum(think_len) / len(think_len))
    print('response:', sum(response_len) / len(response_len))
