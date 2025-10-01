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
The vllm_rollout that can be applied in different backend
When working with FSDP:
- Use DTensor weight loader (recommended) or HF weight loader
- Utilize state_dict from the FSDP to synchronize the weights among tp ranks in vLLM
When working with Megatron:
- Use Megatron weight loader
- During training, only the current pp stage holds the parameters
- Before inference, broadcast the parameters of the current pp rank to all other pp ranks (all pp ranks holds all the parameters)
- Bind the parameters to the inference engine
- Do inference in tp. pp is treated as additional dp
- After inference, all the parameters that doesn't belong to this pp rank is freed.
"""
import numpy as np
from typing import List
from contextlib import contextmanager
from omegaconf import DictConfig
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from typing import Any, Union
from verl import DataProto
from verl.utils.torch_functional import get_response_mask, pad_2d_list_to_length
from verl.workers.rollout.base import BaseRollout
from vllm.distributed import parallel_state as vllm_ps
from vllm import LLM, SamplingParams
from verl.third_party.vllm import vllm_version
from vllm.outputs import RequestOutput
from vllm.sequence import SampleLogprobs

from .vllm_rollout_spmd import vLLMRollout

# NOTE(sgm): add for verl. We can optimize it by making the dataloader yield List[int] without padding.
def _pre_process_inputs(pad_token_id, prompt_token_ids: torch.Tensor) -> List[int]:
    # remove the left padding in the prompt token_id
    # pad_token_id = self.llm_engine.tokenizer.pad_token_id if self.llm_engine.tokenizer.pad_token_id is not None else self.llm_engine.tokenizer.eos_token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids

def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, List[Any]]:
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)

class MCTSNode:
    def __init__(self, parent=None, token_sequence=None, start_token=None, prior_prob=1.0):
        print(f"in MCTSNode init: {token_sequence[:5]}, {len(token_sequence)}")
        self.parent = parent
        self.children = []
        self.visit_count = 0
        self.q_value = 0.0
        self.token_sequence = token_sequence or []
        self.start_token = start_token  # 该节点对应的开始token
        self.prior_prob = prior_prob  # 新增先验概率属性
        self.is_leaf = False # 由于有错误的格式，所以可能叶子的token不是leaf_token

    def uct_value(self, exploration_weight=1.0, default_exist=2.0):
        if self.visit_count == 0:
            return float('inf')
        # 修改UCT公式：q + c * p * sqrt(N) / (n + default_exist)
        return self.q_value + exploration_weight * self.prior_prob * np.sqrt(self.parent.visit_count) / (self.visit_count + default_exist)

#   "</Attribution and Evaluation>": 151670,
#   "</Behavior>": 151678,
#   "</Motivation>": 151672,
#   "</Observation>": 151668,
#   "</Self-efficacy>": 151676,
#   "</Self-regulatory>": 151674,
#   "<Attribution and Evaluation>": 151669,
#   "<Behavior>": 151677,
#   "<Motivation>": 151671,
#   "<Observation>": 151667,
#   "<Self-efficacy>": 151675,
#   "<Self-regulatory>": 151673,

#   "<think>": 151665,
#   "</think>": 151666,

class vLLMRolloutMCTS(vLLMRollout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.start_end_pairs = [
            (151667, 151668), # Observation
            (151669, 151670), # Attribution
            (151671, 151672), # Motivation
            (151673, 151674), # Self-regulatory
            (151675, 151676), # Self-efficacy
            (151677, 151678), # Behavior
        ]
        self.root_token = 151665
        self.leaf_token = 151666
        self.exploration = self.config.mcts.exploration
        self.default_exist = self.config.mcts.default_exist
        print(f"MCTS exploration: {self.exploration}, default_exist: {self.default_exist}")
        
        tokenizer = self.inference_engine.get_tokenizer()
        print(f"tokenizer = {tokenizer}")
        print(f"vocab_size = {tokenizer.vocab_size}")
        print(f"max_token_id = {tokenizer.max_token_id}")
        # exit()
        
    def _find_expand_node(self, node: MCTSNode) -> MCTSNode:
        """使用Predictor UCT算法选择需要扩展的节点"""
        while True:
            print(f"in find_expand_node: {node.start_token}")
            if len(node.children) == 0:
                return node
            # 选择具有最大UCT值的子节点
            possible_children = [child for child in node.children if not child.is_leaf]
            node = max(possible_children, key=lambda x: x.uct_value(self.exploration, self.default_exist))
    
    def _parse_generation(self, curr_node: MCTSNode, result: List[dict], eos_token_id: int | list[int]) -> MCTSNode:
        """解析生成的token序列为节点结构"""
        # result is [{position: , generated_token: , target_logits: }]
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        current_tokens = []
        start_tokens = [pair[0] for pair in self.start_end_pairs] + [self.leaf_token]
        assert len(curr_node.children) == 0, "the start node should not have children"
        assert len(curr_node.token_sequence) == 1, "the start node should only have one start token"
        
        print(f"in parse: curr_node = {curr_node.start_token}, {curr_node.token_sequence[:5]}, {len(curr_node.token_sequence)}, result = {[i['generated_token'] for i in result[:5]]}, {len(result)}")
        print(f"start_tokens = {start_tokens}")
        print(f"eos_token_id = {eos_token_id}")        

        if curr_node.start_token != self.leaf_token:
            # print("  parse expanding start node")
            for possible_token in start_tokens:
                # print(f"    possible_token = {possible_token}")
                curr_node.children.append(MCTSNode(
                    parent=curr_node, 
                    start_token=possible_token, 
                    token_sequence=[possible_token], 
                    prior_prob=1, 
                ))
        
        new_node_cnt = 0
        
        for idx, item in enumerate(result):
            token = item['generated_token']
            assert isinstance(token, int)
            # assert token >= 0
            # if token >= 151679:
            #     print(f"token = {token}, idx = {idx}, top_logprobs = {item['top_logprobs']}")
            # assert token < 151679
            if token in eos_token_id:
                print(f"in end token, idx = {idx}, len of response: {len(current_tokens)}, new_node_cnt = {new_node_cnt}")
                break
            if token in start_tokens and curr_node.start_token != self.leaf_token:
                print(f"in start node token: {token}, idx = {idx}, logprob = {item['target_logits']}")
                new_node_cnt += 1
                # 更新上一个节点：token序列、孩子的概率
                import math
                for child in curr_node.children:
                    child.prior_prob = math.exp(item['target_logits'][child.start_token])
                    if child.start_token == token:
                        next_node = child
                curr_node.token_sequence = [curr_node.start_token] + current_tokens.copy()
                # 开始下一个节点的信息统计
                current_tokens = []
                curr_node = next_node
                if token != self.leaf_token:
                    for possible_token in start_tokens:
                        curr_node.children.append(MCTSNode(
                            parent=curr_node, 
                            start_token=possible_token, 
                            token_sequence=[possible_token], 
                            prior_prob=1, 
                        ))
            else:
                current_tokens.append(token)
        
        print(f"parse end: new_node_cnt = {new_node_cnt}")
        if new_node_cnt == 0 and curr_node.start_token != self.leaf_token:
            # 若格式正确，就不应该到这里
            curr_node.children.append(MCTSNode(
                parent=curr_node, 
                start_token=self.leaf_token, 
                token_sequence=current_tokens.copy(), 
                prior_prob=1,
            ))
            curr_node = curr_node.children[-1]
            curr_node.is_leaf = True
        else:
            # 这个时候curr_node的start_token应该是self.leaf_token，所以不需要再添加子节点
            curr_node.token_sequence = [curr_node.start_token] + current_tokens.copy()
            curr_node.is_leaf = True
        
        return curr_node
    
    def _build_response_paths(self, root: MCTSNode) -> List[List[int]]:
        """收集所有根到叶子的路径"""
        paths = []
        
        def dfs(node: MCTSNode, current_path: list[int]):
            current_path += node.token_sequence
            if node.is_leaf:
                paths.append(current_path.copy())
            else:
                for child in node.children:
                    dfs(child, current_path.copy())
            
        dfs(root, [])
        return paths

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # 验证时直接采样，不再使用mcts
        is_validate = prompts.meta_info.get('validate', False)
        if is_validate:
            return super().generate_sequences(prompts, **kwargs)
        
        print("in mcts rollout")
        # rebuild vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.init_cache_engine()
            
        batch_size = prompts.batch['input_ids'].size(0)
        output_batch_size = batch_size * self.sampling_params.n
        eos_token_id = prompts.meta_info['eos_token_id']
        
        non_tensor_batch = prompts.non_tensor_batch
        if 'raw_prompt_ids' not in non_tensor_batch:
            non_tensor_batch['raw_prompt_ids'] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)], dtype=object)
            
        if batch_size != len(non_tensor_batch['raw_prompt_ids']):
            raise RuntimeError('vllm sharding manager is not work properly.')
        
        all_raw_prompt_ids = non_tensor_batch.pop('raw_prompt_ids')
        
        # 初始化每个输入的树结构
        mcts_forest = [
            MCTSNode(token_sequence=[self.root_token], start_token=self.root_token, prior_prob=1) 
            for id in range(batch_size)
        ]
        
        for _ in range(self.sampling_params.n):
            # 步骤1: 选择扩展节点（修改选择逻辑）
            expand_nodes: list[MCTSNode] = []
            for tree in mcts_forest:
                expand_nodes.append(self._find_expand_node(tree))
            
            # 步骤2: 生成并获取logits
            vllm_inputs = []
            for idx, node in enumerate(expand_nodes):
                # 构造当前生成上下文
                context = []
                current = node
                while current is not None:
                    context = current.token_sequence + context
                    current = current.parent
                user_input = all_raw_prompt_ids[idx]
                # context = user_input + context
                # ensure the type of `prompt_token_ids` passed to vllm is list[int]
                # https://github.com/volcengine/verl/pull/772
                if isinstance(context, np.ndarray):
                    context = context.tolist()
                if isinstance(user_input, np.ndarray):
                    user_input = user_input.tolist()

                # for item in context:
                #     assert isinstance(item, int)
                #     assert item >= 0
                #     assert item < 151679
                    
                # for item in user_input:
                #     assert isinstance(item, int)
                #     assert item >= 0
                #     assert item < 151679

                context = user_input + context
                print(f"current context: {context[-5:]}, {len(context)}")
                vllm_inputs.append({
                    'prompt_token_ids': context,
                    # 'stop_token_ids': [pair[1] for pair in self.start_end_pairs] + [self.leaf_token]
                })
                assert isinstance(context, list)
                for item in context:
                    assert isinstance(item, int)
                    # assert item >= 0
                    # assert item < 151679
            
            tokenizer = self.inference_engine.get_tokenizer()
            while True:
                # 调用vLLM生成（修改后的采样参数）
                with self.update_sampling_params(
                    **kwargs, 
                    logprobs=10,  # 获取top logprobs用于概率计算
                    # prompt_logprobs=0,
                    n=1,
                    # top_p=0.7,
                ):
                    outputs: list[RequestOutput] = self.inference_engine.generate(
                        prompts=vllm_inputs, 
                        sampling_params=self.sampling_params,
                        use_tqdm=False
                    )
                invalid = False
                for idx, request_output in enumerate(outputs):
                    per_request_results = []
                    for completion in request_output.outputs:
                        # 获取生成的token ID
                        generated_token_ids = completion.token_ids
                        for pos_idx, token_id in enumerate(generated_token_ids):
                            if token_id > tokenizer.max_token_id:
                                invalid = True
                                print(f"detected invalid id, retrying: {idx}, {pos_idx}, token = {token_id}")
                            if isinstance(eos_token_id, int):
                                if token_id == eos_token_id:
                                    break
                            else:
                                if token_id in eos_token_id:
                                    break
                if invalid: 
                    continue
                break
                
            # generated = outputs.token_ids
            # generated_seqs.append(generated)
            target_tokens = [s for (s, _) in self.start_end_pairs] + [self.leaf_token]
                
            results = []
            for request_output in outputs:
                per_request_results = []
                for completion in request_output.outputs:
                    # 获取生成的token IDs和对应的logprobs
                    generated_token_ids = completion.token_ids
                    position_logprobs: SampleLogprobs = completion.logprobs  # 每个生成位置的logprobs
                    
                    # 遍历每个生成位置
                    for pos_idx, (token_id, top_logprobs) in enumerate(zip(generated_token_ids, position_logprobs)):
                        # 计算每个target_token在当前位置的概率
                        target_probs = {
                            target: top_logprobs[target].logprob if target in top_logprobs else -1e8
                            for target in target_tokens
                        }
                        # if pos_idx == 0:
                            # print(target_probs)
                            # print(top_logprobs)
                        
                        # 记录当前生成位置的信息
                        per_request_results.append({
                            "position": pos_idx,
                            "generated_token": token_id,
                            "target_logits": target_probs,
                            "top_logprobs": top_logprobs
                        })
                results.append(per_request_results)
            
            # results is [[{position: , generated_token: , target_logits: }]]
            
            print(f"len nodes = {len(expand_nodes)}")
            print(f"len results = {len(results)}")
            
            # 步骤3: 处理生成结果，并更新mcts的统计
            tokenizer = self.inference_engine.get_tokenizer()
            for input, node, result in zip(vllm_inputs, expand_nodes, results):
                # 解析生成结果
                try:
                    leaf_node: MCTSNode = self._parse_generation(node, result, eos_token_id)
                except:
                    output_file = "./tmp_output/mcts_error_3.txt"
                    import os
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    with open(output_file, "a") as f:
                        f.write(f"error in parsing generation\ninput: {input}\ninput_text: {tokenizer.decode(input['prompt_token_ids'], skip_special_tokens=False)}\n\n")
                        correct_tokens = []
                        for idx, item in enumerate(result):
                            if item['generated_token'] < tokenizer.max_token_id:
                                correct_tokens.append(item['generated_token'])
                            else:
                                f.write(f"error token #{idx}: {item} in tokenizer: {item['generated_token'] in list(tokenizer.vocab.values())}")
                                f.write(f"token decoded: {tokenizer.decode(item['generated_token'])}")
                                break
                        f.write(f"correct tokens: {correct_tokens}\ntoken_text: {tokenizer.decode(correct_tokens, skip_special_tokens=False)}\n\n")
                    exit()
                
                # 更新mcts的统计信息
                reward = 0.0  # 这里需要替换为实际reward计算
                current = leaf_node
                while current is not None:
                    current.visit_count += 1
                    current.q_value += (reward - current.q_value) / current.visit_count
                    current = current.parent
        
        # 收集所有路径作为response
        all_responses = []
        for idx, tree in enumerate(mcts_forest):
            paths = self._build_response_paths(tree)
            all_responses.extend(paths)
            
            # TMP ADDED: output
            import uuid
            import json
            from datetime import datetime
            import os
            text = [self.inference_engine.get_tokenizer().decode(path, skip_special_tokens=False) for path in paths]
            input_text = self.inference_engine.get_tokenizer().decode(all_raw_prompt_ids[idx], skip_special_tokens=False)
            output_file = f"/data/chenzheyu/CogGraph/VERL/tmp_output/mcts_response_{idx}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4()}.json"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, "w") as f:
                json.dump({
                    "input_text": input_text, 
                    "response_text": text, 
                    "input": all_raw_prompt_ids[idx], 
                    "response_id": paths
                }, f, indent=4, ensure_ascii=False)
            # END TMP ADDED
        print(f"len of responses: {[len(responses) for responses in all_responses]}")
        
        # 转换为与原始格式兼容的tensor
        padded_responses = pad_2d_list_to_length(
            all_responses, 
            self.pad_token_id,
            max_length=self.config.response_length
        ).to(prompts.batch['input_ids'].device)
        
        if self.config.response_length is not None:
            print(f"doing right pad, padded_responses: {padded_responses.shape}, target length: {self.config.response_length}")
            # right pad 'self.pad_token_id' to response_length
            if padded_responses.size(1) > self.config.response_length:
                padded_responses = padded_responses[:, :self.config.response_length]
            else:
                padded_responses = torch.cat([
                    padded_responses,
                    torch.full(
                        (padded_responses.size(0), self.config.response_length - padded_responses.size(1)),
                        fill_value=self.pad_token_id,
                        dtype=padded_responses.dtype,
                        device=padded_responses.device
                    )
                ], dim=-1)
            
        
        # 构造返回数据结构
        
        # 复制输入的batch信息为n份
        print(f"idx: {prompts.batch['input_ids'].shape}")
        idx = _repeat_interleave(prompts.batch['input_ids'], self.sampling_params.n)
        attention_mask = _repeat_interleave(prompts.batch['attention_mask'], self.sampling_params.n)
        position_ids = _repeat_interleave(prompts.batch['position_ids'], self.sampling_params.n)
        
        print(f"sampling_params.n: {self.sampling_params.n}")
        print(f"idx: {idx.shape}")
        print(f"padded_responses: {padded_responses.shape}")
        
        seq = torch.cat([idx, padded_responses], dim=-1)
        
        # 以下内容与原始完全相同
        response_length = padded_responses.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).expand(output_batch_size, -1)
        if position_ids.dim() == 3:  # qwen2vl mrope
            delta_position_id = delta_position_id.view(output_batch_size, 1, -1).expand(output_batch_size, 3, -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=padded_responses,
            eos_token=eos_token_id,
            dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                'prompts': idx,
                'responses': padded_responses,
                'input_ids': seq,  # here input_ids become the whole sentences
                # 'old_log_probs': log_probs, # we will recompute old log prob with actor
                'attention_mask': attention_mask,
                'position_ids': position_ids
            },
            batch_size=output_batch_size
        )

        # free vllm cache engine
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3') and self.config.free_cache_engine:
            self.inference_engine.free_cache_engine()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)