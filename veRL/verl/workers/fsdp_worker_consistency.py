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
The main entry point to run the PPO algorithm
"""

import logging
import os
import warnings
import psutil

import torch
import torch.distributed
from torch.distributed.device_mesh import init_device_mesh
import verl.utils.torch_functional as verl_F
from omegaconf import DictConfig, open_dict
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
from verl.utils import hf_tokenizer, hf_processor
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, init_fn, get_init_weight_context_manager
from verl.utils.fsdp_utils import offload_fsdp_optimizer, offload_fsdp_model_to_cpu, load_fsdp_optimizer, \
    load_fsdp_model_to_gpu
from verl.utils.import_utils import import_external_libs
from verl.utils.model import compute_position_id_with_mask
from verl.utils.flops_counter import FlopsCounter
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

from codetiming import Timer

import json

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))

def split_response_id(response_ids, attention_mask, src_tokenizer) -> list[tuple[str, str]]:
    # extract response
    response_length = response_ids.shape[-1]
    valid_response_length = attention_mask[-response_length:].sum()
    valid_response_ids = response_ids[:valid_response_length]
    
    # decode
    response: str = src_tokenizer.decode(valid_response_ids, skip_special_tokens=False)
    # remove bos and eos
    response = response.replace(src_tokenizer.eos_token, '')
    
    response = response.split('<think>')[-1].split('</think>')[0]
    
    result = []
    while len(response) > 0:
        if response[0] != '<':
            response = response[1:]
            continue
        spl = response.split('>', 1)
        curr_label, rest = spl[0], spl[-1]
        curr_label = curr_label[1:]
        spl = rest.split(f'</{curr_label}>', 1)
        curr_content, rest = spl[0], spl[-1]
        response = rest
        
        result.append((curr_label, curr_content))
    
    return result

def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])
    else:
        device_mesh = init_device_mesh('cuda',
                                       mesh_shape=(world_size // fsdp_size, fsdp_size),
                                       mesh_dim_names=['ddp', 'fsdp'])
    return device_mesh


def get_sharding_strategy(device_mesh):
    from torch.distributed.fsdp import ShardingStrategy
    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy

unit_class_instruction_template = \
"""## **[Task]**
Given [Text], you should distinguish which class it is belonging to, and directly output the class id. 

The classes are: 
- **[Observation]** 
   * id: 0
   * observe the specific behaviors or attitudes from the current context. If there are multiple things need to be observed, visit step [Observation] multiple times.
   
- **[Attribution and Evaluation]** 
   * id: 1
   * further interprets the result of previous step [Observation], may include Causal reasoning for others' actions / Impact assessment on current context. 
   
- **[Motivation]** 
   * id: 2
   * formulate ones primary drivers of himself, based on his needs / desires identified in other steps. 

- **[Self-regulatory]** 
   * id: 3
   * validate and refine previous thoughts: (1) consider twice to polish the thought, (2) check if there exists more infomation in the scenario need to be considered. 

- **[Self-efficacy]** 
   * id: 4
   * analyze and adjust internal perceptions of the scene and action plan. 

- **[Behavior]** 
   * id: 5
   * derive context-specific behaviors. 

## **[Text]**
{user_input}

[Output]
The class id of the Text is: """

TAG_TO_ID_MAPPING = {
    "Observation": 0,
    "Attribution and Evaluation": 1,
    "Motivation": 3,
    "Self-regulatory": 4,
    "Self-efficacy": 5,
    "Behavior": 2,
}


# TODO(sgm): we may need to extract it to dp_reward_model.py
class RewardModelWorkerConsistency(Worker):
    """
    use AutoModelForTokenClassification for each nodes, and use harmonic mean to merge the results. 
    """

    def __init__(self, config):
        super().__init__()
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh('cuda',
                                                        mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                                                        mesh_dim_names=['dp', 'sp'])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        self.use_remove_padding = self.config.model.get('use_remove_padding', False)

        # normalize config
        if self.config.micro_batch_size is not None:
            self.config.micro_batch_size //= torch.distributed.get_world_size()
            self.config.micro_batch_size_per_gpu = self.config.micro_batch_size
            
        if not hasattr(self.config, "node_merge_method"):
            print("[WARNING] node_merge_method is not specified, use default value 'harmonic'")
            setattr(self.config, "node_merge_method", "harmonic")

    def _build_model(self, config):
        # the following line is necessary
        from transformers import AutoModelForTokenClassification, AutoConfig
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, CPUOffload

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.model.path)

        if self.config.model.input_tokenizer is None:
            self._do_switch_chat_template = False
            raise NotImplementedError("you should specify input_tokenizer to make the consistency reward model work")
        else:
            self._do_switch_chat_template = True
            input_tokenizer_local_path = copy_to_local(config.model.input_tokenizer)
            self.input_tokenizer = hf_tokenizer(input_tokenizer_local_path,
                                                trust_remote_code=config.model.get('trust_remote_code', False))
            self.tokenizer = hf_tokenizer(local_path, trust_remote_code=config.model.get('trust_remote_code', False))

        trust_remote_code = config.model.get('trust_remote_code', False)
        model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        model_config.num_labels = 6

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        init_context = get_init_weight_context_manager(use_meta_tensor=not model_config.tie_word_embeddings,
                                                       mesh=self.device_mesh)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            setattr(model_config, 'classifier_dropout', 0.)
            reward_module = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path=local_path,
                config=model_config,
                torch_dtype=torch.bfloat16,
                attn_implementation='flash_attention_2',
                trust_remote_code=trust_remote_code)

            if config.model.get('use_remove_padding', False) or self.ulysses_sequence_parallel_size > 1:
                from verl.models.transformers.monkey_patch import apply_monkey_patch
                apply_monkey_patch(model=reward_module, ulysses_sp_size=self.ulysses_sequence_parallel_size)

            reward_module.to(torch.bfloat16)

        auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=self.config.model.fsdp_config)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        reward_module = FSDP(
            reward_module,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,  # zero3
            sync_module_states=True,
            cpu_offload=CPUOffload(offload_params=True),
            forward_prefetch=False,
            device_mesh=self.device_mesh)

        return reward_module

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        import_external_libs(self.config.model.get('external_lib', None))
        self.reward_module = self._build_model(config=self.config)

    def _forward_micro_batch(self, micro_batch):
        # 修改前向传播逻辑
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']
            labels = micro_batch['correct_labels']
            batch_size, seqlen = input_ids.shape

            # 直接使用完整输入格式，价值头模型通常需要完整序列
            outputs = self.reward_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                return_dict=True
            )

            # 获取输出
            logits = outputs.logits # (batch_size, seq_len, num_labels)
            # 过一遍softmax
            probs = torch.softmax(logits, dim=-1)
            
            # 提取每个节点所有位置对应的正确标签的概率
            # 原始概率形状 probs: (batch_size, seq_len, num_labels)
            # 正确标签形状 labels: (batch_size, ), 值域 num_labels
            view_labels = labels.unsqueeze(1).expand(-1, seqlen)
            index = view_labels.unsqueeze(-1)
            # 使用 gather 函数提取每个位置上正确标签的概率
            # dim=2: 指定在 num_labels 维度上进行索引
            # index: 提供了在 dim=2 上要选择的索引值                    
            gathered_probs = torch.gather(probs, dim=2, index=index)

            # gather 的输出形状与 index 相同 (batch_size, seq_len, 1)
            # 我们可以使用 squeeze(-1) 去掉最后一个多余的维度
            # 最终形状为 (batch_size, seq_len)
            values = gathered_probs.squeeze(-1)
                    
            assert values.shape == (batch_size, seqlen), f'Unexpected shape of value: {values.shape}, expected: {(batch_size, seqlen)}'

            # 提取最后一个有效位置的奖励值
            # 使用position_ids和attention_mask定位有效序列结束位置
            eos_mask = (position_ids * attention_mask)  # 结合位置编码和注意力掩码
            eos_mask_idx = torch.argmax(eos_mask, dim=-1)  # (batch_size,)
            
            # if self.rank == 0:
            #     with open("tmp/rm_c_probs_and_values.txt", "a") as f:
            #         # f.write(f"input_text: {self.tokenizer.decode(input_ids[0][:eos_mask_idx[0]+1], skip_special_tokens=False)}\n")
            #         f.write(f"probs: {probs[0][eos_mask_idx[0]]}; values: {values[0][eos_mask_idx[0]]}; labels: {labels[0]}; pos: {eos_mask_idx[0]}; \n")
            #         # f.write(f"probs: {probs[0][eos_mask_idx[0]-1]}; values: {values[0][eos_mask_idx[0]-1]}; labels: {labels[0]}; pos: {eos_mask_idx[0]-1}; \n")
            #         # f.write(f"probs: {probs[0][eos_mask_idx[0]+1]}; values: {values[0][eos_mask_idx[0]]}; labels: {labels[0]}; pos: {eos_mask_idx[0]+1}; \n")
            
            # 使用高级索引获取对应位置的奖励值
            rm_score = values[torch.arange(batch_size), eos_mask_idx]
            
            # 确保最终输出形状正确
            assert rm_score.shape == (batch_size,), \
                f"Final reward shape mismatch: {rm_score.shape} vs expected ({batch_size},)"

            return rm_score

    def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor):
        batch_size = data.batch.batch_size[0]
        # expand as token_level_reward
        attention_mask = data.batch['attention_mask']
        position_ids = data.batch['position_ids']
        response_length = data.batch['responses'].shape[-1]
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)  # (bsz, seqlen)
        token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores

        # select the response part
        token_level_scores = token_level_scores[:, -response_length:]

        return token_level_scores

    def _switch_chat_template(self, data: DataProto, to_num: int):
        src_max_length = 8192

        src_tokenizer = self.input_tokenizer
        target_tokenizer = self.tokenizer

        rm_input_ids = []
        rm_attention_mask = []
        data_id = []
        correct_label = []

        def insert_data(response: str, label: str, idx: int):
            """准备一条数据"""
            if label not in TAG_TO_ID_MAPPING.keys():
                response = "none"
                label_int = 0
            else:
                label_int = TAG_TO_ID_MAPPING[label]
            
            if self.rank == 0 and idx == 0:
                # with open("tmp/rm_c_example.txt", "a") as f:
                #     f.write(f"response: {response}, label: {label}, label_int: {label_int}\n")
                print(f"node example: {response}, label: {label}, label_int: {label_int}")
            
            prompt = unit_class_instruction_template.format(
                user_input=response,
            )
            chat = [{'role': 'user', 'content': prompt}]

            prompt_with_chat_template: str = target_tokenizer.apply_chat_template(chat,
                add_generation_prompt=False,
                tokenize=False)
            prompt_with_chat_template = prompt_with_chat_template.strip()
            while prompt_with_chat_template[-1] in [' ', '\n', '\t', '\r']:
                prompt_with_chat_template = prompt_with_chat_template[:-1]
            if prompt_with_chat_template.endswith('<|im_end|>'):
                prompt_with_chat_template = prompt_with_chat_template[:-len('<|im_end|>')]
            if self.rank == 0 and idx == 0:
                # for debugging purpose
                print(f'Switch template. chat: {prompt_with_chat_template}')

            # the maximum length is actually determined by the reward model itself
            max_length = self.config.get('max_length', src_max_length)
            if max_length is None:
                max_length = src_max_length

            model_inputs = target_tokenizer(prompt_with_chat_template, return_tensors='pt', add_special_tokens=False)
            input_ids, attention_mask = verl_F.postprocess_data(
                input_ids=model_inputs['input_ids'],
                attention_mask=model_inputs['attention_mask'],
                max_length=max_length,
                pad_token_id=target_tokenizer.pad_token_id,
                left_pad=False,  # right padding
                truncation=self.config.get('truncation', 'right'))  # truncate from the right

            rm_input_ids.append(input_ids)
            rm_attention_mask.append(attention_mask)
            data_id.append(idx)
            correct_label.append(label_int)

        for i in range(data.batch.batch_size[0]):
            insert_data(data.non_tensor_batch['content'][i], data.non_tensor_batch['label'][i], i)

        while len(rm_input_ids) < to_num:
            rm_input_ids.append(rm_input_ids[-1])
            rm_attention_mask.append(rm_attention_mask[-1])
            data_id.append(-1)
            correct_label.append(correct_label[-1])

        rm_input_ids_tensor = torch.cat(rm_input_ids, dim=0)
        rm_attention_mask_tensor = torch.cat(rm_attention_mask, dim=0)
        correct_label_tensor = torch.tensor(correct_label)

        rm_position_ids = compute_position_id_with_mask(rm_attention_mask_tensor)

        rm_inputs = {'input_ids': rm_input_ids_tensor, 'attention_mask': rm_attention_mask_tensor, 'position_ids': rm_position_ids, 'correct_labels': correct_label_tensor}

        return DataProto.from_dict(rm_inputs)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        import itertools
        from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
        # Support all hardwares
        data = data.to(torch.cuda.current_device())

        to_num = data.batch.batch_size[0]
        
        print(f"rank = {self.rank}, to_num = {to_num}")
        while to_num % self.config.micro_batch_size_per_gpu != 0:
            to_num += 1
        
        if self._do_switch_chat_template:
            rm_data = self._switch_chat_template(data, to_num)
        else:
            assert False, "consistency reward model should use switch chat template"
        
        # with open(f"tmp/rm_c_doing_{self.rank}.txt", "a") as f:
        #     f.write(f"Rank {self.rank} is computing rm score with {rm_data.batch['input_ids'].shape} input_ids\n")
        
        # Support all hardwares
        rm_data.batch = rm_data.batch.to(torch.cuda.current_device())

        print(f"Rank {self.rank} is computing rm score with {rm_data.batch['input_ids'].shape} input_ids")

        # perform forward computation
        with self.ulysses_sharding_manager:
            rm_data = self.ulysses_sharding_manager.preprocess_data(data=rm_data)
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            use_dynamic_bsz = self.config.use_dynamic_bsz
            if use_dynamic_bsz:
                max_token_len = self.config.forward_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, indices = rearrange_micro_batches(batch=rm_data.batch, max_token_len=max_token_len)
            else:
                micro_batches = rm_data.batch.split(self.config.micro_batch_size_per_gpu)
            output = []
            for idx, micro_batch in enumerate(micro_batches):
                # if self.rank == 0 and idx == 0:
                #     print(f'Forward micro batch {idx}')
                #     print(f"{self.tokenizer.decode(micro_batch['input_ids'][0])}")
                # with open(f"tmp/rm_c_doing_{self.rank}.txt", "a") as f:
                #     f.write(f"  micro batch: Rank {self.rank} is computing rm score with {micro_batch['input_ids'].shape} input_ids\n")
                rm_score = self._forward_micro_batch(micro_batch)
                output.append(rm_score)
                # with open(f"tmp/rm_c_doing_{self.rank}.txt", "a") as f:
                #     f.write(f"  micro batch: Rank {self.rank} is done computing rm score with rm_score {rm_score}\n")
            scores = torch.cat(output, dim=0)  # (batch_size)

            if use_dynamic_bsz:
                print("in use_dynamic_bsz")
                indices = list(itertools.chain.from_iterable(indices))
                assert len(indices) == scores.size(0), f"{len(indices)} vs. {scores.size()}"
                revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                scores = scores[revert_indices]

            # Note that this is only the scores, may not be the final rewards used to train RL
            output = DataProto.from_dict(tensors={'rm_consistency_scores': scores})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        # print("rm_consistency_scores: ", output)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        self.reward_module._handle.reshard(True)
 
        output = output.to('cpu')
        # with open(f"tmp/rm_c_doing_{self.rank}.txt", "a") as f:
        #     f.write(f"Rank {self.rank} is done computing rm score")
        print(f"Rank {self.rank} is done computing rm score\n")
        return output
