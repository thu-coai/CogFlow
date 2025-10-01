# Copyright 2025 the LlamaFactory team.
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

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Optional

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import DatasetProcessor, infer_seqlen


if TYPE_CHECKING:
    from ..mm_plugin import AudioInput, ImageInput, VideoInput


logger = logging.get_logger(__name__)


class ClassificationDatasetProcessor(DatasetProcessor):
    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        label_class: int, 
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        messages = self.template.mm_plugin.process_messages(
            prompt + [{"role": "assistant", "content": "?"}], images, videos, audios, self.processor
        )
        prompt_ids, message_ids = self.template.encode_oneturn(self.tokenizer, messages, system, tools)

        if self.template.efficient_eos:
            message_ids += [self.tokenizer.eos_token_id]

        prompt_ids, _ = self.template.mm_plugin.process_token_ids(
            prompt_ids, None, images, videos, audios, self.tokenizer, self.processor
        )
        # consider the response is more important
        source_len, target_len = infer_seqlen(
            len(prompt_ids), len(message_ids), self.data_args.cutoff_len
        )
        prompt_ids = prompt_ids[:source_len]
        message_ids = message_ids[:target_len]

        input_ids = prompt_ids
        while input_ids[-1] != self.tokenizer.encode("<|im_end|>")[0]:
            input_ids = input_ids[:-1]
        input_ids = input_ids[:-1]
        # print(f"input_ids = {self.tokenizer.decode(input_ids)[-100:]} (after decline)")
        input_labels = [-100] * (len(input_ids)-1) + [label_class]
        return input_ids, input_labels

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build input pairs with format `<bos> X`, `Y <eos>`
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):

            input_ids, input_labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                label_class=examples["_label"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(input_labels)
            
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        print("input_ids:\n{}".format(example["input_ids"]))
        print(
            "inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False))
        )
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        print("label_ids:\n{}".format(example["labels"]))
        print(f"decoded_labels:{valid_labels}\nend token id: {self.tokenizer.encode('<|im_end|>')[0]}")
