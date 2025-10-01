# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer.py
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

import json
import os
from types import MethodType
from typing import TYPE_CHECKING, Optional, Union

import torch
from transformers import Trainer
from typing_extensions import override

from ...extras import logging
from ...extras.packages import is_transformers_version_greater_than
from ..callbacks import FixValueHeadModelCallback, SaveProcessorCallback
from ..trainer_utils import create_custom_optimizer, create_custom_scheduler


if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments


logger = logging.get_logger(__name__)

accuracy_list = []
accuracy0_list = []
eval_accuracy_list = []
MSE_list = []
eval_MSE_list = []

class ClassifyTrainer(Trainer):
    r"""Trainer for classification tasks."""

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        if is_transformers_version_greater_than("4.46"):
            kwargs["processing_class"] = kwargs.pop("tokenizer")

        super().__init__(**kwargs)
        self.model_accepts_loss_kwargs = False  # overwrite trainer's default behavior
        self.finetuning_args = finetuning_args
        self.can_return_loss = True  # override property to return eval_loss
        # self.add_callback(FixValueHeadModelCallback)

        if processor is not None:
            self.add_callback(SaveProcessorCallback(processor))

        if finetuning_args.use_badam:
            from badam import BAdamCallback, clip_grad_norm_old_version  # type: ignore

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_old_version, self.accelerator)
            self.add_callback(BAdamCallback)

    @override
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimizer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    @override
    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    @override
    def _get_train_sampler(self) -> Optional["torch.utils.data.Sampler"]:
        if self.finetuning_args.disable_shuffling:
            return torch.utils.data.SequentialSampler(self.train_dataset)

        return super()._get_train_sampler()

    @override
    def compute_loss(
        self, model: "PreTrainedModel", inputs: dict[str, "torch.Tensor"], return_outputs: bool = False, **kwargs
    ) -> Union["torch.Tensor", tuple["torch.Tensor", list["torch.Tensor"]]]:
        # the original implementation of compute_loss
        outputs = model(**inputs)
        loss = outputs.loss
        
        # calculate the accuracy and MSE
        with torch.no_grad():
            preds = torch.argmax(outputs.logits, dim=-1)
            labels = inputs["labels"]
            
            # filter out the invalid labels -100
            mask = labels != -100
            valid_preds = preds[mask]
            valid_labels = labels[mask]

            # print(f"valid_labels: {valid_labels}")
            # print(f"valid_preds : {valid_preds}")
            
            accuracy = (valid_preds == valid_labels).float().mean().item()
            mse = torch.nn.functional.mse_loss(valid_preds.float(), valid_labels.float()).item()
        
        # 记录到训练指标
        if return_outputs:
            if len(accuracy_list) != 0:
                prefix = "train_"
                self.log({
                    prefix+"ACCURACY": sum(accuracy_list) / len(accuracy_list),
                    prefix+"MSE": sum(MSE_list) / len(MSE_list),
                })
                accuracy_list.clear()
                MSE_list.clear()
            eval_accuracy_list.append(accuracy)
            eval_MSE_list.append(mse)
        else:
            if len(eval_accuracy_list) != 0:
                prefix = "eval_"
                self.log({
                    prefix+"ACCURACY": sum(eval_accuracy_list) / len(eval_accuracy_list),
                    prefix+"MSE": sum(eval_MSE_list) / len(eval_MSE_list),
                })
                eval_accuracy_list.clear()
                eval_MSE_list.clear()
            accuracy_list.append(accuracy)
            MSE_list.append(mse)
            if self.state.global_step % self.args.logging_steps == 0:
                prefix = "train_"
                if len(accuracy_list) >= self.args.logging_steps:
                    self.log({
                        prefix+"ACCURACY": sum(accuracy_list) / len(accuracy_list),
                        prefix+"MSE": sum(MSE_list) / len(MSE_list),
                    })
                    accuracy_list.clear()
                    MSE_list.clear()
        
        return (loss, outputs) if return_outputs else loss

    def save_predictions(self, predict_results: "PredictionOutput") -> None:
        r"""Save model predictions to `output_dir`.

        A custom behavior that not contained in Seq2SeqTrainer.
        """
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info_rank0(f"Saving prediction results to {output_prediction_file}")
        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            writer.write(predict_results.predictions)
        
        # chosen_scores, rejected_scores = predict_results.predictions

        # with open(output_prediction_file, "w", encoding="utf-8") as writer:
        #     res: list[str] = []
        #     for c_score, r_score in zip(chosen_scores, rejected_scores):
        #         res.append(json.dumps({"chosen": round(float(c_score), 2), "rejected": round(float(r_score), 2)}))

        #     writer.write("\n".join(res))
