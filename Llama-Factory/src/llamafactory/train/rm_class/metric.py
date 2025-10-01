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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np

from ...extras.misc import numpify


if TYPE_CHECKING:
    from transformers import EvalPrediction


@dataclass
class ComputeAccuracy:
    r"""Compute reward accuracy and support `batch_eval_metrics`. """

    def _dump(self) -> Optional[dict[str, float]]:
        result = None
        if hasattr(self, "score_dict"):
            result = {k: float(np.mean(v)) for k, v in self.score_dict.items()}

        self.score_dict = {"accuracy": [], "mse": []}
        return result

    def __post_init__(self):
        self._dump()

    def __call__(self, eval_preds: "EvalPrediction", compute_result: bool = True) -> Optional[dict[str, float]]:
        predictions = numpify(eval_preds.predictions)  # (batch_size, seq_len, num_labels)
        labels = numpify(eval_preds.label_ids)         # (batch_size, seq_len)
        
        mask = labels != -100 
        predictions = predictions[mask]  # (bs, valid_seq_len, num_labels)
        labels = labels[mask]  # (bs, valid_seq_len)
        print(f"label shape: {labels.shape}")
        print(f"predi shape: {predictions.shape}")
        
        # calculate accuracy
        accuracy = np.mean((predictions.argmax(-1) == labels).astype(np.float32))
        
        # calculate MSE (based on the weighted sum of squared differences of labels)（
        num_labels = predictions.shape[-1]
        k = np.arange(num_labels)  # the value of label [0, 1, 2, ..., num_labels-1]
        #  (real_label - label)^2 
        diff_sq = (labels[..., np.newaxis] - k) ** 2  # shape: (batch, seq, num_labels)
        prob = np.exp(predictions) / np.sum(np.exp(predictions), axis=-1, keepdims=True)  # shape: (batch, seq, num_labels)
        # calculate the weighted MSE：Σ P(label) * (real_label - label)^2
        mse_per_element = np.sum(prob * diff_sq, axis=-1)  # shape: (batch, seq)
        mse = np.mean(mse_per_element) 
        
        self.score_dict["accuracy"].append(accuracy)
        self.score_dict["mse"].append(mse)


        if compute_result:
            return self._dump()
