# Copyright (c) 2022, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

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

from bionemo.model.utils import TrainerBuilder
from omegaconf.omegaconf import OmegaConf, open_dict
from io import StringIO
import pytest

cfg_template = """
trainer:
  devices: {devices}
  num_nodes: {num_nodes}
  accumulate_grad_batches: {accumulate_grad_batches}
model:
  micro_batch_size: {micro_batch_size}
  tensor_model_parallel_size: {tensor_model_parallel_size}
  pipeline_model_parallel_size: {pipeline_model_parallel_size}
"""

def load_cfg_str(cfg):
    return OmegaConf.load(StringIO(cfg))

def test_adjust_config_no_keys():
    cfg = load_cfg_str(cfg_template.format(
        devices=6,
        num_nodes=8,
        accumulate_grad_batches=5,
        micro_batch_size=7,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=3,
    ))

    TrainerBuilder.adjust_config(cfg)
    assert cfg.model.global_batch_size == 280

def test_adjust_config_errors_on_disagreement():
    cfg = load_cfg_str(cfg_template.format(
        devices=6,
        num_nodes=8,
        accumulate_grad_batches=5,
        micro_batch_size=7,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=3,
    ))
    with open_dict(cfg):
        cfg.model.global_batch_size = 240
    with pytest.raises(ValueError):
        TrainerBuilder.adjust_config(cfg)