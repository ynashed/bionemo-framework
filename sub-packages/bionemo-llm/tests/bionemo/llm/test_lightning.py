# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
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


from typing import Dict, List
from unittest import mock

import pytest
import torch
from torch import nn

from bionemo.llm import lightning as bnptl
from bionemo.llm.lightning import MegatronStrategy, PerplexityLoggingCallback, batch_collator, get_dtype_device
from bionemo.testing import megatron_parallel_state_utils


def test_batch_collate_tuple():
    result = batch_collator(tuple((torch.tensor([i]), torch.tensor([i + 1])) for i in range(10)))
    assert isinstance(result, tuple), "expect output container to be the same type as input (tuple)"
    assert torch.equal(result[0], torch.tensor(list(range(10))))
    assert torch.equal(result[1], torch.tensor([i + 1 for i in range(10)]))


def test_batch_collate_dict():
    result = batch_collator(
        [{"fixed key1": torch.tensor([i]), "fixed key2": torch.tensor([i + 1])} for i in range(10)]
    )
    assert isinstance(result, dict), "expect output container to be the same type as input (dict)"
    assert torch.equal(result["fixed key1"], torch.tensor(list(range(10))))
    assert torch.equal(result["fixed key2"], torch.tensor([i + 1 for i in range(10)]))


def test_batch_collate_list():
    result = batch_collator([[torch.tensor([i]), torch.tensor([i + 1])] for i in range(10)])
    assert isinstance(result, list), "expect output container to be the same type as input (list)"
    assert torch.equal(result[0], torch.tensor(list(range(10))))
    assert torch.equal(result[1], torch.tensor([i + 1 for i in range(10)]))


def test_batch_collate_none():
    assert batch_collator(None) is None


def test_batch_collator_tensor_fails():
    with pytest.raises(ValueError, match="Unsupported input structure in batch_collator"):
        batch_collator(torch.tensor([[torch.tensor([i]), torch.tensor([i + 1])] for i in range(10)]))


def test_batch_collator_primitive_fails():
    with pytest.raises(ValueError, match="Unsupported input structure in batch_collator"):
        batch_collator(4)


def test_batch_collator_emptylist_fails():
    with pytest.raises(ValueError, match="Cannot process an empty sequence"):
        batch_collator([])


def test_batch_collator_emptytuple_fails():
    with pytest.raises(ValueError, match="Cannot process an empty sequence"):
        batch_collator(())


def test_batch_collator_emptyset_fails():
    with pytest.raises(ValueError, match="Unsupported input structure in batch_collator"):
        batch_collator(set())


def test_batch_collator_emptydict_fails():
    with pytest.raises(ValueError, match="Unsupported input structure in batch_collator"):
        batch_collator({})


def test_tensor_dtype():
    tensor = torch.tensor(4.0, dtype=torch.float32)
    dtype, _ = get_dtype_device(tensor)
    assert dtype == torch.float32


def test_module_dtype():
    module = MyModule(dtype=torch.float32)
    dtype, _ = get_dtype_device(module)
    assert dtype == torch.float32


def test_nested_dtype():
    module = MyModule(dtype=torch.float32)
    nested = NestedModule(module)
    dtype, _ = get_dtype_device(nested)
    assert dtype == torch.float32


def test_dict_tensor_dtype():
    dtype, _ = get_dtype_device({"tensor": torch.tensor(5, dtype=torch.float32)})
    assert dtype == torch.float32


# Handles the cases where we pass in a valid type, but it does not have an associated dtype
def test_empty_module():
    # Module with no underlying parameters
    empty = MyModuleEmpty()
    with pytest.raises(ValueError, match="Cannot get dtype on a torch module with no parameters."):
        get_dtype_device(empty)


def test_none_fails():
    with pytest.raises(ValueError, match="non-None value not found"):
        get_dtype_device([None, None])


def test_empty_dict_fails():
    with pytest.raises(ValueError, match="Looking up dtype on an empty dict"):
        get_dtype_device({})


def test_empty_list_fails():
    with pytest.raises(ValueError, match="Looking up dtype on an empty list"):
        get_dtype_device([])


def test_garbage_fails():
    # String not a valid input type, should work for other garbage values too.
    with pytest.raises(TypeError, match="Got something we didnt expect"):
        get_dtype_device("flkasdflasd")


class MyModule(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(10, 10, dtype=dtype) for i in range(10)])
        self.others = nn.ModuleList([nn.Linear(10, 10, dtype=dtype) for i in range(10)])

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        for i, linear in enumerate(self.linears):
            x = linear(x)
        return x


class MyModuleEmpty(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class NestedModule(nn.Module):
    def __init__(self, other):
        super().__init__()
        self.other = other

    def forward(self, x):
        return self.other(x)


def test_mixin_strategy_contract_get_loss_reduction():
    with megatron_parallel_state_utils.clean_parallel_state_context():
        strategy = MegatronStrategy(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            ddp="megatron",
            find_unused_parameters=True,
            enable_nemo_ckpt_io=False,
        )
        strategy.connect(bnptl.LightningPassthroughPredictionMixin())
        mixin = bnptl.LightningPassthroughPredictionMixin()
        strategy_reduction_function = strategy._get_loss_reduction("predict")
        assert isinstance(strategy_reduction_function(mixin), bnptl.PassthroughLossReduction)


def test_PerplexityLoggingCallback_golden_values_in_validation():
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        mock_pl_module = mock.MagicMock()
        mock_pl_module.log.return_value = None

        mock_trainer = mock.MagicMock()
        mock_trainer.training = False

        callback = PerplexityLoggingCallback(log_train=False, log_val=True)

        def call_on_megatron_reduce_microbatches_end(
            max_sequence_length: int,
            microbatch_size: int,
            num_microbatches: int,
            microbatch_outputs: List[Dict[str, Dict[str, torch.Tensor]]],
        ) -> None:
            callback.on_megatron_reduce_microbatches_end(
                data=None,
                forward_only=None,
                data_step=None,
                forward_step=None,
                loss_reduction=None,
                seq_length=max_sequence_length,
                micro_batch_size=microbatch_size,
                num_microbatches=num_microbatches,
                wrap_forward_step=None,
                pipeline=None,
                use_global_batch_sampler=None,
                data_iterator=None,
                pl_module=mock_pl_module,
                trainer=mock_trainer,
                microbatch_outputs=microbatch_outputs,
                loss_mean=None,
            )

        # 1. Test with a single microbatch
        microbatch_size, max_sequence_length, vocab_size = 1, 1024, 2
        num_microbatches = 1
        microbatch_outputs = [
            {
                "batch": {
                    "labels": torch.zeros(microbatch_size, max_sequence_length, dtype=torch.long),  # [b s]
                    "loss_mask": torch.ones(microbatch_size, max_sequence_length, dtype=torch.long),  # [b s]
                },
                "forward_out": {
                    "token_logits": torch.ones(microbatch_size, max_sequence_length, vocab_size),  # [b s v]
                },
            },
        ]

        call_on_megatron_reduce_microbatches_end(
            max_sequence_length=max_sequence_length,
            microbatch_size=microbatch_size,
            num_microbatches=num_microbatches,
            microbatch_outputs=microbatch_outputs,
        )

        val_ppl = mock_pl_module.log.call_args[0][1]
        torch.testing.assert_close(
            val_ppl, torch.tensor(vocab_size, dtype=torch.float32), msg="fail test on single microbatch"
        )

        # 2. Test with a single microbatch with masking
        microbatch_size, max_sequence_length, vocab_size = 1, 1024, 2
        num_microbatches = 1
        microbatch_outputs = [
            {
                "batch": {
                    "labels": torch.zeros(microbatch_size, max_sequence_length, dtype=torch.long),  # [b s]
                    "loss_mask": torch.ones(microbatch_size, max_sequence_length, dtype=torch.long),  # [b s]
                },
                "forward_out": {
                    "token_logits": torch.ones(microbatch_size, max_sequence_length, vocab_size),  # [b s v]
                },
            },
        ]

        half_idx = max_sequence_length // 2
        microbatch_outputs[0]["batch"]["labels"][:, :half_idx] = -100
        microbatch_outputs[0]["batch"]["loss_mask"][:, :half_idx] = 0

        call_on_megatron_reduce_microbatches_end(
            max_sequence_length=max_sequence_length,
            microbatch_size=microbatch_size,
            num_microbatches=num_microbatches,
            microbatch_outputs=microbatch_outputs,
        )

        val_ppl = mock_pl_module.log.call_args[0][1]
        torch.testing.assert_close(
            val_ppl, torch.tensor(vocab_size, dtype=torch.float32), msg="fail test on single microbatch with masking"
        )

        # 3. Test with multiple microbatches with variable length
        microbatch_size, max_sequence_length, vocab_size = 2, 1024, 2
        num_microbatches = 2
        microbatch_outputs = [
            {
                "batch": {
                    "labels": torch.zeros(microbatch_size, max_sequence_length // 2, dtype=torch.long),  # [b s]
                    "loss_mask": torch.ones(microbatch_size, max_sequence_length // 2, dtype=torch.long),  # [b s]
                },
                "forward_out": {
                    "token_logits": torch.ones(microbatch_size, max_sequence_length // 2, vocab_size),  # [b s v]
                },
            },
            {
                "batch": {
                    "labels": torch.zeros(microbatch_size, max_sequence_length, dtype=torch.long),  # [b s]
                    "loss_mask": torch.ones(microbatch_size, max_sequence_length, dtype=torch.long),  # [b s]
                },
                "forward_out": {
                    "token_logits": torch.ones(microbatch_size, max_sequence_length, vocab_size),  # [b s v]
                },
            },
        ]

        call_on_megatron_reduce_microbatches_end(
            max_sequence_length=max_sequence_length,
            microbatch_size=microbatch_size,
            num_microbatches=num_microbatches,
            microbatch_outputs=microbatch_outputs,
        )

        val_ppl = mock_pl_module.log.call_args[0][1]
        torch.testing.assert_close(
            val_ppl,
            torch.tensor(vocab_size, dtype=torch.float32),
            msg="fail test on multiple microbatches with variable length",
        )
