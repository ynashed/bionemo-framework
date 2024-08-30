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


from typing import Dict, Optional, Protocol, Sequence, TypedDict, cast

import torch
import torch.distributed
from apex.optimizers import FusedAdam
from megatron.core import parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
from nemo.collections.common.tokenizers.tokenizer_spec import TokenizerSpec
from nemo.lightning.megatron_parallel import DataT, MegatronLossReduction
from nemo.lightning.pytorch.optim import MegatronOptimizerModule
from torch import Tensor

from bionemo.llm.lightning import BionemoLightningModule, default_megatron_optimizer
from bionemo.llm.model.biobert.model import BioBertConfig, MegatronBioBertModel


__all__: Sequence[str] = (
    "biobert_lightning_module",
    "biobert_data_step",
    "bert_forward_step",
    "bert_default_optimizer",
    "BertModel",
    "BertBatch",
    "SequenceBatch",
)


class BertModel(Protocol[DataT]):
    def forward(
        self, input_ids: Tensor, attention_mask: Tensor, packed_seq_params: Optional[Tensor] = None
    ) -> DataT: ...


class BertBatchCore(TypedDict):
    text: Tensor
    attention_mask: Tensor


class BertBatch(BertBatchCore, total=False):
    cu_seqlens: Tensor


class SequenceBatchCore(TypedDict):
    cu_seqlens: Tensor


class SequenceBatch(SequenceBatchCore, total=False):
    cu_seqlens_argmin: Tensor
    max_seqlen: Tensor


def biobert_lightning_module(
    config: BioBertConfig,
    optimizer: Optional[MegatronOptimizerModule] = None,
    tokenizer: Optional[TokenizerSpec] = None,
) -> BionemoLightningModule[MegatronBioBertModel, MegatronLossReduction]:
    """A pytorch lightning module for BioBert-derived models. This module is designed to be used with the Megatron-LM strategy and nemo 2.0 conventions.
    To change the your loss, pass in a different config object that returns a different loss reduction class. To change your model and what it outputs,
    pass in a different config object that returns a different model. Do not modify this function unless you need to change higher level logic. You may
    need to modify the various step and forward functions towards the bottom of this file to handle new/different keys in the batch. In the future some of
    those functions may need to be refactored out into the config object or a different place so that they live closer to the model definition.
    """

    return BionemoLightningModule(
        config=config,
        optimizer=optimizer if optimizer is not None else default_megatron_optimizer(),
        data_step=biobert_data_step,
        forward_step=bert_forward_step,
        tokenizer=tokenizer,
    )


############################################################################################################
# Below are static helper functions for handling various steps in the above class.


def biobert_data_step(dataloader_iter) -> Dict[str, Tensor]:
    """Preprocesses a batch of data for the GeneFormer model, and ingest a single batch of data from the dataloader iterator.
        only necessary batch keys are subsetted and passed to the model's forward pass, and the loss forward pass, depending on stage.
        TODO document how parallel_state pipeline stages work.

    Args:
        dataloader_iter: An iterator over the dataloader.

    Returns:
        output: A dictionary of this batch limiting to relevant keys.

    """  # noqa: D205
    # Based on: https://github.com/NVIDIA/Megatron-LM/blob/main/pretrain_gpt.py#L87
    # https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/nlp/models/language_modeling/megatron_gpt_model.py#L828-L842

    batch = next(dataloader_iter)

    _batch: dict
    if isinstance(batch, tuple) and len(batch) == 3:
        _batch = batch[0]
    else:
        _batch = batch

    required_keys = set()
    required_keys.add("attention_mask")
    if parallel_state.is_pipeline_first_stage():
        required_keys.add("text")
    if parallel_state.is_pipeline_last_stage():
        required_keys.update(("labels", "loss_mask", "types", "is_random"))
    # if self.get_attention_mask_from_fusion:
    #     required_keys.remove('attention_mask')

    _batch = {key: val.cuda(non_blocking=True) if key in required_keys else None for key, val in _batch.items()}
    # slice batch along sequence dimension for context parallelism
    output = get_batch_on_this_context_parallel_rank(_batch)

    return output


def bert_forward_step(model: BertModel[DataT], batch: BertBatch) -> DataT:
    """This subsets the batch keys to the ones actually used by forward pass of the model, and then calls the model's forward pass.
    if "cu_seqsens" are defined in the batch, then the packed sequence parameters are also passed to the model for forward pass efficiency.
    """  # noqa: D205
    forward_args = {
        "input_ids": batch["text"],
        "attention_mask": batch["attention_mask"],
        # TODO support tokentypes when they are meaningful.
        # "tokentype_ids": batch.get("types", None),
    }

    if "cu_seqlens" in batch:
        forward_args["packed_seq_params"] = get_packed_seq_params(cast(SequenceBatch, batch))

    forward_results = model.forward(**forward_args)
    # TODO support losses that also include the binary head, this means doing something more fancy than the one
    #      default GPT reduction function above MaskedTokenLossReduction()
    return forward_results


def bert_default_optimizer(model: torch.nn.Module) -> FusedAdam:
    """Returns the default optimizer for the BERT model.

    Args:
        model: The BERT model.

    Returns:
        The default optimizer initialized for this BERT module's parameters.
        Uses a learning rate of 1e-4 and weight decay of 1e-2.
    """
    return FusedAdam(model.parameters(), lr=1e-4, weight_decay=0.01)


def get_batch_on_this_context_parallel_rank(batch: Dict[str, Tensor], in_place: bool = True) -> Dict[str, Tensor]:
    """Ensures that the input batch is in the right format for context parallel rank.

    Modifies the batch data based on the context parallel rank, if the context parallel world size is greater than 1.
    Otherwise, the batch is returned as-is.

    Args:
        batch: The input batch data.
        in_place: If true, then the input is mutated. The returned dict is a reference to the input.
                  Otherwise, the input data is always shallow-copied and this copy is modified and returned.

    Returns:
        dict: The modified batch data based on the context parallel rank.
    """

    if not in_place:
        batch = dict(**batch)

    if cp_size := parallel_state.get_context_parallel_world_size() > 1:
        num_valid_tokens_in_ub = None
        if "loss_mask" in batch and batch["loss_mask"] is not None:
            num_valid_tokens_in_ub = batch["loss_mask"].sum()

        cp_rank = parallel_state.get_context_parallel_rank()
        for key, val in batch.items():
            if val is not None:
                seq_dim = 1 if key != "attention_mask" else 2
                _val = val.view(
                    *val.shape[0:seq_dim],
                    2 * cp_size,
                    val.shape[seq_dim] // (2 * cp_size),
                    *val.shape[(seq_dim + 1) :],
                )
                index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device="cpu", pin_memory=True).cuda(
                    non_blocking=True
                )
                _val = _val.index_select(seq_dim, index)
                _val = _val.view(*val.shape[0:seq_dim], -1, *_val.shape[(seq_dim + 2) :])
                batch[key] = _val
        batch["num_valid_tokens_in_ub"] = num_valid_tokens_in_ub

    return batch


def get_packed_seq_params(batch: SequenceBatch) -> PackedSeqParams:
    """Get the packed sequence parameters for the given batch.

    This function should only be called if `cu_seqlens` is defined in the batch.

    Args:
        batch: The input batch to pack.

    Returns:
        PackedSeqParams: The packed sequence parameters containing the following attributes:
            - cu_seqlens_q (Tensor): The sequence lengths for query.
            - cu_seqlens_kv (Tensor): The sequence lengths for key and value.
            - max_seqlen_q (Tensor, optional): The maximum sequence length for query.
            - max_seqlen_kv (Tensor, optional): The maximum sequence length for key and value.
            - qkv_format (str): The format of query, key, and value tensors.

    """
    cu_seqlens = batch["cu_seqlens"].squeeze()  # remove batch size dimension (mbs=1)
    # remove -1 "paddings" added in collate_fn
    if cu_seqlens_argmin := batch.get("cu_seqlens_argmin", None) is not None:
        # pre-compute cu_seqlens_argmin in dataset class for perf
        cu_seqlens = cu_seqlens[: cu_seqlens_argmin.item()]
    else:
        cu_seqlens = cu_seqlens[: torch.argmin(cu_seqlens)]

    # pre-compute max_seqlens in dataset class for perf
    max_seqlen = batch["max_seqlen"].squeeze() if "max_seqlen" in batch else None

    # these args are passed eventually into TEDotProductAttention.forward()
    return PackedSeqParams(
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
        qkv_format="thd",
    )
