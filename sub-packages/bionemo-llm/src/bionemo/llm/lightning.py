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


import functools
import inspect
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, TypeVar, Union

import pytorch_lightning as pl
import torch
import torch.distributed
from megatron.core import parallel_state
from nemo import lightning as nl
from nemo.lightning import _strategy_lib
from nemo.lightning.megatron_parallel import (
    CallbackMethods,
    DataT,
    MegatronLossReduction,
    MegatronParallel,
    ReductionT,
    _ModuleStepFunction,
)
from nemo.lightning.pytorch.trainer import Trainer
from pytorch_lightning.accelerators import CPUAccelerator
from pytorch_lightning.loops.fetchers import _DataFetcherWrapper
from pytorch_lightning.utilities.types import STEP_OUTPUT
from typing_extensions import override

from bionemo.llm.model.loss import per_sequence_masked_token_loss, unreduced_token_loss_fn


__all__: Sequence[str] = (
    "get_dtype_device",
    "batch_collator",
    "PassthroughLossReduction",
    "LightningPassthroughPredictionMixin",
    "TypedMegatronCallback",
    "PerplexityLoggingCallback",
)


T = TypeVar("T")
BatchT = TypeVar("BatchT")


def some_first(seq: Iterable[Optional[T]]) -> T:
    """Returns the first non-None value from the sequence or fails"""  # noqa: D415
    for s in seq:
        if s is not None:
            return s
    raise ValueError("non-None value not found")


def get_dtype_device(torch_object) -> Tuple[torch.dtype, torch.device]:  # noqa: D103
    match torch_object:
        case []:
            raise ValueError("Looking up dtype on an empty list")
        case {**data} if not data:
            raise ValueError("Looking up dtype on an empty dict")
        case torch.Tensor(dtype=dtype, device=device):
            return dtype, device
        case torch.nn.Module() as m:
            try:
                p = next(m.parameters())
            except StopIteration as e:
                raise ValueError("Cannot get dtype on a torch module with no parameters.") from e
            return p.dtype, p.device
        case dict(keys=_, values=values):
            val = some_first(values())
            return get_dtype_device(val)
        case list() as l:
            val = some_first(l)
            return get_dtype_device(val)
        case _:
            raise TypeError("Got something we didnt expect")


# NOTE(SKH): These types are all wrong, but are close. The inner type must always be a torch.Tensor, but the outer container should be generic.
def batch_collator(batches: Optional[Union[Tuple[ReductionT], List[ReductionT]]]) -> Optional[ReductionT]:
    """Takes a sequence of batches and collates them into a single batch.
        This is distinct from the standard pytorch default_collator since it does
        not add the batch dimension, it's assumed the batch
        dimension is already present in the input, as would be the case when
        parallelizing across minibatches.

    IMPORTANT: The underlying data primitive _must_ be a torch Tensor. The input to this function is a recurisve type,
    there can be any amount of nesting between dictionaries, tuples, and lists, as long as the inner type is a n-d torch.Tensor.

    Examples:
        Outer container = Dict:
            [{'a': torch.tensor([1]), 'b': torch.tensor([2])}, {'a': torch.tensor([2]), 'b': torch.tensor([3])}] -> {'a': torch.tensor([1, 2]), 'b': torch.tensor([2, 3])}
        Outer container = List:
            [[torch.tensor([1]), torch.tensor([2])], [torch.tensor([2]), torch.tensor([3])]] -> [torch.tensor([1, 2]), torch.tensor([2, 3])]
        Outer container = Tuple:
            ([torch.tensor([1]), torch.tensor([2])], [torch.tensor([2]), torch.tensor([3])]) -> (torch.tensor([1, 2]), torch.tensor([2, 3]))

    Args:
        batches (Optional[Sequence[ReductionT]]): sequence of batches to collate into a single batch.

    Returns:
        A single batch of the same type as the elements of your input sequence.
    """  # noqa: D205
    match batches:
        case [torch.Tensor(), *_]:
            return torch.cat(batches, dim=0)
        case [dict(), *_]:
            return {key: batch_collator([batch[key] for batch in batches]) for key in batches[0]}
        case [tuple(), *_]:
            return tuple(batch_collator([batch[i] for batch in batches]) for i in range(len(batches[0])))
        case [list(), *_]:
            return [batch_collator([batch[i] for batch in batches]) for i in range(len(batches[0]))]
        case None:
            return None
        case []:
            raise ValueError("Cannot process an empty sequence")
        case _:
            raise ValueError("Unsupported input structure in batch_collator")


# TODO(@jstjohn): Properly use the Generic for DataT and ReductionT usage. Define our own batch/output types.
# TODO(@skothenhill): Re-think the generics here- the way that `batch_collator` is expressed, `batches` should be a recursive generic type.
class PassthroughLossReduction(MegatronLossReduction):
    """Internally in NeMo2.0 the forward step is always expected to return a loss reduction class, and forward is expected to return a loss.
    This class hijacks that mechanism to instead pass through the forward output unperturbed as the loss (to enable inference in the predict step), and then the
    reduce method is used to collate the batch of forward outputs into a single batch. This supports the model forward output being a tensor, dict, tuple,
    or list of tensors. The inner type _must always be a torch.Tensor_.
    """  # noqa: D205

    def forward(self, batch: DataT, forward_out: DataT) -> Tuple[torch.Tensor, DataT]:
        """_summary_

        Args:
            batch (DataT): The batch of data that was passed through the model to generate output.
            forward_out (torch.Tensor): The output from your model's forward pass.

        Returns:
            Tuple[torch.Tensor, ReductionT]: A tuple containing the loss tensor (dummy in this case) and the forward output (unmodified).
        """  # noqa: D415
        dtype, device = get_dtype_device(forward_out)
        return torch.zeros(1, device=device, dtype=dtype), forward_out

    def reduce(self, forward_out: List[DataT]) -> DataT:
        """This overrides the standard reduce with a simplified version that just takes a list of your model's forward outputs
            and collates them togehter into a single output.

        Args:
            forward_out (List[ReductionT]): _description_

        Returns:
            ReductionT: _description_
        """  # noqa: D205
        return batch_collator(forward_out)


class LightningPassthroughPredictionMixin:
    """A mixin that allows your model to do inference on the predict step by hijacking the nemo loss
    reduction mechanism and passing the model output through.
    """  # noqa: D205

    def predict_loss_reduction(self) -> PassthroughLossReduction:
        """For the predict step, pass through the forward pass output."""
        return PassthroughLossReduction()


# TODO(@sichu) upstream to NeMo
class MegatronStrategy(nl.MegatronStrategy):
    """Updated MegatronStrategy to support flexible logging callbacks."""

    @override
    def setup_megatron_parallel(self, trainer: pl.Trainer) -> None:
        assert self.model is not None, "Model is not set"

        convert_module_fn = None
        if hasattr(self.precision_plugin, "convert_module"):
            convert_module_fn = self.precision_plugin.convert_module

        self.megatron_parallel = MegatronParallel(
            self.model,
            precision_plugin=self.precision_plugin,
            vp_size=self.virtual_pipeline_model_parallel_size,
            cpu=isinstance(trainer.accelerator, CPUAccelerator),
            ddp_config=self.ddp_config,
            convert_module_fn=convert_module_fn,
        )

        if self._init_model_parallel:
            self.init_model_parallel()

        self.megatron_parallel.trainer = trainer

        # check signature-def of self.model.configure_optimizers to check if there's an optional arg: megatron_parallel
        sig = inspect.signature(self.model.configure_optimizers)
        if "megatron_parallel" in sig.parameters:
            self.model.configure_optimizers = functools.partial(
                self.model.configure_optimizers, megatron_parallel=self.megatron_parallel
            )

        if self._setup_optimizers:
            self.setup_optimizers(trainer)

        self.model = self.megatron_parallel
        self.model.callbacks.add(*getattr(trainer, "callbacks"))  # TODO(@sichu) upstream this bug fix to NeMo2.0
        # MegatronOptimizerModule and WarmupAnnealDecayHoldScheduler inherit from CallbackMethods but didn't override the methods

        if self.data_sampler:
            self.model.callbacks.add(self.data_sampler)

        datamodule = getattr(trainer, "datamodule", None)
        if datamodule:
            self.model.callbacks.add(datamodule)

    @override
    def training_step(self, dataloader_iter, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        assert self.lightning_module is not None
        assert self.model is not None
        kwargs = self._update_step_kwargs(dataloader_iter, kwargs, "training")

        with self.precision_plugin.train_step_context():  # TODO: Do we need this?
            # Set grad to zero.
            for model_chunk in self.model:
                model_chunk.zero_grad_buffer()
            for opt in self.optimizers:
                opt.zero_grad()

            model_outputs: torch.Tensor | Dict[str, torch.Tensor] = self.model(
                dataloader_iter, forward_only=False, *args, **kwargs
            )  # NOTE allow flexible model outputs
            if torch.is_tensor(model_outputs):
                reduced_train_loss = model_outputs
            else:
                reduced_train_loss = model_outputs["loss"]

            self.lightning_module.log(
                "global_step",
                self.trainer.global_step,
                prog_bar=True,
                batch_size=1,
            )

            self.lightning_module.log(
                "step",
                self.trainer.global_step,
            )

            if self.log_memory_usage:
                max_memory_reserved = torch.cuda.max_memory_reserved()
                memory_allocated = torch.cuda.memory_allocated()
                self.lightning_module.log(
                    "peak_memory_usage",
                    max_memory_reserved,
                    prog_bar=True,
                    batch_size=1,
                )
                self.lightning_module.log(
                    "memory_allocated",
                    memory_allocated,
                    prog_bar=True,
                    batch_size=1,
                )

            if self.log_train_loss:
                # p2p now, broadcast later at ckpt. only with pp, some ranks will log 0.0
                # WHICH IS OK because we broadcast later at checkpoint time
                _strategy_lib._sync_from_last_pipeline_stage(reduced_train_loss, broadcast=False)
                self.lightning_module.log(
                    "reduced_train_loss", reduced_train_loss, prog_bar=True, batch_size=1, sync_dist=False
                )

            return model_outputs

    @override
    def validation_step(self, dataloader_iter, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        assert self.lightning_module is not None
        assert self.model is not None
        kwargs = self._update_step_kwargs(dataloader_iter, kwargs, "validation")

        with self.precision_plugin.val_step_context():  # TODO: Do we need this?
            model_outputs = self.model(dataloader_iter, forward_only=True, *args, **kwargs)
            if torch.is_tensor(model_outputs):
                reduced_val_loss = model_outputs
            else:
                reduced_val_loss = model_outputs["loss"]

            from megatron.core import parallel_state

            pp_size = parallel_state.get_pipeline_model_parallel_world_size()
            if pp_size > 1:
                # ranks that are not final pp stage have 0 for loss, and out will be mean-reduced over pp
                # groups (due to sync_dist), which divides val_loss by pp_size. so we multiply by pp_size to cancel out
                self.lightning_module.log(
                    "val_loss",
                    reduced_val_loss * pp_size,
                    prog_bar=True,
                    sync_dist=True,
                    sync_dist_group=parallel_state.get_pipeline_model_parallel_group(),
                    on_epoch=True,
                )
            else:
                self.lightning_module.log("val_loss", reduced_val_loss, prog_bar=True, on_epoch=True)

            return model_outputs


# TODO(@sichu) upstream to NeMo2.0
class TypedMegatronCallback(CallbackMethods):  # noqa: D101
    def on_megatron_step_start(  # noqa: D102
        self,
        data: _DataFetcherWrapper,
        forward_only: bool,
        data_step: _ModuleStepFunction,
        forward_step: _ModuleStepFunction,
        loss_reduction: _ModuleStepFunction,
        seq_length: int,
        micro_batch_size: int,
        num_microbatches: int,
        wrap_forward_step: bool,
        pipeline: MegatronParallel,
        use_global_batch_sampler: bool,
        data_iterator: _DataFetcherWrapper,
        pl_module: MegatronParallel,
        trainer: Trainer,
    ) -> None: ...

    def on_megatron_microbatch_start(
        self,
        data: _DataFetcherWrapper,
        forward_only: bool,
        data_step: _ModuleStepFunction,
        forward_step: _ModuleStepFunction,
        loss_reduction: _ModuleStepFunction,
        seq_length: int,
        micro_batch_size: int,
        num_microbatches: int,
        wrap_forward_step: bool,
        pipeline: MegatronParallel,
        use_global_batch_sampler: bool,
        data_iterator: _DataFetcherWrapper,
        pl_module: MegatronParallel,
        trainer: Trainer,
        # method specific
        batch: Dict,
        forward_callback: MegatronLossReduction,
    ):
        """Same as on_megatron_step_start."""
        ...

    def on_megatron_microbatch_end(  # noqa: D102
        self,
        data: _DataFetcherWrapper,
        forward_only: bool,
        data_step: _ModuleStepFunction,
        forward_step: _ModuleStepFunction,
        loss_reduction: _ModuleStepFunction,
        seq_length: int,
        micro_batch_size: int,
        num_microbatches: int,
        wrap_forward_step: bool,
        pipeline: MegatronParallel,
        use_global_batch_sampler: bool,
        data_iterator: _DataFetcherWrapper,
        pl_module: MegatronParallel,
        trainer: Trainer,
        # method specific
        batch: Dict,
        forward_callback: MegatronLossReduction,
        microbatch_outputs: List[Any],  # outputs from forward method in MegatronLossReduction across microbatches
    ): ...

    def on_megatron_reduce_microbatches_start(
        self,
        data: _DataFetcherWrapper,
        forward_only: bool,
        data_step: _ModuleStepFunction,
        forward_step: _ModuleStepFunction,
        loss_reduction: _ModuleStepFunction,
        seq_length: int,
        micro_batch_size: int,
        num_microbatches: int,
        wrap_forward_step: bool,
        pipeline: MegatronParallel,
        use_global_batch_sampler: bool,
        data_iterator: _DataFetcherWrapper,
        pl_module: MegatronParallel,
        trainer: Trainer,
        # method specific
        microbatch_outputs: List[Any],  # outputs from forward method in MegatronLossReduction across microbatches
    ) -> None:
        """Same as on_megatron_microbatch_end if microbatch_outputs is not None."""
        ...

    def on_megatron_reduce_microbatches_end(  # noqa: D102
        self,
        data: _DataFetcherWrapper,
        forward_only: bool,
        data_step: _ModuleStepFunction,
        forward_step: _ModuleStepFunction,
        loss_reduction: _ModuleStepFunction,
        seq_length: int,
        micro_batch_size: int,
        num_microbatches: int,
        wrap_forward_step: bool,
        pipeline: MegatronParallel,
        use_global_batch_sampler: bool,
        data_iterator: _DataFetcherWrapper,
        pl_module: MegatronParallel,
        trainer: Trainer,
        # method specific
        microbatch_outputs: List[Any],  # outputs from forward method in MegatronLossReduction across microbatches
        loss_mean: Any,  # output from reduce method in MegatronLossReduction
    ) -> None: ...

    def on_megatron_log_step_end(  # noqa: D102
        self,
        data: _DataFetcherWrapper,
        forward_only: bool,
        data_step: _ModuleStepFunction,
        forward_step: _ModuleStepFunction,
        loss_reduction: _ModuleStepFunction,
        seq_length: int,
        micro_batch_size: int,
        num_microbatches: int,
        wrap_forward_step: bool,
        pipeline: MegatronParallel,
        use_global_batch_sampler: bool,
        data_iterator: _DataFetcherWrapper,
        pl_module: MegatronParallel,
        trainer: Trainer,
        # method specific
        microbatch_outputs: List[Any],
        loss_mean: Any,  # output from reduce method in MegatronLossReduction
    ) -> None: ...

    def on_megatron_step_end(  # noqa: D102
        self,
        data: _DataFetcherWrapper,
        forward_only: bool,
        data_step: _ModuleStepFunction,
        forward_step: _ModuleStepFunction,
        loss_reduction: _ModuleStepFunction,
        seq_length: int,
        micro_batch_size: int,
        num_microbatches: int,
        wrap_forward_step: bool,
        pipeline: MegatronParallel,
        use_global_batch_sampler: bool,
        data_iterator: _DataFetcherWrapper,
        pl_module: MegatronParallel,
        trainer: Trainer,
        # method specific
        microbatch_outputs: List[Any],  # outputs from forward method in MegatronLossReduction
        loss_mean: Any,  # output from reduce method in MegatronLossReduction
    ) -> None: ...


class PerplexityLoggingCallback(pl.Callback, TypedMegatronCallback):
    """Megatron Callback to log perplexity in validation and optionally training.

    NeMo2.0 checks whether a callback is an instance of {LightningModule,LightningDataModule,Callback} but only megatron_hooks are useful.
    """

    def __init__(self, log_train: bool = False, log_val: bool = True):
        """Initialize PerplexityLoggingCallback.

        Args:
            log_train: whether to log train perplexity. Defaults to False.
            log_val: whether to log validation perplexity. Defaults to True.
        """
        super().__init__()
        self.log_train = log_train
        self.log_val = log_val

    def _pad_to_max_length(
        self, microbatch_outputs: List[Dict[str, Dict[str, torch.Tensor]]], key1: str, key2: str, pad_value: int = 0
    ) -> torch.Tensor:
        """Pad tensors to max length in microbatch_outputs."""
        max_sequence_length = max(output[key1][key2].size(1) for output in microbatch_outputs)

        tensors = []
        for microbatch_output in microbatch_outputs:
            tensor = microbatch_output[key1][key2]
            assert (
                tensor.dim() >= 2
            ), f"Tensor in microbatch_outputs must have at least 2 dimensions, but got {tensor.dim()} dimensions"
            tensors.append(
                torch.nn.functional.pad(  # padding reverse in order
                    tensor,
                    (0, 0) * (tensor.dim() - 2)
                    + (0, max_sequence_length - tensor.shape[1], 0, 0),  # [b s *] -> [* s b]
                    value=pad_value,
                )
            )

        return torch.cat(tensors, dim=0)  # concat on batch dim

    @override
    def on_megatron_reduce_microbatches_end(
        self,
        data: _DataFetcherWrapper,
        forward_only: bool,
        data_step: _ModuleStepFunction,
        forward_step: _ModuleStepFunction,
        loss_reduction: _ModuleStepFunction,
        seq_length: int,
        micro_batch_size: int,
        num_microbatches: int,
        wrap_forward_step: bool,
        pipeline: MegatronParallel,
        use_global_batch_sampler: bool,
        data_iterator: _DataFetcherWrapper,
        pl_module: MegatronParallel,
        trainer: Trainer,
        # method specific
        microbatch_outputs: List[Any],  # outputs from forward method in MegatronLossReduction
        loss_mean: Any,  # output from reduce method in MegatronLossReduction
    ) -> None:
        """Log after MegatronReductionLoss.reduce is called.

        Expected microbatch_outputs to be a list of dicts with the following keys:
            - batch: dict of tensors with the following keys:
                - labels: [b s]
                - loss_mask: [b s]; 1 means included 0 means ignored
            - forward_out: dict of tensors with the following keys:
                - token_logits: [b s vocab]
        """
        if trainer.training and not self.log_train:
            return

        assert num_microbatches > 0, "num_microbatches must be greater than 0"
        assert len(microbatch_outputs) == num_microbatches, "microbatch_outputs length does not match num_microbatches"
        labels = self._pad_to_max_length(microbatch_outputs, "batch", "labels", pad_value=-100)
        loss_mask = self._pad_to_max_length(microbatch_outputs, "batch", "loss_mask")
        token_logits = self._pad_to_max_length(microbatch_outputs, "forward_out", "token_logits")

        unreduced_token_loss = unreduced_token_loss_fn(token_logits, labels)  #  [b s]

        cp_size = parallel_state.get_context_parallel_world_size()
        if cp_size == 1:
            losses_for_microbatch = per_sequence_masked_token_loss(unreduced_token_loss, loss_mask)  # [b]
            ppl_for_microbatch = torch.exp(losses_for_microbatch).mean()
        else:
            raise NotImplementedError("Context parallel perplexity logging is not supported yet")

        if self.log_val and trainer.training is False:
            pp_size = parallel_state.get_pipeline_model_parallel_world_size()
            if pp_size > 1:
                # ranks that are not final pp stage have 0 for loss, and out will be mean-reduced over pp
                # groups (due to sync_dist), which divides val_loss by pp_size. so we multiply by pp_size to cancel out
                pl_module.log(
                    "val_ppl",
                    ppl_for_microbatch * pp_size,
                    prog_bar=True,
                    sync_dist=True,
                    sync_dist_group=parallel_state.get_pipeline_model_parallel_group(),
                    on_epoch=True,
                )
            else:
                pl_module.log("val_ppl", ppl_for_microbatch, prog_bar=True, on_epoch=True)
        elif self.log_train and trainer.training is True:
            if parallel_state.is_pipeline_last_stage():
                pl_module.log("train_ppl", ppl_for_microbatch, prog_bar=True, batch_size=1, sync_dist=False)
