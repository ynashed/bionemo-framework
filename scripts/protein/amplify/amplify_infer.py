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

import argparse
import os
from datetime import datetime
from pathlib import Path
from typing import Sequence

import torch
import torch.utils.data
from torch import Tensor
from torch.utils.data import Dataset

from nemo import lightning as nl

from bionemo.esm2.model.finetune.datamodule import ESM2FineTuneDataModule
from bionemo.amplify.api import AMPLIFYConfig
from bionemo.amplify.data import tokenizer
from bionemo.llm.data.types import BertSample
from bionemo.llm.model.biobert.lightning import biobert_lightning_module


__all__: Sequence[str] = ("infer_model",)


SUPPORTED_CONFIGS = {
    "AMPLIFYConfig": AMPLIFYConfig,
}


class InMemoryProteinDataset(Dataset):
    """An in-memory dataset that tokenize strings into BertSample instances."""

    def __init__(
        self,
        input_path: str | os.PathLike,
        tokenizer: tokenizer.BioNeMoAMPLIFYTokenizer = tokenizer.get_tokenizer(),
    ):
        """Initializes a dataset of protein sequences.

        This is an in-memory dataset that does not apply masking to the sequence. But keeps track of <mask> in the
        dataset sequences provided.

        Args:
            input_path (str | os.PathLike): Path to the input data file containing AA sequences.
            tokenizer (tokenizer.BioNeMoAMPLIFYTokenizer, optional): The tokenizer to use. Defaults to tokenizer.get_tokenizer().
            that __getitem__ is deterministic, but can be random across different runs. If None, a random seed is
            generated.
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"The input file {input_path} does not exist.")
        with open(input_path, 'r') as file:
            self.sequences = file.readlines()

        self._len = len(self.sequences)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        """The size of the dataset."""
        return self._len

    def __getitem__(self, index: int) -> BertSample:
        """Obtains the BertSample at the given index."""
        sequence = self.sequences[index]
        tokenized_sequence = self._tokenize(sequence)

        # Overall mask for a token being masked in some capacity - either mask token, random token, or left as-is
        loss_mask = ~torch.isin(tokenized_sequence, Tensor(self.tokenizer.all_special_ids))

        return {
            "text": tokenized_sequence,
            "types": torch.zeros_like(tokenized_sequence, dtype=torch.int64),
            "attention_mask": torch.ones_like(tokenized_sequence, dtype=torch.int64),
            "labels": tokenized_sequence,
            "loss_mask": loss_mask,
            "is_random": torch.zeros_like(tokenized_sequence, dtype=torch.int64),
        }

    def _tokenize(self, sequence: str) -> Tensor:
        """Tokenize a protein sequence.

        Args:
            sequence: The protein sequence.

        Returns:
            The tokenized sequence.
        """
        tensor = self.tokenizer.encode(sequence, add_special_tokens=True, return_tensors="pt")
        return tensor.flatten()  # type: ignore

def infer_model(
    data_path: Path,
    checkpoint_path: Path,
    results_path: Path,
) -> None:
    """Runs inference on a BioNeMo AMPLIFY model using PyTorch Lightning.

    Args:
        data_path (Path): Path to the input data.
        checkpoint_path (Path): Path to the model checkpoint.
        results_path (Path): Path to save the inference results.
    """
    # create the directory to save the inference results
    os.makedirs(results_path, exist_ok=True)

    dataset = InMemoryProteinDataset(input_path=data_path)
    datamodule = ESM2FineTuneDataModule(predict_dataset=dataset, 
                                         global_batch_size=len(dataset), 
                                         micro_batch_size=len(dataset))
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1, 
        pipeline_model_parallel_size=1, 
        ddp="megatron", 
        find_unused_parameters=True
    )

    trainer = nl.Trainer(
        accelerator="gpu",
        devices=1,
        strategy=strategy,
        num_nodes=1,
        plugins=nl.MegatronMixedPrecision(precision="bf16-mixed"),
    )

    config = AMPLIFYConfig(
        initial_ckpt_path=str(checkpoint_path),
        initial_ckpt_skip_keys_with_these_prefixes=[],  # load everything from the checkpoint.
    )

    amplify_tokenizer = tokenizer.get_tokenizer()
    module = biobert_lightning_module(config=config, tokenizer=amplify_tokenizer)

    logits = trainer.predict(module, datamodule=datamodule)  # return_predictions=False failing due to a lightning bug
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = results_path / f"logits_{timestamp}.pt"
    torch.save(logits, output_file)
    print("Inference completed")

def infer_amplify_entrypoint():
    """Entrypoint for running inference on a geneformer checkpoint and data."""
    # 1. get arguments
    parser = get_parser()
    args = parser.parse_args()
    # 2. Call infer with args
    infer_model(
        data_path=args.data_path,
        checkpoint_path=args.checkpoint_path,
        results_path=args.results_path,
    )


def get_parser():
    """Return the cli parser for this tool."""
    parser = argparse.ArgumentParser(description="Infer AMPLIFY.")
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        required=True,
        help="Path to the AMPLIFY pretrained checkpoint",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Path to the file containing sequences for inference",
    )
    parser.add_argument(
        "--results-path", 
        type=Path, 
        required=True, 
        help="Path to the results directory.")
    
    return parser


if __name__ == "__main__":
    infer_amplify_entrypoint()