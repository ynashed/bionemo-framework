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

import os
import pathlib
import tarfile
from copy import deepcopy
from typing import List, Tuple

import pytest
import torch
from nemo import lightning as nl
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from bionemo.contrib.data.resamplers import PRNGDatasetShuffler
from bionemo.contrib.data.singlecell.dataset import SingleCellDataset
from bionemo.contrib.data.singlecell.preprocess import GeneformerPreprocess
from bionemo.contrib.model.biobert.lightning import BioBertLightningModule
from bionemo.contrib.model.biobert.model import BioBertConfig, BiobertSpecOption
from bionemo.contrib.testing import megatron_parallel_state_utils
from bionemo.contrib.testing.utils import assert_matrix_correlation_above_value, assert_matrix_mape_below_value
from bionemo.contrib.utils.batching_utils import pad_token_ids
from bionemo.contrib.utils.dtypes import get_autocast_dtype
from bionemo.contrib.utils.random_utils import random_numpy_context
from bionemo.contrib.utils.weight_utils import (
    nemo1_to_nemo2_biobert_key_mapping,
)


# TODO(@jstjohn) use fixtures for pulling down data and checkpoints
# python scripts/download_artifacts.py --models all --model_dir ./models --data all --data_dir ./ --verbose --source pbss
test_script_dir = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))
bionemo2_root = test_script_dir.parent.parent.parent.parent.parent.parent.parent
nemo1_checkpoint_path = bionemo2_root / "models/singlecell/geneformer/geneformer-qa.nemo"
nemo1_release_checkpoint_path = bionemo2_root / "models/singlecell/geneformer/geneformer-10M-240530.nemo"
nemo_1_per_layer_outputs_path = bionemo2_root / "test_data/nemo1-test-outputs-geneformer-qa.pt"
nemo_1_expected_values_path = bionemo2_root / "test_data/nemo1_geneformer_qa_test_golden_values.pt"
data_path = bionemo2_root / "test_data/cellxgene_2023-12-15_small/processed_data"


CELLS_FOR_TEST = [
    [
        "ENSG00000288623",
        "ENSG00000288658",
        "ENSG00000288681",
        "ENSG00000288698",
        "ENSGR0000002586",
        "ENSGR0000124333",
        "ENSGR0000124334",
        "ENSGR0000167393",
        "ENSGR0000168939",
        "ENSGR0000169084",
    ],
    [
        "ENSG00000259900",
        "ENSG00000259916",
        "ENSG00000259956",
        "ENSG00000259958",
        "ENSG00000259991",
        "ENSG00000260001",
        "ENSG00000260007",
        "ENSG00000260027",
        "ENSG00000260040",
        "ENSG00000260045",
        "ENSG00000260092",
        "ENSG00000260099",
        "ENSG00000260119",
    ],
    [
        "ENSG00000269743",
        "ENSG00000269746",
        "ENSG00000269748",
        "ENSG00000269753",
        "ENSG00000269754",
        "ENSG00000269755",
        "ENSG00000269759",
        "ENSG00000269766",
        "ENSG00000269773",
        "ENSG00000269781",
        "ENSG00000269782",
        "ENSG00000269783",
        "ENSG00000269790",
        "ENSG00000269791",
        "ENSG00000269795",
    ],
]

MODEL_PRECISION: str = "bf16-mixed"


@pytest.fixture()
def cells() -> List[List[str]]:
    return deepcopy(CELLS_FOR_TEST)


@pytest.fixture
def geneformer_config():
    autocast_dtype = get_autocast_dtype(MODEL_PRECISION)
    return BioBertConfig(
        num_layers=6,
        hidden_size=256,
        ffn_hidden_size=512,
        num_attention_heads=4,
        seq_length=2048,
        fp32_residual_connection=False,  # TODO(@jstjohn) check this
        hidden_dropout=0.02,
        init_method_std=0.02,
        kv_channels=None,
        apply_query_key_layer_scaling=True,
        make_vocab_size_divisible_by=128,
        masked_softmax_fusion=True,  # TODO(@jstjohn) check this
        fp16_lm_cross_entropy=False,
        params_dtype=torch.float32,
        pipeline_dtype=torch.float32,
        autocast_dtype=autocast_dtype,  # setting this speeds things up a lot
        gradient_accumulation_fusion=False,  # THIS BREAKS STUFF, leave False
        layernorm_zero_centered_gamma=False,  # TODO(@jstjohn) check this
        layernorm_epsilon=1.0e-12,
        activation_func=F.gelu,  # TODO(@jstjohn) check this
        qk_layernorm=True,  # TODO(@jstjohn) check this
        apply_residual_connection_post_layernorm=True,  # False is new default, True was BERT pub.
        bias_activation_fusion=True,  # TODO(@jstjohn) check this
        bias_dropout_fusion=True,  # TODO(@jstjohn) check this
        get_attention_mask_from_fusion=False,
        attention_dropout=0.1,
        share_embeddings_and_output_weights=True,
        enable_autocast=False,  # This has to be set to True if we use the mixed precision plugin
        biobert_spec_option=BiobertSpecOption.bert_layer_local_spec,
        nemo1_ckpt_path=nemo1_checkpoint_path,
        return_only_hidden_states=True,  # This is what we did in nemo1 for inference
    )


def test_bionemo2_rootdir():
    assert (bionemo2_root / "sub-packages").exists(), "Could not find bionemo2 root directory."
    assert (bionemo2_root / "sub-packages").is_dir(), "sub-packages is supposed to be a directory."


def test_nemo1_nemo2_weight_shapes_match(geneformer_config, seed: int = 42):
    data_error_str = "Please download test data with:\n`python scripts/download_artifacts.py --models all --model_dir ./models --data all --data_dir ./ --verbose --source pbss`"
    data_dir = pathlib.Path(data_path)
    train_data_path = data_dir / "train"
    if not nemo1_checkpoint_path.exists():
        raise FileNotFoundError(f"Could not find checkpoint at {nemo1_checkpoint_path}. {data_error_str}")
    if not train_data_path.exists():
        raise FileNotFoundError(f"Could not find train data at {train_data_path}. {data_error_str}")

    with tarfile.open(
        nemo1_checkpoint_path, "r"
    ) as old_ckpt, torch.no_grad(), megatron_parallel_state_utils.distributed_model_parallel_state(seed):
        ckpt_file = old_ckpt.extractfile("./model_weights.ckpt")
        old_weights = torch.load(ckpt_file)
        preprocessor = GeneformerPreprocess(
            download_directory=train_data_path,
            medians_file_path=train_data_path / "medians.json",
            tokenizer_vocab_path=train_data_path / "geneformer.vocab",
        )
        match preprocessor.preprocess():
            case {"tokenizer": tokenizer, "median_dict": _}:
                pass
            case _:
                assert False
        new_model = geneformer_config.configure_model(tokenizer)
        new_state_dict = new_model.state_dict_for_save_checkpoint()
        # Set the new_model_prefix to "" since we are looking at the base megatron model and not the lightning module which stores a copy of
        #  this model into self.module
        old_keys = {nemo1_to_nemo2_biobert_key_mapping(k, new_model_prefix="") for k in old_weights}
        assert len(old_keys) == len(old_weights), "Mapping unexpectedly discarded some keys."
        new_keys = set(new_state_dict)
        for k, v in old_weights.items():
            # Make sure the shapes of the weights match.
            assert new_state_dict[nemo1_to_nemo2_biobert_key_mapping(k, new_model_prefix="")].shape == v.shape
        extra_keys = new_keys - old_keys
        extra_non_null_keys = {k for k in extra_keys if new_state_dict[k] is not None}
        assert not extra_non_null_keys, "There are new keys that have state that is missing from the old checkpoint."
        missing_old_keys = old_keys - new_keys
        assert not missing_old_keys, "There are keys in the old checkpoint that are missing from the new model."


def _apply_tokenizer(tokenizer, sequences: List[List[str]], device) -> List[torch.Tensor]:
    # parent pulls the tokenizer from the loaded model.
    try:
        token_ids = [
            torch.tensor(
                [tokenizer.class_id] + [tokenizer.token_to_id(gene_symbol) for gene_symbol in gene_symbols],
                device=device,
                dtype=torch.long,
            )
            for gene_symbols in sequences
        ]
    except TypeError as e:
        invalid_tokens = {gene_symbol for gene_symbols in sequences for gene_symbol in gene_symbols} - set(
            tokenizer.vocab.keys()
        )
        raise ValueError(
            f"Unknown token in gene symbols. Please filter genes for those present in self.tokenizer:\n{invalid_tokens}"
        ) from e
    return token_ids


def _batched_tokenizer(
    tokenizer, sequences: List[List[str]], device, seq_length: int = 2048, dynamic_padding: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenize sequences.
    Returns:
        token_ids (torch.Tensor, long): token ids
        mask (torch.Tensor, long, float): boolean mask for padded sections
    """
    token_ids = _apply_tokenizer(tokenizer=tokenizer, sequences=sequences, device=device)

    # Validate input sequences length
    if any(len(t) > seq_length for t in token_ids):
        raise ValueError(f"One or more sequence exceeds max length({seq_length}).")

    # Set fixed padding when dynamic padding is disabled
    if not dynamic_padding:
        padding_length = seq_length
    else:
        padding_length = None
    # Pad token ids (1/True = Active, 0/False = Inactive)
    token_ids, mask = pad_token_ids(
        token_ids,
        padding_value=tokenizer.pad_id,
        padding_len=padding_length,
        device=device,
    )

    return token_ids, mask


class _DummyDataSet(torch.utils.data.Dataset):
    def __init__(self, cells: List[List[str]], tokenizer):
        input_ids, mask = _batched_tokenizer(tokenizer, cells, device=torch.device("cuda"))
        self.input_ids = input_ids
        self.mask = mask
        assert len(self.input_ids) == len(self.mask)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"text": self.input_ids[idx], "attention_mask": self.mask[idx]}


def test_geneformer_nemo1_v_nemo2_inference_golden_values(
    geneformer_config: BioBertConfig, cells: List[List[str]], seed: int = 42
):
    """NOTE: this test is against old nemo1 inference golden values. It may be deprecated in the future as we move away from nemo1.
      This test documents _how_ different the two models are at the moment.
    Original model summary:
    BertModel(
    (language_model): TransformerLanguageModel(
        (embedding): Embedding(
            (word_embeddings): VocabParallelEmbedding()
            (position_embeddings): Embedding(2048, 256)
            (embedding_dropout): Dropout(p=0.02, inplace=False)
        )
        (encoder): ParallelTransformer(
        (layers): ModuleList(
            (0-5): 6 x ParallelTransformerLayer(
            (input_layernorm): MixedFusedLayerNorm(torch.Size([256]), eps=1e-12, elementwise_affine=True)
            (self_attention): ParallelAttention(
                (query_key_value): ColumnParallelLinear()
                (core_attention): CoreAttention(
                (scale_mask_softmax): MatchedScaleMaskSoftmax()
                (attention_dropout): Dropout(p=0.1, inplace=False)
                )
                (dense): RowParallelLinear()
            )
            (post_attention_layernorm): MixedFusedLayerNorm(torch.Size([256]), eps=1e-12, elementwise_affine=True)
            (mlp): ParallelMLP(
                (dense_h_to_4h): ColumnParallelLinear()
                (dense_4h_to_h): RowParallelLinear()
            )
            )
        )
        (final_layernorm): MixedFusedLayerNorm(torch.Size([256]), eps=1e-12, elementwise_affine=True)
        )
    )
    (lm_head): BertLMHead(
        (dense): Linear(in_features=256, out_features=256, bias=True)
        (layernorm): MixedFusedLayerNorm(torch.Size([256]), eps=1e-12, elementwise_affine=True)
    )
    )

    New model summary:
    MegatronBioBertModel(
    (embedding): LanguageModelEmbedding(
        (word_embeddings): VocabParallelEmbedding()
        (position_embeddings): Embedding(2048, 256)
        (embedding_dropout): Dropout(p=0.02, inplace=False)
    )
    (encoder): TransformerBlock(
        (layers): ModuleList(
        (0-5): 6 x TransformerLayer(
            (input_layernorm): FusedLayerNorm()
            (self_attention): SelfAttention(
            (core_attention): DotProductAttention(
                (scale_mask_softmax): FusedScaleMaskSoftmax()
                (attention_dropout): Dropout(p=0.1, inplace=False)
            )
            (linear_proj): RowParallelLinear()
            (linear_qkv): ColumnParallelLinear()
            (q_layernorm): IdentityOp()
            (k_layernorm): IdentityOp()
            )
            (pre_cross_attn_layernorm): IdentityOp()
            (cross_attention): IdentityOp()
            (cross_attn_bda): IdentityFuncOp()
            (pre_mlp_layernorm): FusedLayerNorm()
            (mlp): MLP(
            (linear_fc1): ColumnParallelLinear()
            (linear_fc2): RowParallelLinear()
            )
        )
        )
        (final_layernorm): LayerNorm()
    )
    (lm_head): BertLMHead(
        (dense): Linear(in_features=256, out_features=256, bias=True)
        (layer_norm): FusedLayerNorm()
    )
    (output_layer): ColumnParallelLinear()
    )


    """

    assert nemo_1_expected_values_path.exists(), f"Could not find expected values at {nemo_1_expected_values_path}."

    data_error_str = "Please download test data with:\n`python scripts/download_artifacts.py --models all --model_dir ./models --data all --data_dir ./ --verbose --source pbss`"
    data_dir = pathlib.Path(data_path)
    train_data_path = data_dir / "train"
    if not nemo1_checkpoint_path.exists():
        raise FileNotFoundError(f"Could not find checkpoint at {nemo1_checkpoint_path}. {data_error_str}")
    if not train_data_path.exists():
        raise FileNotFoundError(f"Could not find train data at {train_data_path}. {data_error_str}")

    preprocessor = GeneformerPreprocess(
        download_directory=train_data_path,
        medians_file_path=train_data_path / "medians.json",
        tokenizer_vocab_path=train_data_path / "geneformer.vocab",
    )
    match preprocessor.preprocess():
        case {"tokenizer": tokenizer, "median_dict": _}:
            pass
        case _:
            assert False

    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        ddp="megatron",
        find_unused_parameters=True,
        enable_nemo_ckpt_io=False,
        data_sampler=nl.MegatronDataSampler(
            micro_batch_size=3,
            global_batch_size=3,
            seq_len=16,
        ),
    )
    trainer = nl.Trainer(
        devices=1,
        accelerator="gpu",
        strategy=strategy,
        num_nodes=1,
        plugins=nl.MegatronMixedPrecision(precision=MODEL_PRECISION, amp_O2=False),
    )
    module = BioBertLightningModule(config=geneformer_config, tokenizer=tokenizer)

    dataloader = torch.utils.data.DataLoader(_DummyDataSet(cells, tokenizer), batch_size=3, num_workers=0)
    with megatron_parallel_state_utils.distributed_model_parallel_state(seed):
        result = torch.cat(trainer.predict(module, dataloaders=dataloader), dim=1).transpose(1, 0).contiguous()
    assert len(result) == 3
    expected_vals = {k: v.to(result.device) for k, v in torch.load(nemo_1_expected_values_path).items()}
    assert_matrix_mape_below_value(
        result,
        expected_vals["expected_hidden_state"],
        mask=expected_vals["expected_pad_masks"],
        eps=0.1,
        max_mape=2.04,  # 2.04% average difference in final values with a magnitude over 0.1
    )
    assert_matrix_correlation_above_value(
        result,
        expected_vals["expected_hidden_state"],
        mask=expected_vals["expected_pad_masks"],
        min_correlation=0.9999,
    )


def test_geneformer_inference_nemo1_v_nemo2_golden_values_by_layer(
    geneformer_config: BioBertConfig, cells: List[List[str]], seed: int = 42
):
    """NOTE: this test is against old nemo1 inference golden values. It may be deprecated in the future as we move away from nemo1.
    This test documents _how_ different the two models are at the moment at each layer, and highlights which layers are the most
    different. This test is useful for debugging and understanding the differences between the two models.
    """
    assert (
        nemo_1_per_layer_outputs_path.exists()
    ), f"Could not find per-layer expected values at {nemo_1_per_layer_outputs_path}."
    data_error_str = "Please download test data with:\n`python scripts/download_artifacts.py --models all --model_dir ./models --data all --data_dir ./ --verbose --source pbss`"
    data_dir = pathlib.Path(data_path)
    train_data_path = data_dir / "train"
    if not nemo1_checkpoint_path.exists():
        raise FileNotFoundError(f"Could not find checkpoint at {nemo1_checkpoint_path}. {data_error_str}")
    if not train_data_path.exists():
        raise FileNotFoundError(f"Could not find train data at {train_data_path}. {data_error_str}")

    with tarfile.open(
        nemo1_checkpoint_path, "r"
    ) as old_ckpt, torch.inference_mode(), megatron_parallel_state_utils.distributed_model_parallel_state(seed):
        ckpt_file = old_ckpt.extractfile("./model_weights.ckpt")
        old_weights = torch.load(ckpt_file)
        new_state_dict_from_old = {}
        for k, v in old_weights.items():
            new_key = nemo1_to_nemo2_biobert_key_mapping(k, new_model_prefix="")
            new_v = v
            new_state_dict_from_old[new_key] = new_v
        preprocessor = GeneformerPreprocess(
            download_directory=train_data_path,
            medians_file_path=train_data_path / "medians.json",
            tokenizer_vocab_path=train_data_path / "geneformer.vocab",
        )
        match preprocessor.preprocess():
            case {"tokenizer": tokenizer, "median_dict": _}:
                pass
            case _:
                assert False
        geneformer_config
        new_model = geneformer_config.configure_model(tokenizer).eval().cuda()
        new_model.load_state_dict(new_state_dict_from_old)
        for k, v in new_model.state_dict().items():
            # Make sure the weights were properly loaded
            if v is not None:
                torch.testing.assert_close(new_state_dict_from_old[k], v, check_dtype=False, check_device=False)
            else:
                assert k.endswith("_extra_state")

        input_ids, mask = _batched_tokenizer(tokenizer, cells, device=torch.device("cuda"))

        # with torch.autocast(device_type="cuda", dtype=get_autocast_dtype("bf16-mixed")):
        # new_model = new_model.bfloat16()  # if we move to the lightning way of calling forward we can drop this
        new_model.post_process = False  # so we get hidden states rather than logits
        new_model.encoder.post_process = True
        new_model.encoder.post_layer_norm = True
        new_outputs = {}
        from functools import partial

        def register_hooks(model, hook_fn):
            for name, module in model.named_modules():
                module.register_forward_hook(partial(hook_fn, name))

        def hook_fn(name, module, input, output):
            new_outputs[name] = (str(type(module)), input, output)

        register_hooks(new_model, hook_fn)
        # Fill up the new_outputs
        _ = new_model(input_ids, mask)
        ori_outputs = torch.load(nemo_1_per_layer_outputs_path)

        # Test settings for MAPE https://en.wikipedia.org/wiki/Mean_absolute_percentage_error thresholds
        softmax_mape_threshold = 9.8
        mape_tolerances = {
            "encoder.layers.0.self_attention.core_attention.scale_mask_softmax": softmax_mape_threshold,
            "encoder.layers.0.self_attention.core_attention.attention_dropout": softmax_mape_threshold,
            "encoder.layers.1.self_attention.core_attention.scale_mask_softmax": softmax_mape_threshold,
            "encoder.layers.1.self_attention.core_attention.attention_dropout": softmax_mape_threshold,
            "encoder.layers.2.self_attention.core_attention.scale_mask_softmax": softmax_mape_threshold,
            "encoder.layers.2.self_attention.core_attention.attention_dropout": softmax_mape_threshold,
            "encoder.layers.3.self_attention.core_attention.scale_mask_softmax": softmax_mape_threshold,
            "encoder.layers.3.self_attention.core_attention.attention_dropout": softmax_mape_threshold,
            "encoder.layers.4.self_attention.core_attention.scale_mask_softmax": softmax_mape_threshold,
            "encoder.layers.4.self_attention.core_attention.attention_dropout": softmax_mape_threshold,
            "encoder.layers.5.self_attention.core_attention.scale_mask_softmax": softmax_mape_threshold,
            "encoder.layers.5.self_attention.core_attention.attention_dropout": softmax_mape_threshold,
            "encoder.layers.4.pre_mlp_layernorm": 3.6,
            "encoder.layers.5.input_layernorm": 3.6,
            "encoder.layers.5.pre_mlp_layernorm": 4.1,
        }
        default_mape_tolerance = 3.3  # 3.3% difference in larger magnitude values with values over a magnitude of 0.1

        # Test settings for correlation https://en.wikipedia.org/wiki/Pearson_correlation_coefficient thresholds
        correlation_tolerances = {
            "encoder.layers.0.self_attention.core_attention.scale_mask_softmax": 0.985,
            "encoder.layers.0.self_attention.core_attention.attention_dropout": 0.985,
            "encoder.layers.1.self_attention.core_attention.scale_mask_softmax": 0.975,
            "encoder.layers.1.self_attention.core_attention.attention_dropout": 0.975,
            "encoder.layers.2.self_attention.core_attention.scale_mask_softmax": 0.975,
            "encoder.layers.2.self_attention.core_attention.attention_dropout": 0.975,
            "encoder.layers.3.self_attention.core_attention.scale_mask_softmax": 0.975,
            "encoder.layers.3.self_attention.core_attention.attention_dropout": 0.975,
            "encoder.layers.4.self_attention.core_attention.scale_mask_softmax": 0.96,
            "encoder.layers.4.self_attention.core_attention.attention_dropout": 0.96,
            "encoder.layers.5.self_attention.core_attention.scale_mask_softmax": 0.925,
            "encoder.layers.5.self_attention.core_attention.attention_dropout": 0.925,
        }
        default_correlation_tolerance = 0.9998  # 0.9999 correlation for final layer

        mask_t = mask.transpose(1, 0).contiguous()
        mask = mask[..., None]
        mask_t = mask_t[..., None]
        for module_name, (ori_cls_name, _, ori_output) in ori_outputs.items():
            new_module_name = nemo1_to_nemo2_biobert_key_mapping(module_name, new_model_prefix="")
            if new_module_name == "language_model":
                new_module_name = "encoder"
            if new_module_name == "model":
                new_module_name = ""
            new_cls_name, _, new_output = new_outputs[new_module_name]
            if new_module_name == "" and module_name == "":
                new_output = new_output.transpose(0, 1).contiguous()
            if isinstance(ori_output, (tuple, list)) or isinstance(new_output, (tuple, list)):
                if isinstance(ori_output, (tuple, list)):
                    ori_output = [o for o in ori_output if o is not None]
                else:
                    ori_output = [ori_output]
                if isinstance(new_output, (tuple, list)):
                    new_output = [o for o in new_output if o is not None]
                else:
                    new_output = [new_output]
                assert type(ori_output) == type(new_output)
                assert len(ori_output) == len(new_output)
                for ori, new in zip(ori_output, new_output):
                    if ori is None and new is None:
                        continue
                    if ori is None or new is None:
                        assert False, f"One of the outputs is None, but the other is not. {ori}, {new}"
                    assert ori.shape == new.shape
                    if ori.shape[0:2] == (16, 3):
                        _mask = mask_t
                    elif ori.shape[0:2] == (3, 16):
                        _mask = mask
                    else:
                        _mask = None
                    assert_matrix_mape_below_value(
                        new,
                        ori,
                        mask=_mask,
                        max_mape=mape_tolerances.get(new_module_name, default_mape_tolerance),
                        eps=1e-1,
                        msg=f"Module: {new_module_name}",
                    )
                    assert_matrix_correlation_above_value(
                        new,
                        ori,
                        mask=_mask,
                        min_correlation=correlation_tolerances.get(new_module_name, default_correlation_tolerance),
                        msg=f"Module: {new_module_name}",
                    )
            else:
                if new_output.shape[0:2] == (16, 3):
                    _mask = mask_t
                elif new_output.shape[0:2] == (3, 16):
                    _mask = mask
                else:
                    _mask = None
                assert_matrix_mape_below_value(
                    new_output,
                    ori_output,
                    mask=_mask,
                    eps=1e-1,
                    max_mape=mape_tolerances.get(new_module_name, default_mape_tolerance),
                    msg=f"Module: {new_module_name}",
                )
                assert_matrix_correlation_above_value(
                    new_output,
                    ori_output,
                    mask=_mask,
                    min_correlation=correlation_tolerances.get(new_module_name, default_correlation_tolerance),
                    msg=f"Module: {new_module_name}",
                )


@pytest.mark.parametrize("break_model", [True, False])
def test_inference_loss_10m_released_checkpoint(geneformer_config: BioBertConfig, break_model: bool, seed: int = 42):
    data_dir = pathlib.Path(data_path)
    train_data_path = data_dir / "train"
    test_data_path = data_dir / "test"
    with torch.inference_mode(), megatron_parallel_state_utils.distributed_model_parallel_state(
        seed
    ), random_numpy_context(seed):
        preprocessor = GeneformerPreprocess(
            download_directory=train_data_path,
            medians_file_path=train_data_path / "medians.json",
            tokenizer_vocab_path=train_data_path / "geneformer.vocab",
        )
        match preprocessor.preprocess():
            case {"tokenizer": tokenizer, "median_dict": median_dict}:
                pass
            case _:
                assert False
        geneformer_config_logit = deepcopy(geneformer_config)
        geneformer_config_logit.return_only_hidden_states = False  # return logits
        geneformer_config_logit.nemo1_ckpt_path = nemo1_release_checkpoint_path  # release checkpoint is important
        if break_model:
            # introduce a breaking change with a future xfail as a negative control for our test
            geneformer_config_logit.activation_func = torch.nn.functional.relu  # the model should be gelu
            geneformer_config_logit.bias_activation_fusion = False  # this needs to be off for ReLu support
        new_model = geneformer_config_logit.configure_model(tokenizer).eval().cuda()
        # NOTE: a small change to randomization in the single-cell dataset could throw our test below off by a small amount
        #  maybe 0.02 or so, if the value is above that range then disable the 200 batch limit and check the global number
        #  going back to `n += 1` and `loss += F.cross_entropy(logits[loss_mask], target[loss_mask], reduction="mean")`
        #  for consistency with the old results. Then if those look good, redefine the target with our seeds and the
        #  updated dataset.
        ds = SingleCellDataset(
            test_data_path,
            tokenizer=tokenizer,
            median_dict=median_dict,
            max_len=2048,
            mask_prob=0.15,
            mask_token_prob=0.8,
            random_token_prob=0.1,  # TODO: once this is fixed, change to 0.02 to match the prior numbers.
            prepend_cls_token=True,
        )
        dss = PRNGDatasetShuffler(
            ds,
            seed=seed,
        )
        dl = DataLoader(
            dataset=dss,  # pre-shuffled with our method
            batch_size=8,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        loss = 0
        n = 0
        limit_batches = 200
        for i, batch in tqdm(enumerate(dl), total=len(dl)):
            result = new_model(
                input_ids=batch["text"].cuda(),
                attention_mask=batch["attention_mask"].cuda(),
            )
            loss_mask = batch["loss_mask"].cuda()
            logits = result["token_logits"]
            target = batch["labels"].cuda()

            loss += F.cross_entropy(logits[loss_mask], target[loss_mask], reduction="sum")
            n += loss_mask.sum()

            if limit_batches is not None and i + 1 >= limit_batches:
                break

        mean_loss: float = (loss / n).cpu().numpy().item()
        # NOTE: the values in the table were from the average of averages of 8 sized batches
        # Experiment 1) loaded the 10M model and did the mean of mean loss with 8 sized batches
        #  this gives: 2.3558831214904785 vs 2.357126723703872, so we actually do better!
        # For NVIDIA employees see work here:
        #   https://docs.google.com/document/d/1CofamqHbQlp5U8SjmW7NR7PbTbF72Lj9L9xz1W5t3ZI/edit
        # Experiment 2)
        #  With a proper loss (sum/n) and limiting to 200 _random_ batches of size 8 for speed
        #  we get a similar range number of 2.368649959564209.
        #  a small change that has lower impact than the change between models is probably acceptable.
        #  the target is defined as described above for the 10M checkpoint based on our first pass
        #  of the megatron implementation. Since we manually passed experiment 1 this experiment
        #  will define our initial "golden value" test target.
        target: float = 2.368649959564209
        # test that we are within 0.01 or better
        test_pass = mean_loss < target or mean_loss == pytest.approx(target, abs=1e-2, rel=None)
        if break_model:
            assert not test_pass
        else:
            assert test_pass
