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

import itertools
import sys
from warnings import warn

import pytest
import torch
from torch.utils.data import SequentialSampler, default_collate

from bionemo.size_aware_batching.sampler import SizeAwareBatchSampler, size_aware_batching


@pytest.mark.parametrize(
    "collate_fn, max_total_size, warn_logger", itertools.product([None, default_collate], [0, 15, 31], [None, warn])
)
def test_sabs_iter(dataset, collate_fn, max_total_size, warn_logger):
    def sizeof(data: torch.Tensor):
        return ((data[0].item() + 1) % 3) * 10

    if warn_logger is not None and (max_total_size == 0 or max_total_size == 15):
        with pytest.warns(UserWarning):
            meta_batch_ids = list(
                size_aware_batching(dataset, sizeof, max_total_size, collate_fn=collate_fn, warn_logger=warn_logger)
            )
    else:
        meta_batch_ids = list(
            size_aware_batching(dataset, sizeof, max_total_size, collate_fn=collate_fn, warn_logger=warn_logger)
        )

    meta_batch_ids_expected = []
    ids_batch = []
    s_all = 0
    for data in dataset:
        s = sizeof(data)
        if s > max_total_size:
            continue
        if s + s_all > max_total_size:
            meta_batch_ids_expected.append(ids_batch)
            s_all = s
            ids_batch = [data]
            continue
        s_all += s
        ids_batch.append(data)
    if len(ids_batch) > 0:
        meta_batch_ids_expected.append(ids_batch)

    if collate_fn is not None:
        meta_batch_ids_expected = [collate_fn(batch) for batch in meta_batch_ids_expected]

    for i in range(len(meta_batch_ids)):
        torch.testing.assert_close(meta_batch_ids[i], meta_batch_ids_expected[i])


def test_SABS_init_valid_input(sampler, get_sizeof_dataset):
    sizeof = get_sizeof_dataset
    max_total_size = 60
    batch_sampler = SizeAwareBatchSampler(sampler, sizeof, max_total_size)
    assert batch_sampler._sampler == sampler
    assert batch_sampler._max_total_size == max_total_size

    for idx in sampler:
        if callable(sizeof):
            assert batch_sampler._sizeof(idx) == sizeof(idx)
        else:
            assert batch_sampler._sizeof(idx) == sizeof[idx]


def test_SABS_init_invalid_max_total_size(sampler):
    with pytest.raises(ValueError):
        SizeAwareBatchSampler(sampler, -1, {})

    with pytest.raises(ValueError):
        SizeAwareBatchSampler(sampler, 0, {})


def test_SABS_init_invalid_sampler_type():
    max_total_size = 60
    sampler = "not a sampler"
    with pytest.raises(TypeError):
        SizeAwareBatchSampler(sampler, max_total_size, {})


def test_SABS_init_invalid_sizeof_type(sampler):
    max_total_size = 60
    sizeof = " invalid type"
    with pytest.raises(TypeError):
        SizeAwareBatchSampler(sampler, sizeof, max_total_size)


def test_SABS_init_sizeof_seq_bounds_check(sampler):
    max_total_size = 60
    sizeof = [10] * (len(sampler) - 1)  # invalid length

    sys.gettrace = lambda: True

    with pytest.raises(ValueError):
        SizeAwareBatchSampler(sampler, sizeof, max_total_size)

    sys.gettrace = lambda: None


def test_SABS_init_max_size_exceeds_max_total_size(sampler):
    max_total_size = 100
    sizeof = {i: (1000 if i == 0 else 1) for i in sampler}

    sys.gettrace = lambda: True
    with pytest.warns(UserWarning):
        SizeAwareBatchSampler(sampler, sizeof, max_total_size)
    sys.gettrace = lambda: None


def test_SABS_init_min_size_exceeds_max_total_size(sampler):
    max_total_size = 60
    sizeof = {i: max_total_size + 1 for i in range(len(sampler))}  # invalid value

    sys.gettrace = lambda: True

    with pytest.raises(ValueError), pytest.warns(UserWarning):
        SizeAwareBatchSampler(sampler, sizeof, max_total_size)

    sys.gettrace = lambda: None


@pytest.mark.parametrize("max_total_size, warn_logger", itertools.product([0, 31, 60], [None, warn]))
def test_SABS_iter(sampler, get_sizeof_dataset, max_total_size, warn_logger):
    sizeof = get_sizeof_dataset

    if max_total_size == 0 and not callable(sizeof):
        sys.gettrace = lambda: True
        if warn_logger is not None:
            with pytest.raises(ValueError, match=r"exceeds max_total_size"), pytest.warns(UserWarning):
                size_aware_sampler = SizeAwareBatchSampler(sampler, sizeof, max_total_size, warn_logger=warn_logger)
        else:
            with pytest.raises(ValueError):
                size_aware_sampler = SizeAwareBatchSampler(sampler, sizeof, max_total_size, warn_logger=warn_logger)
        sys.gettrace = lambda: None
    else:
        # construction should always succeed
        size_aware_sampler = SizeAwareBatchSampler(sampler, sizeof, max_total_size, warn_logger=warn_logger)

        if max_total_size == 0 and warn_logger is not None:
            with pytest.warns(UserWarning):
                meta_batch_ids = list(size_aware_sampler)
        else:
            meta_batch_ids = list(size_aware_sampler)

        def fn_sizeof(i: int):
            if callable(sizeof):
                return sizeof(i)
            else:
                return sizeof[i]

        # Check that the batches are correctly sized
        for ids_batch in meta_batch_ids:
            size_batch = sum(fn_sizeof(idx) for idx in ids_batch)
            assert size_batch <= max_total_size

        meta_batch_ids_expected = []
        ids_batch = []
        s_all = 0
        for idx in sampler:
            s = fn_sizeof(idx)
            if s > max_total_size:
                continue
            if s + s_all > max_total_size:
                meta_batch_ids_expected.append(ids_batch)
                s_all = s
                ids_batch = [idx]
                continue
            s_all += s
            ids_batch.append(idx)
        if len(ids_batch) > 0:
            meta_batch_ids_expected.append(ids_batch)

        assert meta_batch_ids == meta_batch_ids_expected

        # the 2nd pass should return the same result
        if max_total_size == 0 and warn_logger is not None:
            with pytest.warns(UserWarning):
                meta_batch_ids_2nd_pass = list(size_aware_sampler)
        else:
            meta_batch_ids_2nd_pass = list(size_aware_sampler)
        assert meta_batch_ids == meta_batch_ids_2nd_pass


def test_SABS_iter_no_samples():
    # Test iterating over a batch of indices with no samples
    sampler = SequentialSampler([])
    size_aware_sampler = SizeAwareBatchSampler(sampler, {}, 100)

    batched_indices = list(size_aware_sampler)

    assert not batched_indices


def test_SABS_iter_empty_sizeof(sampler):
    size_aware_sampler = SizeAwareBatchSampler(sampler, {}, 1)

    with pytest.raises(RuntimeError, match="sizeof raises error at data"):
        list(size_aware_sampler)
