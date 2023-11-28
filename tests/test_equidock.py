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


import numpy as np
import pytest
import torch

from bionemo.model.protein.equidock.loss_metrics.eval import rigid_transform_Kabsch_3D, rigid_transform_Kabsch_3D_torch
from bionemo.model.protein.equidock.loss_metrics.intersection_loss import compute_body_intersection_loss
from bionemo.model.protein.equidock.model.utils import compute_cross_attention


torch.use_deterministic_algorithms(False)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


@pytest.mark.needs_gpu
@pytest.mark.parametrize("data_type", [torch.float32, torch.float64])
def test_Kabsch_algorithm(data_type):
    for _ in range(10):
        # Random points
        A = np.random.randn(3, 10)
        B = np.random.randn(3, 10)

        # Get transformations from both functions
        R_numpy, t_numpy = rigid_transform_Kabsch_3D(A, B)
        R_torch, t_torch = rigid_transform_Kabsch_3D_torch(
            torch.from_numpy(A).to('cuda').to(data_type), torch.from_numpy(B).to('cuda').to(data_type)
        )

        # Convert the torch tensors to numpy arrays for easy comparison
        R_torch = R_torch.detach().cpu().numpy()
        t_torch = t_torch.detach().cpu().numpy()

        # Assert if matrices and vectors are close
        assert np.allclose(
            R_numpy, R_torch, atol=1e-4 if data_type == torch.float32 else 1e-6
        ), f"Rotation matrices differ: \n{R_numpy}\n{R_torch}"
        assert np.allclose(
            t_numpy, t_torch, atol=1e-4 if data_type == torch.float32 else 1e-6
        ), f"Translation vectors differ: \n{t_numpy}\n{t_torch}"


def test_body_intersection_loss():
    model_ligand_coors_deform = torch.zeros(1, 3)
    bound_receptor_repres_nodes_loc_array = 5.0 * torch.ones(1, 3)
    sigmas = torch.linspace(1.0, 10.0, 10)
    old_res = compute_body_intersection_loss(
        model_ligand_coors_deform, bound_receptor_repres_nodes_loc_array, sigmas[0], 1.0
    )

    for sigma in sigmas[1:]:
        new_res = compute_body_intersection_loss(
            model_ligand_coors_deform, bound_receptor_repres_nodes_loc_array, sigma, 1.0
        )
        if new_res < old_res:
            assert False, "Body intersection loss should be increasing with sigma increase!"
        old_res = new_res


def test_cross_attention():
    N = 21
    M = 11
    D = 16
    d = 4

    q = np.random.randn(N, D)
    k = np.random.randn(M, D)
    v = np.random.randn(M, d)
    mask = np.random.randint(0, 2, size=(N, M))

    a = mask * (q @ k.transpose([1, 0])) - 1000.0 * (1.0 - mask)

    a_x = np.exp(a) / np.exp(a).sum(axis=1, keepdims=True)
    expected = a_x @ v

    q = torch.from_numpy(q)
    k = torch.from_numpy(k)
    v = torch.from_numpy(v)
    mask = torch.from_numpy(mask)

    result = compute_cross_attention(q, k, v, mask, True).detach().cpu().numpy()

    assert np.allclose(expected, result, atol=1e-4), f"Cross message attention fails: \n{expected}\n{result}"
