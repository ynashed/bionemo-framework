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


from typing import TypedDict

import torch


class BertSample(TypedDict):
    """The type expected by NeMo/Megatron for a single dataset item.

    Attributes:
        text: The tokenized, masked input text.
        types: The token type ids, if applicable.
        attention_mask: A mask over all valid tokens, excluding padding.
        labels: The true values of the masked tokens at each position covered by loss_mask.
        loss_mask: The mask over the text indicating which tokens are masked and should be predicted.
        is_random: ??
    """

    text: torch.Tensor
    types: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    loss_mask: torch.Tensor
    is_random: torch.Tensor
