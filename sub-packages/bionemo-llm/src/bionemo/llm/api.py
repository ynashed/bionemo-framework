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


from abc import ABC, abstractmethod
from typing import Sequence

from megatron.core.transformer.module import MegatronModule
from nemo.lightning.megatron_parallel import MegatronLossReduction

from bionemo.core.model.config import Model, ModelOutput


__all__: Sequence[str] = (
    "BionemoMegatronModel",
    "Loss",
)

Loss = MegatronLossReduction
"""Loss functions that interact with Megatron need to either be or inherit from `MegatronLossReduction`.
"""


class BionemoMegatronModel(MegatronModule, ABC):
    """Models that use Megatron must be a MegatronModule type.

    The only major difference is the explicit `forward` pass method signature that makes this class compatible
    with bionemo-core's `Model` structural type.
    """

    @abstractmethod
    def forward(self, *args, **kwargs) -> ModelOutput:  # D102
        raise NotImplementedError()


# Typechecking: ensure that the bionemo megatron model abstraction is compliant with bionemo-core's Model
_: type[Model] = BionemoMegatronModel
