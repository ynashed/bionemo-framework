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
import importlib
import pkgutil

import pytest


MODULE = 'dgl'


@pytest.mark.needs_gpu
def test_module_exist():
    """
    test to check if dgl exist
    """
    eggs_loader = pkgutil.find_loader(MODULE)
    assert eggs_loader is not None


@pytest.mark.needs_gpu
def test_module_import():
    """
    test to check if dgl can be safely imported
    """
    importlib.import_module(MODULE)