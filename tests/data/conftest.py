# Copyright 2026 the authors of NeuRAD and contributors.
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

"""Lazily load optional dataset fixtures without interrupting test collection."""

from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def mini_log(tmp_path_factory) -> Path:
    """Module-scoped synthetic py123d log directory."""
    pytest.importorskip("py123d", reason="py123d is an optional dependency for these tests")
    from py123d_fixtures import build_mini_py123d_log

    root = tmp_path_factory.mktemp("py123d_data")
    return build_mini_py123d_log(root)
