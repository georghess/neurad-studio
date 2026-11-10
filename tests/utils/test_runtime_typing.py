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

"""Ensure forward-reference checks remain active on supported Python releases."""

import pytest
from typeguard import typechecked


@typechecked
def _identity(value: "float") -> "float":
    return value


def test_forward_reference_type_checking():
    assert _identity(1.5) == 1.5
    assert _identity(1) == 1
    with pytest.raises(TypeError):
        _identity("invalid")
