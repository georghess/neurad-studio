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

"""Compatibility for the runtime type checker used by the test suite."""

import typeguard
from typing_extensions import evaluate_forward_ref


# Typeguard 2 supports jaxtyping's shape strings, unlike Typeguard 3/4, but calls
# ForwardRef._evaluate with the pre-Python-3.12 signature. Use the public backport
# API so tests retain their runtime checks on supported Python versions.
def _evaluate_forwardref(forward_ref, globalns, localns, _recursive_guard=frozenset()):
    return evaluate_forward_ref(forward_ref, globals=globalns, locals=localns)


typeguard.evaluate_forwardref = _evaluate_forwardref
