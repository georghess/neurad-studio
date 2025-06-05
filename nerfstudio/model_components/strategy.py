# Copyright 2024 the authors of NeuRAD and contributors.
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
from dataclasses import dataclass
from typing import Any, Dict, Union

import torch
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat.strategy.ops import remove, reset_opa


@dataclass
class ADDefaultStrategy(DefaultStrategy):
    """Default strategy for AD models."""

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        packed: bool = False,
        dynamic_actors: Any = None,
    ):
        """Callback function to be executed after the `loss.backward()` call."""
        if step >= self.refine_stop_iter:
            return
        update_state = True
        if info["width"] <= 0 or info["height"] <= 0:
            update_state = False

        if update_state:
            self._update_state(params, state, info, packed=packed)

        if (
            step > self.refine_start_iter
            and step % self.refine_every == 0
            and step % self.reset_every >= self.pause_refine_after_reset
        ):
            # grow GSs
            n_dupli, n_split = self._grow_gs(params, optimizers, state, step)
            if self.verbose:
                print(
                    f"Step {step}: {n_dupli} GSs duplicated, {n_split} GSs split. "
                    f"Now having {len(params['means'])} GSs."
                )

            # prune GSs
            n_prune = self._prune_gs(params, optimizers, state, step, dynamic_actors)
            if self.verbose:
                print(f"Step {step}: {n_prune} GSs pruned. " f"Now having {len(params['means'])} GSs.")

            # reset running stats
            state["grad2d"].zero_()
            state["count"].zero_()
            if self.refine_scale2d_stop_iter > 0:
                state["radii"].zero_()
            torch.cuda.empty_cache()

        if step % self.reset_every == 0:
            reset_opa(
                params=params,
                optimizers=optimizers,
                state=state,
                value=self.prune_opa * 2.0,
            )

    @torch.no_grad()
    def _prune_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        dynamic_actors: Any,
    ) -> int:
        is_prune = torch.sigmoid(params["opacities"].flatten()) < self.prune_opa
        if step > self.reset_every:
            is_too_big = torch.exp(params["scales"]).max(dim=-1).values > self.prune_scale3d * state["scene_scale"]
            # The official code also implements sreen-size pruning but
            # it's actually not being used due to a bug:
            # https://github.com/graphdeco-inria/gaussian-splatting/issues/123
            # We implement it here for completeness but set `refine_scale2d_stop_iter`
            # to 0 by default to disable it.
            if step < self.refine_scale2d_stop_iter:
                is_too_big |= state["radii"] > self.prune_scale2d

            is_prune = is_prune | is_too_big

        # Remove points outside of bbvox + check that we don't cull away an id completely
        outsidebox_count = 0
        for i in params["id"].unique():
            if i >= dynamic_actors.actor_sizes.shape[0]:
                continue  # skip static
            current_id_idx = torch.where(((params["id"] - i).abs() < 0.1).squeeze())[0]
            current_means = params["means"][current_id_idx]
            # check if means are outside of the actor bounding box
            current_actor_box = dynamic_actors.actor_bounds()[i.int()]
            outside_box_mask = ((current_means < -current_actor_box) | (current_means > current_actor_box)).any(dim=1)
            is_prune[current_id_idx] = is_prune[current_id_idx] | outside_box_mask

            if is_prune[current_id_idx].all():
                rand_culls = torch.ones_like(is_prune[current_id_idx])
                rand_indices = torch.randperm(rand_culls.shape[0])
                rand_culls[rand_indices[: (rand_culls.shape[0] // 2 + 1)]] = False
                is_prune[current_id_idx] = rand_culls
                outsidebox_count += rand_culls.sum().item()
            else:
                outsidebox_count += torch.sum(outside_box_mask).item()

        n_prune = is_prune.sum().item()
        if n_prune > 0:
            remove(params=params, optimizers=optimizers, state=state, mask=is_prune)

        return n_prune


class ADMCMCStrategy(MCMCStrategy):
    """MCMC strategy for AD models."""

    pass
