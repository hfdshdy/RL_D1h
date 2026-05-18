# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from collections.abc import Sequence


@configclass
class DdtWheelPActionCfg(ActionTermCfg):
    """DDT P-mode wheel action.

    This reproduces ddt_ros2_control's wheel P-mode torque computation:

        tau_wheel = wheel_kp * action_scale * action - wheel_kd * wheel_vel

    With the default D1H rl_flat parameters, this becomes:

        tau_wheel = 10.0 * 0.5 * action - 0.6 * wheel_vel
                  = 5.0 * action - 0.6 * wheel_vel
    """

    class_type: type[ActionTerm] = MISSING

    joint_names: list[str] = MISSING
    kp: float = 10.0
    kd: float = 0.6
    action_scale: float = 0.5
    effort_limit: float = 20.0
    preserve_order: bool = True

    def __post_init__(self):
        if self.class_type is MISSING:
            self.class_type = DdtWheelPAction


class DdtWheelPAction(ActionTerm):
    """Apply wheel efforts that match DDT's P-mode wheel controller."""

    cfg: DdtWheelPActionCfg
    _asset: Articulation

    def __init__(self, cfg: DdtWheelPActionCfg, env):
        super().__init__(cfg, env)

        self._joint_ids, self._joint_names = self._asset.find_joints(
            self.cfg.joint_names,
            preserve_order=self.cfg.preserve_order,
        )

        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

    @property
    def action_dim(self) -> int:
        return len(self._joint_ids)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions

        wheel_vel = self._asset.data.joint_vel[:, self._joint_ids]
        torques = self.cfg.kp * self.cfg.action_scale * actions - self.cfg.kd * wheel_vel
        self._processed_actions[:] = torch.clamp(
            torques,
            min=-self.cfg.effort_limit,
            max=self.cfg.effort_limit,
        )

    def apply_actions(self):
        self._asset.set_joint_effort_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0


DdtWheelPActionCfg.class_type = DdtWheelPAction