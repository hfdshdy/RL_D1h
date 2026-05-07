from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp import UniformVelocityCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformVelocityWithZCommand(UniformVelocityCommand):
    """Velocity command with an additional discrete height target."""

    def __init__(self, cfg: UniformVelocityWithZCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.vel_command_b = torch.zeros(self.num_envs, 4, device=self.device)
        self.time_elapsed = torch.zeros(self.num_envs, device=self.device)
        self.track_z_flag = cfg.ranges.pos_z[0] != 0.0 or cfg.ranges.pos_z[1] != 0.0
        self.standing_choice = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    def _update_metrics(self):
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=-1) / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2]) / max_command_step
        )

    def _resample_command(self, env_ids: Sequence[int]):
        r = torch.empty(len(env_ids), device=self.device)
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)

        if self.track_z_flag:
            self.vel_command_b[env_ids, 3] = self._sample_height_categories(env_ids, 5)
        else:
            self.vel_command_b[env_ids, 3] = 0.0

        if self.cfg.heading_command:
            self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
            self.is_heading_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_heading_envs
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        if len(standing_env_ids) > 0:
            if self.track_z_flag:
                self.standing_choice[standing_env_ids] = self._sample_height_categories(standing_env_ids, 2)
            else:
                self.standing_choice[standing_env_ids] = 0.0

    def _update_command(self):
        self.time_elapsed += self._env.step_dt

        initial_phase_env_ids = (self.time_elapsed <= self.cfg.initial_phase_time).nonzero(as_tuple=False).flatten()
        if len(initial_phase_env_ids) > 0:
            self.vel_command_b[initial_phase_env_ids, :3] = 0.0
            if self.track_z_flag:
                self.vel_command_b[initial_phase_env_ids, 3] = self._sample_height_categories(initial_phase_env_ids, 5)
            else:
                self.vel_command_b[initial_phase_env_ids, 3] = 0.0

        reset_env_ids = self._env.reset_buf.nonzero(as_tuple=False).flatten()
        if len(reset_env_ids) > 0:
            self.time_elapsed[reset_env_ids] = 0.0

        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :3] = 0.0
        if len(standing_env_ids) > 0:
            self.vel_command_b[standing_env_ids, 3] = self.standing_choice[standing_env_ids]

    def _sample_height_categories(self, env_ids: Sequence[int], num_categories: int) -> torch.Tensor:
        if len(env_ids) == 0:
            return torch.tensor([], device=self.device)
        probabilities = torch.ones(num_categories, device=self.device) / num_categories
        categories = torch.linspace(self.cfg.ranges.pos_z[0], self.cfg.ranges.pos_z[1], num_categories, device=self.device)
        return categories[torch.multinomial(probabilities, len(env_ids), replacement=True)]


@configclass
class UniformVelocityWithZCommandCfg(UniformVelocityCommandCfg):
    class_type: type = UniformVelocityWithZCommand

    @configclass
    class Ranges(UniformVelocityCommandCfg.Ranges):
        pos_z: tuple[float, float] = MISSING

    ranges: Ranges = MISSING
    initial_phase_time: float = 2.0