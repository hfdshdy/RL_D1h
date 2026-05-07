from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Update terrain difficulty based on walked distance and commanded speed."""
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    distance = torch.norm(asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    move_down = distance < torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up

    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


def modify_base_velocity_range(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], term_name: str, mod_range: dict, num_steps: int
):
    """Expand command ranges after the configured number of common steps."""
    del env_ids
    if env.common_step_counter < num_steps:
        return

    command_term = env.command_manager.get_term(term_name)
    for key, target_range in mod_range.items():
        if hasattr(command_term.cfg.ranges, key):
            setattr(command_term.cfg.ranges, key, target_range)