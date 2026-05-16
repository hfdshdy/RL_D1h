from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_out_of_bounds(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distance_buffer: float = 3.0,
) -> torch.Tensor:
    """Terminate when the robot approaches the terrain boundary."""
    if env.scene.cfg.terrain.terrain_type == "plane":
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if env.scene.cfg.terrain.terrain_type != "generator":
        raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'.")

    terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
    grid_width, grid_length = terrain_gen_cfg.size
    map_width = terrain_gen_cfg.num_rows * grid_width + 2 * terrain_gen_cfg.border_width
    map_height = terrain_gen_cfg.num_cols * grid_length + 2 * terrain_gen_cfg.border_width

    asset: RigidObject = env.scene[asset_cfg.name]
    x_out = torch.abs(asset.data.root_pos_w[:, 0]) > 0.5 * map_width - distance_buffer
    y_out = torch.abs(asset.data.root_pos_w[:, 1]) > 0.5 * map_height - distance_buffer
    return torch.logical_or(x_out, y_out)


def illegal_contact_after_time(
    env: ManagerBasedRLEnv,
    threshold: float,
    start_time_s: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Terminate on illegal contact only after a configured elapsed episode time."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    has_illegal_contact = torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold,
        dim=1,
    )
    elapsed_time = env.episode_length_buf * env.step_dt
    return torch.logical_and(has_illegal_contact, elapsed_time >= start_time_s)