from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def base_lin_vel_x_link(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_b[:, 0:1]


def base_lin_vel_y_link(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_b[:, 1:2]


def base_lin_vel_z_link(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_link_lin_vel_b[:, 2:3]


def base_ang_vel_link(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_link_ang_vel_b


def base_pos_z_rel_link(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    if sensor_cfg is None:
        return asset.data.root_link_pos_w[:, 2:3]

    sensor: RayCaster = env.scene[sensor_cfg.name]
    terrain_height = torch.mean(sensor.data.ray_hits_w[..., 2], dim=1, keepdim=True)
    return asset.data.root_link_pos_w[:, 2:3] - terrain_height


def current_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, "reward_buf"):
        return torch.zeros((env.num_envs, 1), dtype=torch.float32, device=env.device)
    return env.reward_buf.unsqueeze(-1)


def is_contact(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return contact.float()


def lift_mask_by_height_scan(
    env: ManagerBasedRLEnv,
    sensor_cfg_left: SceneEntityCfg,
    sensor_cfg_right: SceneEntityCfg,
    command_name: str = "base_velocity",
) -> torch.Tensor:
    left_mask = env.scene.sensors[sensor_cfg_left.name].data.mask
    right_mask = env.scene.sensors[sensor_cfg_right.name].data.mask
    lift_mask = torch.stack([left_mask, right_mask], dim=1)
    command_norm = torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1)
    return lift_mask * (command_norm > 0.1).unsqueeze(-1).float()


def generated_scaled_commands(env: ManagerBasedRLEnv, command_name: str, scale: tuple[float, ...]) -> torch.Tensor:
    scaled_command = env.command_manager.get_command(command_name).clone()
    scaled_command[:, : len(scale)] *= torch.tensor(scale, device=env.device)
    return scaled_command


def joint_pos_leg_gear(
    env: ManagerBasedEnv,
    gear_ratio: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids] * gear_ratio


def joint_vel_leg_gear(
    env: ManagerBasedEnv,
    gear_ratio: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids] * gear_ratio