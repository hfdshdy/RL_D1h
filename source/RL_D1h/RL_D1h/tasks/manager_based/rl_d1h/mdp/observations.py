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


# ---------------------------------------------------------------------------
# DDT-compatible observation functions
#
# These functions produce observations whose order, scale and meaning exactly
# match the ddt_ros2_control rl_controller interface defined in
# controller/rl_controller/config/d1h/controllers.yaml (rl_flat policy).
#
# Single-frame observation layout (33 dims):
#   [ang_vel ×3 | gravity ×3 | commands ×3 | dof_pos ×8 | dof_vel ×8 | last_actions ×8]
#
# DDT joint order:
#   [FL_hip, FL_thigh, FL_calf, FL_foot, FR_hip, FR_thigh, FR_calf, FR_foot]
# ---------------------------------------------------------------------------

# Canonical DDT joint order used when building SceneEntityCfg in the env config.
DDT_JOINT_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FL_foot_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FR_foot_joint",
]

# Indices of wheel joints within the DDT-ordered joint list (FL_foot=3, FR_foot=7)
_DDT_WHEEL_LOCAL_IDX = [3, 7]


def ddt_dof_pos(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint positions (relative to default) in DDT joint order.

    - Joint positions are subtracted from the robot's default joint positions
      (set via ``init_state.joint_pos`` in the ArticulationCfg).
    - Wheel joint positions (indices 3 and 7 in DDT order) are zeroed because
      continuous rotation makes the absolute angle meaningless for the policy.
    - Scale = 1.0 (matching ddt_ros2_control ``dof_pos_scale: 1.0``).

    The asset_cfg must specify joint_names in DDT order, e.g.::

        SceneEntityCfg("robot", joint_names=DDT_JOINT_NAMES)
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]            # [B, 8] DDT order
    default_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]  # [B, 8]
    result = joint_pos - default_pos
    # Zero wheel joints (no meaningful position signal for velocity wheels)
    result[:, _DDT_WHEEL_LOCAL_IDX[0]] = 0.0
    result[:, _DDT_WHEEL_LOCAL_IDX[1]] = 0.0
    return result


def ddt_dof_vel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Joint velocities in DDT joint order.

    Apply ``scale=0.05`` in the ObsTerm to match ddt_ros2_control
    ``dof_vel_scale: 0.05``.
    The asset_cfg must specify joint_names in DDT order.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel[:, asset_cfg.joint_ids]                 # [B, 8] DDT order


def ddt_velocity_commands(
    env: "ManagerBasedRLEnv",
    command_name: str,
    scale: tuple[float, float, float] = (2.0, 2.0, 0.25),
) -> torch.Tensor:
    """Velocity commands [lin_vel_x, lin_vel_y, ang_vel_z] with DDT scales.

    Returns exactly 3 dimensions (the pos_z component of the command is
    intentionally dropped here; it is used only by reward functions).

    Default scales match ddt_ros2_control
    ``commands_scale: [2.0, 2.0, 0.25]``.
    """
    cmd = env.command_manager.get_command(command_name)                 # [B, 4]
    lin_vel_x = cmd[:, 0:1] * scale[0]
    lin_vel_y = cmd[:, 1:2] * scale[1]
    ang_vel_z = cmd[:, 2:3] * scale[2]
    return torch.cat([lin_vel_x, lin_vel_y, ang_vel_z], dim=-1)        # [B, 3]