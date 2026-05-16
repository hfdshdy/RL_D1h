# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DDT-compatible flat-terrain environment configuration for D1H.

Key design choices
------------------
- Observation structure exactly matches ddt_ros2_control's rl_flat policy spec
  (33-dim single frame in DDT joint order).
- History (10 frames) is managed externally by DdtHistoryVecEnvWrapper so the
  env itself only exposes the 33-dim current-frame observations.
- Action groups are defined in DDT joint order with DDT-matching scales so that
  ``last_action`` returned by the env also follows DDT order.
- Policy frequency is 100 Hz (sim.dt=0.005 s, decimation=2).
- PD gains and torque limits are aligned with rl_flat in controllers.yaml.

Observation layout (33 dims)
-----------------------------
    Index   Field            Dim   Scale   Notes
    ------  ---------------  ---   -----   ----------------------
    0-2     ang_vel           3    0.25    IMU body-frame angular velocity
    3-5     gravity           3    1.0     projected gravity vector
    6-8     velocity_cmds     3    (2,2,0.25)  [lin_x, lin_y, ang_z]
    9-16    dof_pos           8    1.0     joint pos - default; wheels=0
    17-24   dof_vel           8    0.05    joint velocities
    25-32   last_actions      8    1.0     previous policy output

DDT joint order for dof_pos / dof_vel / last_actions
-----------------------------------------------------
    [FL_hip, FL_thigh, FL_calf, FL_foot,
     FR_hip, FR_thigh, FR_calf, FR_foot]

Action groups (must match joint order above for last_action consistency)
------------------------------------------------------------------------
    fl_hip_pos      : FL_hip_joint      scale=0.25  position
    fl_thigh_calf_pos: FL_thigh/calf   scale=0.50  position
    fl_wheel_vel    : FL_foot_joint     scale=10.0  velocity
    fr_hip_pos      : FR_hip_joint      scale=0.25  position
    fr_thigh_calf_pos: FR_thigh/calf   scale=0.50  position
    fr_wheel_vel    : FR_foot_joint     scale=10.0  velocity
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import RL_D1h.tasks.manager_based.rl_d1h.mdp as mdp
from RL_D1h.assets.D1h import D1H_CFG
from RL_D1h.tasks.manager_based.rl_d1h.velocity_env_cfg import LocomotionVelocityFlatEnvCfg
from RL_D1h.tasks.manager_based.rl_d1h.mdp.observations import DDT_JOINT_NAMES


# ---------------------------------------------------------------------------
# DDT observation spec (two groups: policy + critic)
# ---------------------------------------------------------------------------

@configclass
class DdtObservationsCfg:
    """33-dim single-frame observations compatible with ddt_ros2_control rl_flat."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations – with sensor noise for domain randomisation."""

        # 3 dims: ang_vel * 0.25
        ang_vel = ObsTerm(
            func=mdp.base_ang_vel_link,
            noise=Unoise(n_min=-0.15, n_max=0.15),
            scale=0.25,
        )

        # 3 dims: projected gravity (normalised, no extra scale)
        gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        # 3 dims: [lin_vel_x * 2.0, lin_vel_y * 2.0, ang_vel_z * 0.25]
        velocity_commands = ObsTerm(
            func=mdp.ddt_velocity_commands,
            params={"command_name": "base_velocity", "scale": (2.0, 2.0, 0.25)},
        )

        # 8 dims: dof_pos - default, DDT order, wheel joints zeroed (scale=1.0)
        dof_pos = ObsTerm(
            func=mdp.ddt_dof_pos,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=DDT_JOINT_NAMES),
            },
        )

        # 8 dims: dof_vel * 0.05, DDT order
        dof_vel = ObsTerm(
            func=mdp.ddt_dof_vel,
            noise=Unoise(n_min=-1.5, n_max=1.5),
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=DDT_JOINT_NAMES),
            },
            scale=0.05,
        )

        # 8 dims: raw policy output from previous step
        last_actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations – same structure, no noise (privileged)."""

        ang_vel = ObsTerm(func=mdp.base_ang_vel_link, scale=0.25)
        gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(
            func=mdp.ddt_velocity_commands,
            params={"command_name": "base_velocity", "scale": (2.0, 2.0, 0.25)},
        )
        dof_pos = ObsTerm(
            func=mdp.ddt_dof_pos,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=DDT_JOINT_NAMES),
            },
        )
        dof_vel = ObsTerm(
            func=mdp.ddt_dof_vel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=DDT_JOINT_NAMES),
            },
            scale=0.05,
        )
        last_actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # Observation groups (named "policy" and "critic" for RslRlVecEnvWrapper)
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# ---------------------------------------------------------------------------
# DDT action spec (6 groups in DDT joint order for last_action consistency)
# ---------------------------------------------------------------------------

@configclass
class DdtActionsCfg:
    """8-dim actions in DDT joint order with DDT-matching scales.

    Concatenation order of the 6 groups gives last_action in DDT order:
        [FL_hip | FL_thigh, FL_calf | FL_foot | FR_hip | FR_thigh, FR_calf | FR_foot]
    = [FL_hip, FL_thigh, FL_calf, FL_foot, FR_hip, FR_thigh, FR_calf, FR_foot]  ✓

    Scales match ``action_scales`` in controllers.yaml:
        hip=0.25, thigh=0.50, calf=0.50
    Wheel velocity scale 10.0 rad/s is chosen to be comparable to DDT's
    effective wheel speed (~8.3 rad/s per unit action with its PD controller).
    """

    # --- left side ---
    fl_hip_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["FL_hip_joint"],
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
    )
    fl_thigh_calf_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["FL_thigh_joint", "FL_calf_joint"],
        scale=0.5,
        use_default_offset=True,
        preserve_order=True,
    )
    fl_wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["FL_foot_joint"],
        scale=0.5,
        use_default_offset=False,
        preserve_order=True,
    )

    # --- right side ---
    fr_hip_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["FR_hip_joint"],
        scale=0.25,
        use_default_offset=True,
        preserve_order=True,
    )
    fr_thigh_calf_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["FR_thigh_joint", "FR_calf_joint"],
        scale=0.5,
        use_default_offset=True,
        preserve_order=True,
    )
    fr_wheel_vel = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["FR_foot_joint"],
        scale=0.5,
        use_default_offset=False,
        preserve_order=True,
    )


# ---------------------------------------------------------------------------
# Rewards (reuse most of the original flat-vel rewards)
# ---------------------------------------------------------------------------

@configclass
class DdtRewardsCfg:
    """Reward specification for the DDT-compatible task."""

    # --- task tracking ---
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.25},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_link_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-20.0)

    # --- regularisation ---
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_link_l2, weight=-1.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_link_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0) #惩罚机身不水平

    # --- joint limits & tracking ---
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-2.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_joint"])},
    )
    joint_deviation_calf = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-2.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_calf_joint"])},
    )
    dof_pos_limits_hip = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint")},
    )
    dof_pos_limits_calf = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_calf_joint")},
    )
    dof_pos_limits_leg = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_thigh_joint")},
    )
    base_contact_soft = RewTerm(
        func=mdp.undesired_contacts_after_time,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["base_link"]
            ),
            "threshold": 20.0,
            "start_time_s": 0.5,
        },
    )
    leg_contact_soft = RewTerm(
        func=mdp.undesired_contacts_after_time,
        weight=-4.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=[".*_thigh", ".*_calf", ".*_hip"]
            ),
            "threshold": 20.0,
            "start_time_s": 1.5,
        },
    )
    joint_applied_torque_limits = RewTerm(
        func=mdp.applied_torque_limits,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_joint")},
    )
    calf_align_l1 = RewTerm(
        func=mdp.joint_align_l1,
        weight=-0.6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_calf_joint")},
    )
    leg_align_l1 = RewTerm(
        func=mdp.joint_align_l1,
        weight=-0.6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_thigh_joint")},
    )

    # --- height tracking ---
    base_height = RewTerm(
        func=mdp.track_pos_z_rel_exp,
        weight=5.0,
        params={
            "temperature": 30.0,
            "default_height": 0.45,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": None,
        },
    )

    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-5.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.0005)


# ---------------------------------------------------------------------------
# Main environment config
# ---------------------------------------------------------------------------

@configclass
class D1hDdtFlatEnvCfg(LocomotionVelocityFlatEnvCfg):
    """DDT-compatible flat terrain environment for D1H.

    Inherits from LocomotionVelocityFlatEnvCfg (terrain/terminations/events base)
    and replaces observations, actions and rewards with DDT-aligned versions.

    Training / deployment correspondence
    -------------------------------------
    - Policy frequency: 100 Hz (decimation=2 × sim.dt=0.005 s)
    - Leg PD: kp=40.0, kd=1.2  (matches rl_flat joint_kp/kd for legs)
    - Wheel control: velocity-based (kp=0, kd=1.0), scale=10 rad/s
    - Default joint pos: [0.0, 0.8, -1.5, 0.0, ...] (matches DDT defaults)
    """

    # ---- replace inherited obs / actions / rewards ----
    observations: DdtObservationsCfg = DdtObservationsCfg()
    actions: DdtActionsCfg = DdtActionsCfg()
    rewards: DdtRewardsCfg = DdtRewardsCfg()

    def __post_init__(self):
        # --- call the flat-terrain base setup (terrain=plane, no curriculum) ---
        super().__post_init__()

        # ---- robot: set DDT-matching defaults and PD gains ----
        self.scene.robot = D1H_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Override init_state to match DDT default_joint_angles
        # DDT: [0.0, 0.8, -1.5, 0.0, 0.0, 0.8, -1.5, 0.0]
        self.scene.robot.init_state.joint_pos = {
            "FL_hip_joint":   0.0,
            "FL_thigh_joint": 0.8,
            "FL_calf_joint":  -1.5,
            "FL_foot_joint":  0.0,
            "FR_hip_joint":   0.0,
            "FR_thigh_joint": 0.8,
            "FR_calf_joint":  -1.5,
            "FR_foot_joint":  0.0,
        }
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.17)

        # Override leg actuator damping to match DDT kd=1.2
        self.scene.robot.actuators["legs"] = ImplicitActuatorCfg(
            joint_names_expr=[
                "FL_hip_joint", "FR_hip_joint",
                "FL_thigh_joint", "FR_thigh_joint",
                "FL_calf_joint", "FR_calf_joint",
            ],
            effort_limit=80.0,           # matches DDT torque_limit for legs
            velocity_limit=20.0,
            stiffness=40.0,              # kp matches DDT joint_kp legs
            damping=1.2,                 # kd matches DDT joint_kd legs
        )

        # Wheel actuator: velocity control (kp=0, kd=1.0)
        self.scene.robot.actuators["wheels"] = ImplicitActuatorCfg(
            joint_names_expr=["FL_foot_joint", "FR_foot_joint"],
            effort_limit=20.0,           # matches DDT torque_limit for wheels
            velocity_limit=12.5,
            stiffness=0.0,
            damping=0.6,
        )

        # ---- policy frequency: 100 Hz ----
        # sim.dt = 0.0025 s, decimation = 4  →  policy_dt = 0.01 s = 100 Hz
        self.decimation = 4

        # ---- disable all height/mask sensors (not needed for flat terrain) ----
        self.scene.height_scanner = None
        self.scene.base_height_scanner = None
        self.scene.left_wheel_height_scanner = None
        self.scene.right_wheel_height_scanner = None
        self.scene.left_mask_sensor = None
        self.scene.right_mask_sensor = None

        # ---- physx tweaks ----
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # ---- curriculum ----
        self.curriculum.terrain_levels = None

        # ---- events ----
        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_fixed_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=list(DDT_JOINT_NAMES),
                    preserve_order=True,
                ),
                "joint_pos": mdp.D1H_TRANSUP_JOINT_OFFSETS,
                "joint_vel": 0.0,
            },
        )
        self.events.push_robot.interval_range_s = (10.0, 15.0)
        self.events.push_robot.params = {
            "velocity_range": {"x": (-0.1, 0.05), "y": (-0.01, 0.01), "z": (-0.01, 0.01)},
        }
        self.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (-0.75, 1.0)
        self.events.physics_material.params["asset_cfg"].body_names = [
            ".*_hip", ".*_thigh", ".*_calf", ".*_foot", ".*base.*"
        ]
        self.events.physics_material.params["static_friction_range"] = (0.7, 1.0)
        self.events.physics_material.params["dynamic_friction_range"] = (0.6, 0.8)

        # commands
        self.commands.base_velocity.ranges.lin_vel_x = (-0.02, 0.02)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.01, 0.01)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.02, 0.02)
        # self.commands.base_velocity.ranges.heading = (-math.pi, math.pi)  #目标朝向
        self.commands.base_velocity.ranges.pos_z = (-0.01, 0.01)

        # terminations
        self.terminations.terrain_out_of_bounds = None
        self.terminations.base_contact = DoneTerm(
            func=mdp.illegal_contact_after_time,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[ "base_link",
                 ".*_hip",
                 ".*_calf", 
                 ".*_thigh",
                 ]),
                "threshold": 100.0,
                "start_time_s": 0.5,
            },
        )

        


        # Print summary for verification
        print(
            "\n[D1hDdtFlatEnvCfg] === DDT-Compatible Environment Summary ===\n"
            f"  policy obs dim   : 33  (per frame)\n"
            f"  history_len      : 10  (managed by DdtHistoryVecEnvWrapper)\n"
            f"  combined obs dim : {33 + 10 * 33}  (current + history flat)\n"
            f"  action dim       : 8   (DDT order)\n"
            f"  policy frequency : 100 Hz (decimation={self.decimation}, sim.dt={self.sim.dt})\n"
            f"  joint order      : {DDT_JOINT_NAMES}\n"
            "  ONNX expected    : input0=[1,33], input1=[1,10,33], output=[1,8]\n"
        )


@configclass
class D1hDdtFlatEnvCfg_PLAY(D1hDdtFlatEnvCfg):
    """Play (evaluation) version: fewer envs, no video sensors."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.episode_length_s = 30.0
        # Disable domain randomisation during play
        self.events.push_robot = None
        self.events.add_base_mass = None
        self.events.randomize_com_positions = None
