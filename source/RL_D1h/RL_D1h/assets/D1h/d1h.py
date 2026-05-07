from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg


D1H_ASSET_DIR = Path(__file__).resolve().parent / "D1h"
D1H_ASSET_DIR = Path(__file__).resolve().parent
D1H_USD_PATH = str(D1H_ASSET_DIR / "d1h.usd")


D1H_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=D1H_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.55),
        joint_pos={
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "FL_thigh_joint": 0.6,
            "FR_thigh_joint": 0.6,
            "FL_calf_joint": -1.4,
            "FR_calf_joint": -1.4,
            "FL_foot_joint": 0.0,
            "FR_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "FL_hip_joint",
                "FR_hip_joint",
                "FL_thigh_joint",
                "FR_thigh_joint",
                "FL_calf_joint",
                "FR_calf_joint",
            ],
            effort_limit=60.0,
            velocity_limit=20.0,
            stiffness=40.0,
            damping=2.0,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[
                "FL_foot_joint",
                "FR_foot_joint",
            ],
            effort_limit=12.0,
            velocity_limit=12.5,
            stiffness=0.0,
            damping=1.0,
        ),
    },
)