import gymnasium as gym

from .. import agents


gym.register(
	id="Isaac-Velocity-Flat-D1h-v0",
	entry_point="isaaclab.envs:ManagerBasedRLEnv",
	disable_env_checker=True,
	kwargs={
		"env_cfg_entry_point": f"{__name__}.flat_vel_env_cfg:D1HFlatEnvCfg",
		"co_rl_cfg_entry_point": f"{agents.__name__}.co_rl_ppo_cfg:CoRlPPORunnerCfg",
		"rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
	},
)


gym.register(
	id="Isaac-Velocity-Flat-D1h-Play-v0",
	entry_point="isaaclab.envs:ManagerBasedRLEnv",
	disable_env_checker=True,
	kwargs={
		"env_cfg_entry_point": f"{__name__}.flat_vel_env_cfg:D1HFlatEnvCfg_PLAY",
		"co_rl_cfg_entry_point": f"{agents.__name__}.co_rl_ppo_cfg:CoRlPPORunnerCfg",
		"rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
	},
)
