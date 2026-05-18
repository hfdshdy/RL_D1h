# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play script for the DDT-compatible D1H flat-velocity task.

Usage
-----
    python scripts/co_rl/ddt_play.py --task Isaac-Velocity-Flat-D1h-DDT-v0 \
        --load_run 2026-05-18_12-00-00 --checkpoint model_2000.pt

The script restores a DDT checkpoint, rebuilds the history-augmented policy
observation used during training, and runs deterministic inference in Isaac Sim.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

OBS_DIM = 33
HISTORY_LEN = 10
LATENT_DIM = 32
NUM_ACTIONS = 8
HISTORY_ENCODER_HIDDEN_DIMS = [256, 128]


parser = argparse.ArgumentParser(description="Play a D1H DDT-compatible policy with co-rl.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video during inference.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video in steps.")
parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-D1h-DDT-v0", help="Task name.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment.")

cli_args.add_co_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
import os
import torch

from scripts.co_rl.core.runners import OnPolicyRunner
from scripts.co_rl.core.wrapper import CoRlPolicyRunnerCfg
from scripts.co_rl.core.wrapper.ddt_history_wrapper import DdtHistoryVecEnvWrapper
from isaaclab.envs import DirectMARLEnv, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import RL_D1h.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "co_rl_ddt_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: CoRlPolicyRunnerCfg):
    """Run deterministic inference with a DDT-compatible D1H policy."""

    agent_cfg = cli_args.update_co_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed

    log_root_path = os.path.abspath(os.path.join("logs", "co_rl", agent_cfg.experiment_name, "ddt_ppo"))
    print(f"[INFO] Loading DDT experiment from directory: {log_root_path}")

    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    print(
        f"\n[ddt_play] === DDT Inference Configuration ===\n"
        f"  current_obs_dim  = {OBS_DIM}\n"
        f"  history_len      = {HISTORY_LEN}\n"
        f"  combined_obs_dim = {OBS_DIM + HISTORY_LEN * OBS_DIM}\n"
        f"  action_dim       = {NUM_ACTIONS}\n"
        f"  checkpoint       = {resume_path}\n"
    )

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = DdtHistoryVecEnvWrapper(env, history_len=HISTORY_LEN, obs_dim=OBS_DIM)

    agent_dict = agent_cfg.to_dict()
    agent_dict["policy"].update(
        {
            "obs_dim": OBS_DIM,
            "history_len": HISTORY_LEN,
            "latent_dim": LATENT_DIM,
            "history_encoder_hidden_dims": HISTORY_ENCODER_HIDDEN_DIMS,
        }
    )

    runner = OnPolicyRunner(env, agent_dict, log_dir=None, device=agent_cfg.device)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.load(resume_path)

    policy = runner.get_inference_policy(device=env.unwrapped.device)
    obs, _ = env.get_observations()

    timestep = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            actions = policy(obs)
            clipped_actions = torch.clamp(actions, -1.0, 1.0)
            obs, _, _, _ = env.step(clipped_actions)

        if args_cli.video:
            timestep += 1
            if timestep >= args_cli.video_length:
                break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()