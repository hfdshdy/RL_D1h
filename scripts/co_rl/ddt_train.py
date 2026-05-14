"""Train script for the DDT-compatible D1H flat-velocity task.

Usage
-----
    python scripts/co_rl/ddt_train.py --task Isaac-Velocity-Flat-D1h-DDT-v0 \\
        --num_envs 4096 --max_iterations 2000

After training, the ONNX policy is exported to:
    logs/co_rl/<experiment_name>/<timestamp>/exported/flat.onnx

Place flat.onnx in:
    ddt_ros2_control/controller/rl_controller/config/d1h/

And in controllers.yaml set::

    rl_flat:
        policy_path: config/d1h/flat.onnx
        output_name: "nn_output"
        num_obs: 33
        num_actions: 8
        history_len: 10
        observations_name: ["ang_vel", "gravity", "commands", "dof_pos", "dof_vel", "last_actions"]
        commands_scale: [2.0, 2.0, 0.25]
        ang_vel_scale:   0.25
        dof_pos_scale:   1.0
        dof_vel_scale:   0.05
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

# ── argument parsing ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Train D1H DDT-compatible policy with co-rl.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=200)
parser.add_argument("--video_interval", type=int, default=2000)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-D1h-DDT-v0")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--experiment_description", type=str, default=None)

cli_args.add_co_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── rest of the imports (after Isaac Sim is running) ─────────────────────────

import gymnasium as gym
import os
import torch
from datetime import datetime

from scripts.co_rl.core.runners import OnPolicyRunner
from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

from scripts.co_rl.core.wrapper import CoRlPolicyRunnerCfg
from scripts.co_rl.core.wrapper.ddt_history_wrapper import DdtHistoryVecEnvWrapper
from scripts.co_rl.core.modules import export_ddt_policy_as_onnx

import RL_D1h.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "co_rl_ddt_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: CoRlPolicyRunnerCfg):
    """Train DDT-compatible D1H policy."""

    agent_cfg = cli_args.update_co_rl_cfg(agent_cfg, args_cli)

    # ── override CLI args ────────────────────────────────────────────────────
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs else env_cfg.scene.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.experiment_description is not None:
        agent_cfg.experiment_description = args_cli.experiment_description

    env_cfg.seed = agent_cfg.seed

    # ── DDT-specific hyper-params (from agent_cfg or defaults) ───────────────
    # These match the values documented in co_rl_ddt_ppo_cfg.py comments.
    obs_dim     = 33    # single-frame obs dimension
    history_len = 10    # history buffer length (DdtHistoryVecEnvWrapper)
    num_actions = 8     # action dimension
    latent_dim  = 32    # history encoder output dimension
    combined_obs_dim = obs_dim + history_len * obs_dim   # 363

    print(
        f"\n[ddt_train] === DDT Training Configuration ===\n"
        f"  current_obs_dim  = {obs_dim}\n"
        f"  history_len      = {history_len}\n"
        f"  combined_obs_dim = {combined_obs_dim}\n"
        f"  action_dim       = {num_actions}\n"
        f"  empirical_norm   = {agent_cfg.empirical_normalization}\n"
        f"  ONNX export      : input0=[1,{obs_dim}], input1=[1,{history_len},{obs_dim}], output=[1,{num_actions}]\n"
    )

    # ── log dir ──────────────────────────────────────────────────────────────
    log_root_path = os.path.abspath(
        os.path.join("logs", "co_rl", agent_cfg.experiment_name, "ddt_ppo")
    )
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if getattr(agent_cfg, "run_name", ""):
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    print(f"Exact experiment name requested from command line: {log_dir}")

    # ── create environment ───────────────────────────────────────────────────
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(log_dir, "videos", "train"),
            step_trigger=lambda step: step % args_cli.video_interval == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )

    # ── wrap with DDT history wrapper ────────────────────────────────────────
    env = DdtHistoryVecEnvWrapper(env, history_len=history_len, obs_dim=obs_dim)

    # ── inject history-encoder kwargs into policy config ─────────────────────
    # co-rl converts the policy configclass to a dict and passes **kwargs to
    # the actor-critic constructor.  We add the DDT-specific fields here.
    policy_dict = agent_cfg.policy.to_dict()
    policy_dict.setdefault("obs_dim",     obs_dim)
    policy_dict.setdefault("history_len", history_len)
    policy_dict.setdefault("latent_dim",  32)
    policy_dict.setdefault("history_encoder_hidden_dims", [256, 128])

    # Override policy dict in agent_cfg (co-rl reads agent_cfg.to_dict())
    agent_cfg.policy.actor_hidden_dims  = policy_dict.get("actor_hidden_dims",  [256, 128, 64])
    agent_cfg.policy.critic_hidden_dims = policy_dict.get("critic_hidden_dims", [256, 128, 64])

    # ── save resume path ─────────────────────────────────────────────────────
    if agent_cfg.resume:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # ── dump configs ─────────────────────────────────────────────────────────
    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"),   env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"),   env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # ── build runner ─────────────────────────────────────────────────────────
    agent_dict = agent_cfg.to_dict()

    # Inject DDT extra kwargs into policy so the runner passes them to the constructor
    agent_dict["policy"].update(
        {
            "obs_dim":                      obs_dim,
            "history_len":                  history_len,
            "latent_dim":                   32,
            "history_encoder_hidden_dims":  [256, 128],
        }
    )

    runner = OnPolicyRunner(env, agent_dict, log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)

    if agent_cfg.resume:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.load(resume_path)

    env.seed(agent_cfg.seed)

    # ── train ────────────────────────────────────────────────────────────────
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # ── export ONNX after training ────────────────────────────────────────────
    export_dir = os.path.join(log_dir, "exported")
    os.makedirs(export_dir, exist_ok=True)

    # JIT (pt) export
    try:
        from scripts.co_rl.core.wrapper import export_policy_as_jit
        jit_path = os.path.join(export_dir, "policy.pt")
        export_policy_as_jit(runner.alg.actor_critic, runner.obs_normalizer, path=export_dir, filename="policy.pt")
        print(f"[ddt_train] JIT policy exported to: {jit_path}")
    except Exception as e:
        print(f"[ddt_train] JIT export failed (non-critical): {e}")

    # DDT ONNX export (two-input format)
    onnx_path = os.path.join(export_dir, "flat.onnx")
    try:
        from scripts.co_rl.core.modules.actor_critic_history import export_ddt_policy_as_onnx
        export_ddt_policy_as_onnx(
            runner.alg.actor_critic,
            path=onnx_path,
            obs_dim=obs_dim,
            history_len=history_len,
            num_actions=num_actions,
        )
        print(f"\n[ddt_train] === ONNX Export Complete ===")
        print(f"  Path: {onnx_path}")
        print(f"  Copy to: ddt_ros2_control/controller/rl_controller/config/d1h/flat.onnx")
    except Exception as e:
        print(f"[ddt_train] ONNX export failed: {e}")
        import traceback
        traceback.print_exc()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
