# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Keyboard-controlled play script for D1H CO-RL policy.

Save as:
    scripts/co_rl/keyboard_play.py

Example:
    python scripts/co_rl/keyboard_play.py \
      --task Isaac-Velocity-Flat-D1h-Play-v0 \
      --algo ppo \
      --num_envs 1 \
      --num_policy_stacks 2 \
      --num_critic_stacks 2 \
      --load_run 2026-05-xx_xx-xx-xx \
      --checkpoint model_xxxx.pt

Important:
    Do not use --headless for keyboard control.
    Keyboard events come from the Isaac Sim/Omniverse window.
"""

import argparse
import os
import sys
from pathlib import Path

# Make sure running from project root works:
#   python scripts/co_rl/keyboard_play.py ...
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from isaaclab.app import AppLauncher

# Local imports. Must be before app launcher only if they do not import omni/pxr heavy modules.
import cli_args  # isort: skip
from scripts.co_rl.core.runners import OffPolicyRunner
from scripts.co_rl.core.utils.str2bool import str2bool


# -----------------------------------------------------------------------------
# Argparse
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Keyboard play for D1H CO-RL policy.")

parser.add_argument("--video", action="store_true", default=False, help="Record video during play.")
parser.add_argument("--video_length", type=int, default=1000, help="Length of the recorded video in steps.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--algo", type=str, default="ppo", help="Algorithm name: ppo or srmppo.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-D1h-Play-v0", help="Task name.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for the environment.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Try to run in real time.")
parser.add_argument("--num_policy_stacks", type=int, default=2, help="Number of policy stacks.")
parser.add_argument("--num_critic_stacks", type=int, default=2, help="Number of critic stacks.")

# Keyboard command parameters
parser.add_argument("--keyboard_lin_vel", type=float, default=0.6, help="Forward/backward velocity command in m/s.")
parser.add_argument("--keyboard_yaw_vel", type=float, default=0.5, help="Yaw velocity command in rad/s.")
parser.add_argument("--keyboard_height_step", type=float, default=0.02, help="Height command step.")
parser.add_argument("--keyboard_height_limit", type=float, default=0.05, help="Max absolute height offset command.")
parser.add_argument("--keyboard_start_height", type=float, default=0.0, help="Initial height offset command.")

# Keep compatibility with your co_rl cli args.
cli_args.add_co_rl_args(parser)

# Isaac Lab app args, includes --headless, --device, --enable_cameras, etc.
AppLauncher.add_app_launcher_args(parser)

args_cli = parser.parse_args()

# Always enable cameras when recording video.
if args_cli.video:
    args_cli.enable_cameras = True

# Launch Isaac Sim / Omniverse app.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# -----------------------------------------------------------------------------
# Everything below can import Isaac/Omni modules safely.
# -----------------------------------------------------------------------------

import time

import gymnasium as gym
import torch

import carb
import omni.appwindow
import omni
import omni.usd
from omni.kit.viewport.utility import get_viewport_from_window_name
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf, Sdf


from isaaclab.utils.math import quat_apply
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

from scripts.co_rl.core.runners import OnPolicyRunner, SRMOnPolicyRunner
from scripts.co_rl.core.wrapper import (
    CoRlPolicyRunnerCfg,
    CoRlVecEnvWrapper,
    export_env_as_pdf,
    export_policy_as_jit,
    export_policy_as_onnx,
    export_srm_as_onnx,
)

# Import D1H tasks to register gym envs.
import RL_D1h.tasks  # noqa: F401


class KeyboardVelocityCommand:
    """Keyboard controller for command layout [lin_vel_x, lin_vel_y, ang_vel_z, pos_z]."""

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        lin_vel_step: float = 0.3,
        yaw_vel_step: float = 0.3,
        height_step: float = 0.02,
        height_min: float = -0.08,
        height_max: float = 0.05,
        height_init: float = 0.0,
    ):
        self.num_envs = num_envs
        self.device = device

        self.lin_vel_step = lin_vel_step
        self.yaw_vel_step = yaw_vel_step
        self.height_step = height_step
        self.height_min = height_min
        self.height_max = height_max
        self.height_offset = max(min(height_init, height_max), height_min)

        self.active_keys: set[str] = set()
        self.commands = torch.zeros(self.num_envs, 4, device=self.device)
        self._refresh_command()

        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            self._on_keyboard_event,
        )

        print(
            "\n[Keyboard Control]\n"
            "  W / UP       : forward\n"
            "  S / DOWN     : backward\n"
            "  A / LEFT     : turn left\n"
            "  D / RIGHT    : turn right\n"
            "  Q            : height +\n"
            "  E            : height -\n"
            "  SPACE        : stop velocity\n"
            "  R            : reset height offset to 0\n"
            "  ESC          : stop velocity and reset height\n"
            "\n"
            f"  lin_vel_step    = {self.lin_vel_step:.3f} m/s\n"
            f"  yaw_vel_step    = {self.yaw_vel_step:.3f} rad/s\n"
            f"  height_step     = {self.height_step:.3f}\n"
            f"  height_range    = [{self.height_min:.3f}, {self.height_max:.3f}]\n"
        )

    def _refresh_command(self):
        vx = 0.0
        yaw = 0.0

        if "W" in self.active_keys or "UP" in self.active_keys:
            vx += self.lin_vel_step
        if "S" in self.active_keys or "DOWN" in self.active_keys:
            vx -= self.lin_vel_step

        # If turn direction is opposite to your expectation, swap these signs.
        if "A" in self.active_keys or "LEFT" in self.active_keys:
            yaw += self.yaw_vel_step
        if "D" in self.active_keys or "RIGHT" in self.active_keys:
            yaw -= self.yaw_vel_step

        self.commands[:, 0] = vx
        self.commands[:, 1] = 0.0
        self.commands[:, 2] = yaw
        self.commands[:, 3] = self.height_offset

    def _on_keyboard_event(self, event):
        key = event.input.name

        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if key in ["W", "S", "A", "D", "UP", "DOWN", "LEFT", "RIGHT"]:
                self.active_keys.add(key)

            elif key == "Q":
                self.height_offset = min(self.height_offset + self.height_step, self.height_max)
                print(f"[Keyboard] height_offset = {self.height_offset:.3f}")

            elif key == "E":
                self.height_offset = max(self.height_offset - self.height_step, self.height_min)
                print(f"[Keyboard] height_offset = {self.height_offset:.3f}")

            elif key == "R":
                self.height_offset = 0.0
                print("[Keyboard] height_offset reset to 0.000")

            elif key == "SPACE":
                self.active_keys.clear()
                print("[Keyboard] stop velocity")

            elif key == "ESCAPE":
                self.active_keys.clear()
                self.height_offset = 0.0
                print("[Keyboard] stop velocity and reset height")

            self._refresh_command()

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            if key in self.active_keys:
                self.active_keys.remove(key)
            self._refresh_command()

        return True


class SelectionCameraController:
    """Handle mouse selection and third-person camera following a selected robot.

    - Click on a robot prim to select it (expects env_* naming under /World).
    - Press `C` to toggle between perspective and third-person camera.
    - Press `ESC` to clear selection and return to perspective camera.
    """

    def __init__(self, env):
        self.env = env
        self.device = env.unwrapped.device
        self._prim_selection = omni.usd.get_context().get_selection()
        self._selected_id = None
        self._previous_selected_id = None
        self._camera_local_transform = torch.tensor([-2.5, 0.0, 0.8], device=self.device)

        self.viewport = get_viewport_from_window_name("Viewport")
        self.camera_path = "/World/Camera"
        self.perspective_path = "/OmniverseKit_Persp"
        self.create_camera()

        # Subscribe to keyboard events to handle C / ESC toggles
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)

    def create_camera(self):
        stage = omni.usd.get_context().get_stage()
        try:
            camera_prim = stage.DefinePrim(self.camera_path, "Camera")
            camera_prim.GetAttribute("focalLength").Set(8.5)
            coi_prop = camera_prim.GetProperty("omni:kit:centerOfInterest")
            if not coi_prop or not coi_prop.IsValid():
                camera_prim.CreateAttribute(
                    "omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d, True, Sdf.VariabilityUniform
                ).Set(Gf.Vec3d(0, 0, -10))
        except Exception:
            # If camera already exists or stage not ready, ignore.
            pass
        try:
            self.viewport.set_active_camera(self.perspective_path)
        except Exception:
            pass

    def _on_keyboard_event(self, event):
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "ESCAPE":
                self._prim_selection.clear_selected_prim_paths()
            elif event.input.name == "C":
                try:
                    if self.viewport.get_active_camera() == self.camera_path:
                        self.viewport.set_active_camera(self.perspective_path)
                    else:
                        self.viewport.set_active_camera(self.camera_path)
                except Exception:
                    pass

    def update_selected_object(self):
        self._previous_selected_id = self._selected_id
        selected_prim_paths = self._prim_selection.get_selected_prim_paths()
        if len(selected_prim_paths) == 0:
            self._selected_id = None
            try:
                self.viewport.set_active_camera(self.perspective_path)
            except Exception:
                pass
        elif len(selected_prim_paths) > 1:
            print("Multiple prims are selected. Please only select one!")
        else:
            prim_splitted_path = selected_prim_paths[0].split("/")
            # Find an env_N segment in the path
            env_index = None
            for part in prim_splitted_path:
                if part.startswith("env_"):
                    try:
                        env_index = int(part[4:])
                    except Exception:
                        env_index = None
                    break
            if env_index is not None:
                self._selected_id = env_index
                if self._previous_selected_id != self._selected_id:
                    try:
                        self.viewport.set_active_camera(self.camera_path)
                    except Exception:
                        pass
                self._update_camera()
            else:
                print("The selected prim was not an environment robot")

        # If selection changed, reset previous env command (if manager supports it)
        if self._previous_selected_id is not None and self._previous_selected_id != self._selected_id:
            try:
                self.env.unwrapped.command_manager.reset([self._previous_selected_id])
            except Exception:
                pass

    def _update_camera(self):
        if self._selected_id is None:
            return
        try:
            base_pos = self.env.unwrapped.scene["robot"].data.root_pos_w[self._selected_id, :]
            base_quat = self.env.unwrapped.scene["robot"].data.root_quat_w[self._selected_id, :]

            camera_pos = quat_apply(base_quat, self._camera_local_transform) + base_pos

            camera_state = ViewportCameraState(self.camera_path, self.viewport)
            eye = Gf.Vec3d(camera_pos[0].item(), camera_pos[1].item(), camera_pos[2].item())
            target = Gf.Vec3d(base_pos[0].item(), base_pos[1].item(), base_pos[2].item() + 0.6)
            camera_state.set_position_world(eye, True)
            camera_state.set_target_world(target, True)
        except Exception:
            pass


def _set_keyboard_demo_env_cfg(env_cfg):
    """Make the env suitable for interactive keyboard play."""

    env_cfg.scene.num_envs = args_cli.num_envs

    # Very long episode for interactive play.
    # This does not change obs/action dimensions.
    env_cfg.episode_length_s = 1_000_000.0

    # Disable command resampling and initial zero-command phase.
    # We will overwrite the command tensor every step.
    env_cfg.commands.base_velocity.initial_phase_time = 0.0
    env_cfg.commands.base_velocity.resampling_time_range = (1_000_000.0, 1_000_000.0)

    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    env_cfg.commands.base_velocity.ranges.pos_z = (args_cli.keyboard_start_height, args_cli.keyboard_start_height)

    # For visual/demo play, reduce random disturbances.
    # Keep base contact termination if you want to know when it really falls.
    if hasattr(env_cfg.events, "push_robot"):
        env_cfg.events.push_robot = None
    if hasattr(env_cfg.events, "add_base_mass"):
        env_cfg.events.add_base_mass = None
    if hasattr(env_cfg.events, "randomize_com_positions"):
        env_cfg.events.randomize_com_positions = None

    # Fixed reset is easier for demo.
    if hasattr(env_cfg.events, "reset_robot_joints") and env_cfg.events.reset_robot_joints is not None:
        env_cfg.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
        env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)

    if hasattr(env_cfg.events, "reset_base") and env_cfg.events.reset_base is not None:
        env_cfg.events.reset_base.params = {
            "pose_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        }

    # Observation corruption/noise off for clean inference.
    for group_name in ["stack_policy", "none_stack_policy", "stack_critic", "none_stack_critic"]:
        if hasattr(env_cfg.observations, group_name):
            getattr(env_cfg.observations, group_name).enable_corruption = False

    return env_cfg


def main():
    """Keyboard play with trained CO-RL agent."""

    if args_cli.headless:
        print(
            "\n[WARNING] You are running with --headless. "
            "Omniverse keyboard events normally require a visible Isaac Sim window. "
            "Keyboard control may not work in true headless SSH mode.\n"
        )

    # Parse environment and agent configuration.
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    env_cfg = _set_keyboard_demo_env_cfg(env_cfg)

    agent_cfg: CoRlPolicyRunnerCfg = cli_args.parse_co_rl_cfg(args_cli.task, args_cli)
    agent_cfg.num_policy_stacks = (
        args_cli.num_policy_stacks if args_cli.num_policy_stacks is not None else agent_cfg.num_policy_stacks
    )
    agent_cfg.num_critic_stacks = (
        args_cli.num_critic_stacks if args_cli.num_critic_stacks is not None else agent_cfg.num_critic_stacks
    )

    is_off_policy = False if agent_cfg.to_dict()["algorithm"]["class_name"] in ["PPO", "SRMPPO"] else True

    # Logging/checkpoint path.
    log_root_path = os.path.join("logs", "co_rl", agent_cfg.experiment_name, args_cli.algo)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if getattr(args_cli, "use_pretrained_checkpoint", False):
        resume_path = get_published_pretrained_checkpoint("co_rl", args_cli.task)
        if not resume_path:
            print("[INFO] No published pre-trained checkpoint is available for this task.")
            return
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    print(f"[INFO] Loading model checkpoint from: {resume_path}")

    # Create environment.
    render_mode = "rgb_array" if args_cli.video else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    # Convert multi-agent to single-agent if needed.
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Optional video recording.
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "keyboard_play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording video during keyboard play.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Wrap for CO-RL observation stacking.
    env = CoRlVecEnvWrapper(env, agent_cfg)

    # Create runner.
    if is_off_policy:
        runner = OffPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        if args_cli.algo == "srmppo":
            runner = SRMOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif args_cli.algo == "ppo":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported algo for this script: {args_cli.algo}")

    runner.load(resume_path)

    # Optional SRM encoder.
    srm = None
    if hasattr(runner.alg, "srm") and hasattr(runner.alg, "srm_fc"):
        srm = runner.alg.srm

    # Get inference policy.
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Export policy, same as play.py.
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if is_off_policy:
        export_policy_as_jit(runner.alg, runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(runner.alg, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx")
    else:
        export_policy_as_jit(runner.alg.actor_critic, runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(
            runner.alg.actor_critic,
            normalizer=runner.obs_normalizer,
            path=export_model_dir,
            filename="policy.onnx",
        )

    if args_cli.algo == "srmppo":
        export_srm_as_onnx(
            runner.alg.srm,
            runner.alg.srm_fc,
            device=agent_cfg.device,
            path=export_model_dir,
            filename="srm.onnx",
        )

    try:
        export_env_as_pdf(
            yaml_path=os.path.join(log_dir, "params", "env.yaml"),
            pdf_path=os.path.join(export_model_dir, "env.pdf"),
        )
    except Exception as exc:
        print(f"[WARNING] Failed to export env pdf: {exc}")

    # Reset environment.
    obs, _ = env.get_observations()

    # Keyboard controller.
    keyboard = KeyboardVelocityCommand(
        num_envs=env.num_envs,
        device=env.unwrapped.device,
        lin_vel_step=args_cli.keyboard_lin_vel,
        yaw_vel_step=args_cli.keyboard_yaw_vel,
        height_step=args_cli.keyboard_height_step,
        height_min=-abs(args_cli.keyboard_height_limit),
        height_max=abs(args_cli.keyboard_height_limit),
        height_init=args_cli.keyboard_start_height,
    )

    # Selection & third-person camera controller (optional)
    try:
        selection_controller = SelectionCameraController(env)
    except Exception:
        selection_controller = None

    # Command term.
    base_command_term = env.unwrapped.command_manager.get_term("base_velocity")

    timestep = 0
    last_time = time.time()

    # Play loop.
    while simulation_app.is_running():
        # update selection and camera each frame
        if selection_controller is not None:
            selection_controller.update_selected_object()
        with torch.inference_mode():
            # Write keyboard command before policy inference.
            # If a robot is selected, apply keyboard command only to that env index.
            if selection_controller is not None and selection_controller._selected_id is not None:
                # start with the default/last command tensor
                default_cmds = env.unwrapped.command_manager.get_term("base_velocity").vel_command_b.clone()
                sid = selection_controller._selected_id
                try:
                    default_cmds[sid : sid + 1, :] = keyboard.commands[sid : sid + 1, :]
                except Exception:
                    # fallback: apply keyboard.commands to the selected row
                    default_cmds[sid : sid + 1, :] = keyboard.commands[0:1, :]
                base_command_term.vel_command_b[:] = default_cmds
            else:
                base_command_term.vel_command_b[:] = keyboard.commands

            if srm is not None:
                encoded_obs = runner.alg.encode_obs(obs)
                actions = policy(encoded_obs)
            else:
                actions = policy(obs)

            # Match your play.py behavior: clip actions before stepping.
            clipped_actions = torch.clamp(actions, -1.0, 1.0)

            # Write again right before env.step(). Use selection logic.
            if selection_controller is not None and selection_controller._selected_id is not None:
                sid = selection_controller._selected_id
                try:
                    base_command_term.vel_command_b[:] = default_cmds
                except Exception:
                    base_command_term.vel_command_b[:] = keyboard.commands
            else:
                base_command_term.vel_command_b[:] = keyboard.commands
            obs, _, dones, extras = env.step(clipped_actions)

            # Write after env.step() so the next computed observation sees the latest command.
            if selection_controller is not None and selection_controller._selected_id is not None:
                try:
                    base_command_term.vel_command_b[:] = default_cmds
                except Exception:
                    base_command_term.vel_command_b[:] = keyboard.commands
            else:
                base_command_term.vel_command_b[:] = keyboard.commands

        if args_cli.real_time:
            # Policy step is normally sim.dt * decimation, but use env cfg if available.
            try:
                step_dt = env.unwrapped.step_dt
            except Exception:
                step_dt = 0.02

            elapsed = time.time() - last_time
            sleep_time = step_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
            last_time = time.time()

        if args_cli.video:
            timestep += 1
            if timestep >= args_cli.video_length:
                break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()