# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""VecEnv wrapper that adds a history buffer on top of the standard
isaaclab_rl.rsl_rl.RslRlVecEnvWrapper.

The wrapper maintains a rolling buffer of the last ``history_len`` single-frame
observations per environment.  At each step it:
  1. Appends the *current* observation (before the step) to the buffer.
  2. Resets history for environments whose episode just ended.
  3. Returns a *combined* observation:
        [current_obs (obs_dim) || history_flat (history_len * obs_dim)]
     to the training runner (e.g. co-rl / rsl-rl OnPolicyRunner).

The combined observation is consumed by ActorCriticHistoryEncoder, which
internally re-splits it into the current frame and the history buffer.
"""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv


class DdtHistoryVecEnvWrapper:
    """History-augmented VecEnv wrapper for DDT-compatible training.

    Parameters
    ----------
    env : gymnasium.Env
        Raw gymnasium environment (the ManagerBasedRLEnv returned by gym.make).
    history_len : int
        Number of past frames to keep in the history buffer.
    obs_dim : int
        Dimension of a single observation frame (must match the 'policy' obs group).
    """

    def __init__(self, env, history_len: int = 10, obs_dim: int = 33):
        # Wrap with RslRlVecEnvWrapper to get the standard step/obs API
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        self._base = RslRlVecEnvWrapper(env)

        self.history_len = history_len
        self.obs_dim = obs_dim
        self.num_envs: int = self._base.num_envs
        self.num_actions: int = self._base.num_actions
        self.device: torch.device = self._base.device
        self.max_episode_length: int = self._base.max_episode_length

        # History buffer: oldest frame at index 0, newest frame at index history_len-1  [oldest ... newest]
        # Matches DDT FSMState_RL obs_history_vec_ layout
        self._history: torch.Tensor = torch.zeros(
            self.num_envs, history_len, obs_dim, device=self.device
        )
        # Cache last single-frame obs so step() can add it to history
        self._last_obs: torch.Tensor | None = None

        print(
            f"[DdtHistoryVecEnvWrapper] history_len={history_len}, obs_dim={obs_dim}\n"
            f"  Combined policy obs dim = {obs_dim + history_len * obs_dim}\n"
            f"  num_envs={self.num_envs}, num_actions={self.num_actions}"
        )

    # ------------------------------------------------------------------
    # internal utilities
    # ------------------------------------------------------------------

    def _combine(self, current_obs: torch.Tensor) -> torch.Tensor:
        """Concatenate current obs with flattened history buffer."""
        history_flat = self._history.reshape(self.num_envs, -1)        # [N, history_len*obs_dim]
        return torch.cat([current_obs, history_flat], dim=-1)           # [N, combined_dim]

    def _push_to_history(self, obs: torch.Tensor, new_obs: torch.Tensor | None = None, reset_mask: torch.Tensor | None = None):
        """Shift buffer left and insert obs as the newest entry (index -1).

        History layout: [oldest ... newest], matching DDT obs_history_vec_.

        For environments that just reset (reset_mask=True), the buffer is
        filled with the first observation of the new episode (new_obs),
        matching DDT FSMState_RL::enter() which fills all history with current obs.
        """
        # Shift older entries toward lower indices (remove oldest at index 0)
        self._history[:, :-1, :] = self._history[:, 1:, :].clone()
        # Insert the obs at the END (newest at last index)
        self._history[:, -1, :] = obs
        # For reset envs, fill all history slots with the first obs of the new episode
        if reset_mask is not None and reset_mask.any() and new_obs is not None:
            fill = new_obs[reset_mask].unsqueeze(1).expand(-1, self.history_len, -1).clone()
            self._history[reset_mask] = fill

    # ------------------------------------------------------------------
    # VecEnv API (compatible with co-rl / rsl-rl OnPolicyRunner)
    # ------------------------------------------------------------------

    @property
    def cfg(self):
        return self._base.cfg

    @property
    def unwrapped(self) -> ManagerBasedRLEnv:
        return self._base.unwrapped

    @property
    def episode_length_buf(self) -> torch.Tensor:
        return self._base.episode_length_buf

    @episode_length_buf.setter
    def episode_length_buf(self, value: torch.Tensor):
        self._base.episode_length_buf = value

    def seed(self, seed: int = -1) -> int:
        return self._base.seed(seed)

    def close(self):
        return self._base.close()

    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Return combined (current || history) obs and extras dict.

        Called once at the beginning of training by the runner.
        Initializes history with current_obs repeated history_len times,
        matching DDT FSMState_RL::enter() behaviour.
        """
        current_obs, extras = self._base.get_observations()     # [N, obs_dim]
        # Fill all history slots with current_obs (like DDT enter())
        self._history = current_obs.unsqueeze(1).expand(-1, self.history_len, -1).clone()
        self._last_obs = current_obs.clone()

        combined = self._combine(current_obs)                   # [N, combined_dim]

        # Provide combined obs also as critic obs
        extras = self._augment_extras(extras, combined)
        return combined, extras

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Step the environment and return history-augmented observations.

        Timing:
          - self._last_obs  = obs at time t   (set by previous step / get_observations)
          - new_obs         = obs at time t+1 (returned by env.step)
          - history after step contains obs[t], obs[t-1], ..., obs[t-(history_len-1)]
        """
        new_obs_raw, rewards, dones, infos = self._base.step(actions)  # [N, obs_dim]

        # Update history with the obs that was current BEFORE this step
        reset_mask = dones.bool()
        if self._last_obs is not None:
            self._push_to_history(self._last_obs, new_obs=new_obs_raw, reset_mask=reset_mask)

        # Cache the new obs for the next call
        self._last_obs = new_obs_raw.clone()

        combined = self._combine(new_obs_raw)                           # [N, combined_dim]
        infos = self._augment_extras(infos, combined)
        return combined, rewards, dones, infos

    def reset(self) -> tuple[torch.Tensor, dict]:
        """Reset all environments and initialize history with first observation."""
        obs_raw, extras = self._base.reset()
        # Fill history with initial obs (like DDT enter())
        self._history = obs_raw.unsqueeze(1).expand(-1, self.history_len, -1).clone()
        self._last_obs = obs_raw.clone()
        combined = self._combine(obs_raw)
        extras = self._augment_extras(extras, combined)
        return combined, extras

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _augment_extras(self, extras: dict, combined_obs: torch.Tensor) -> dict:
        """Ensure extras['observations'] has a 'critic' key with combined obs."""
        if "observations" not in extras:
            extras["observations"] = {}

        obs_dict = extras["observations"]

        # Replace policy obs with combined version
        obs_dict["policy"] = combined_obs

        # Build critic obs from raw critic obs (if available) combined with history
        critic_raw = obs_dict.get("critic", None)
        if critic_raw is not None and critic_raw.shape[-1] == self.obs_dim:
            # critic raw is single-frame -> augment with same history
            critic_combined = self._combine(critic_raw)
        else:
            # Fall back: use same combined obs for critic
            critic_combined = combined_obs

        obs_dict["critic"] = critic_combined
        return extras
