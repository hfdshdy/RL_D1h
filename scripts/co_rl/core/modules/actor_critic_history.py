# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Actor-Critic with a history encoder for DDT-compatible deployment.

Architecture:
    Training input (flat):  [current_obs (obs_dim) || history_flat (history_len * obs_dim)]
    Internal pipeline:      history_encoder(history_flat) -> latent
                            actor(cat(current_obs, latent))  -> actions
    ONNX export (2-input):  current_obs [B, obs_dim]  +  history_obs [B, history_len, obs_dim]
                            -> actions [B, num_actions]

This design lets the training pipeline feed a single flat tensor (standard rsl-rl /
co-rl OnPolicyRunner API) while the exported ONNX file matches the two-input interface
expected by ddt_ros2_control's rl_controller.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _get_activation(act_name: str) -> nn.Module:
    activations = {
        "elu": nn.ELU(),
        "selu": nn.SELU(),
        "relu": nn.ReLU(),
        "lrelu": nn.LeakyReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
    }
    act = activations.get(act_name)
    if act is None:
        print(f"[ActorCriticHistoryEncoder] Unknown activation '{act_name}', using ELU.")
        act = nn.ELU()
    return act


def _build_mlp(in_dim: int, hidden_dims: list[int], out_dim: int, activation: nn.Module) -> nn.Sequential:
    """Build a fully-connected MLP with the given layer sizes."""
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dims[0]), activation]
    for i in range(len(hidden_dims) - 1):
        layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), _get_activation(activation.__class__.__name__.lower())]
    layers.append(nn.Linear(hidden_dims[-1], out_dim))
    return nn.Sequential(*layers)


def _build_mlp_str(
    in_dim: int,
    hidden_dims: list[int],
    out_dim: int,
    act_name: str,
    use_layer_norm: bool = False,
) -> nn.Sequential:
    """Convenience wrapper that accepts an activation name string.

    When ``use_layer_norm=True`` each hidden layer becomes:
        Linear -> LayerNorm -> Activation
    The final output layer has no norm or activation.
    """
    layers: list[nn.Module] = []
    dims = [in_dim] + hidden_dims + [out_dim]
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:          # no activation after the last linear layer
            if use_layer_norm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            layers.append(_get_activation(act_name))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# main module
# ---------------------------------------------------------------------------

class ActorCriticHistoryEncoder(nn.Module):
    """PPO Actor-Critic that encodes a history buffer into a latent vector.

    Parameters
    ----------
    num_actor_obs : int
        Flat dimension of the *training* policy observation, i.e.
        ``obs_dim + history_len * obs_dim``.  E.g. 33 + 10*33 = 363.
    num_critic_obs : int
        Same structure but for the critic (can be identical to num_actor_obs
        when no privileged observations are available).
    num_actions : int
        Dimension of the action space.  E.g. 8 for D1H.
    obs_dim : int
        Dimension of a single observation frame (e.g. 33).
    history_len : int
        Number of past frames stored in the history buffer (e.g. 10).
    history_encoder_hidden_dims : list[int]
        Hidden layer sizes for the history-encoder MLP.
    latent_dim : int
        Output dimension of the history encoder (latent vector size).
    actor_hidden_dims : list[int]
        Hidden layer sizes for the actor MLP.
    critic_hidden_dims : list[int]
        Hidden layer sizes for the critic MLP.
    activation : str
        Activation function name (e.g. 'elu').
    init_noise_std : float
        Initial std for the action-noise parameter.
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        obs_dim: int = 33,
        history_len: int = 10,
        history_encoder_hidden_dims: list[int] | None = None,
        latent_dim: int = 32,
        actor_hidden_dims: list[int] | None = None,
        critic_hidden_dims: list[int] | None = None,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        use_layer_norm: bool = False,
        **kwargs,
    ):
        if kwargs:
            print(
                "[ActorCriticHistoryEncoder] Ignoring unexpected kwargs: "
                + str(list(kwargs.keys()))
            )
        super().__init__()

        if history_encoder_hidden_dims is None:
            history_encoder_hidden_dims = [256, 128]
        if actor_hidden_dims is None:
            actor_hidden_dims = [256, 128, 64]
        if critic_hidden_dims is None:
            critic_hidden_dims = [256, 128, 64]

        self.obs_dim = obs_dim
        self.history_len = history_len
        self.latent_dim = latent_dim
        self.history_flat_dim = history_len * obs_dim     # e.g. 10 * 33 = 330
        self.use_layer_norm = use_layer_norm

        expected_num_obs = obs_dim + self.history_flat_dim  # e.g. 363
        if num_actor_obs != expected_num_obs:
            print(
                f"[ActorCriticHistoryEncoder] Warning: num_actor_obs={num_actor_obs} "
                f"!= obs_dim + history_len*obs_dim = {expected_num_obs}. "
                "Proceeding with provided num_actor_obs for history splitting."
            )
            # Recalculate history_flat_dim based on actual num_actor_obs
            self.history_flat_dim = num_actor_obs - obs_dim

        # ---- history encoder: [history_len * obs_dim] -> [latent_dim] ----
        self.history_encoder = _build_mlp_str(
            self.history_flat_dim, history_encoder_hidden_dims, latent_dim, activation, use_layer_norm
        )

        actor_in = obs_dim + latent_dim    # e.g. 33 + 32 = 65
        critic_in = obs_dim + latent_dim

        # Use critic's num_obs to re-split if different from actor's
        critic_history_flat_dim = num_critic_obs - obs_dim
        if critic_history_flat_dim != self.history_flat_dim:
            self._critic_history_flat_dim = critic_history_flat_dim
            self.critic_history_encoder = _build_mlp_str(
                critic_history_flat_dim, history_encoder_hidden_dims, latent_dim, activation, use_layer_norm
            )
        else:
            self._critic_history_flat_dim = self.history_flat_dim
            self.critic_history_encoder = self.history_encoder   # shared encoder

        # ---- actor: [obs_dim + latent_dim] -> [num_actions] ----
        self.actor = _build_mlp_str(actor_in, actor_hidden_dims, num_actions, activation, use_layer_norm)

        # ---- critic: [obs_dim + latent_dim] -> [1] ----
        self.critic = _build_mlp_str(critic_in, critic_hidden_dims, 1, activation, use_layer_norm)

        # ---- action noise (trainable std) ----
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution: Normal | None = None
        Normal.set_default_validate_args = False

        # diagnostic output
        print(
            f"[ActorCriticHistoryEncoder] "
            f"obs_dim={obs_dim}, history_len={history_len}, latent_dim={latent_dim}\n"
            f"  history_encoder: {self.history_flat_dim} -> {history_encoder_hidden_dims} -> {latent_dim}\n"
            f"  actor:  {actor_in} -> {actor_hidden_dims} -> {num_actions}\n"
            f"  critic: {critic_in} -> {critic_hidden_dims} -> 1\n"
            f"  Training policy obs dim = {num_actor_obs} (current {obs_dim} + history {self.history_flat_dim})\n"
            f"  ONNX export: input0=[B,{obs_dim}], input1=[B,{history_len},{obs_dim}], output=[B,{num_actions}]"
        )

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _actor_encode(self, observations: torch.Tensor) -> torch.Tensor:
        """Split flat obs, shift history left + append current, encode -> actor input."""
        current_obs = observations[:, : self.obs_dim]                       # [B, obs_dim]
        history_flat = observations[:, self.obs_dim:]                       # [B, history_flat_dim]
        # Shift history left and append current obs at end: [oldest..prev] -> [oldest+1..prev, current]
        history = history_flat.reshape(-1, self.history_len, self.obs_dim)  # [B, history_len, obs_dim]
        new_history = torch.cat([history[:, 1:, :], current_obs.unsqueeze(1)], dim=1)  # [B, history_len, obs_dim]
        new_history_flat = new_history.reshape(-1, self.history_flat_dim)   # [B, history_flat_dim]
        latent = self.history_encoder(new_history_flat)                     # [B, latent_dim]
        return torch.cat([current_obs, latent], dim=-1)                     # [B, obs_dim + latent_dim]

    def _critic_encode(self, observations: torch.Tensor) -> torch.Tensor:
        current_obs = observations[:, : self.obs_dim]
        history_flat = observations[:, self.obs_dim:]
        critic_history_len = self._critic_history_flat_dim // self.obs_dim
        history = history_flat.reshape(-1, critic_history_len, self.obs_dim)
        new_history = torch.cat([history[:, 1:, :], current_obs.unsqueeze(1)], dim=1)
        new_history_flat = new_history.reshape(-1, self._critic_history_flat_dim)
        latent = self.critic_history_encoder(new_history_flat)
        return torch.cat([current_obs, latent], dim=-1)

    # ------------------------------------------------------------------
    # RSL-RL / co-rl compatible API
    # ------------------------------------------------------------------

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations: torch.Tensor):
        mean = self.actor(self._actor_encode(observations))
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        """Deterministic inference (used during play/evaluation)."""
        return self.actor(self._actor_encode(observations))

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.critic(self._critic_encode(critic_observations))


# ---------------------------------------------------------------------------
# ONNX export wrapper
# ---------------------------------------------------------------------------

class DdtOnnxWrapper(nn.Module):
    """Two-input ONNX wrapper compatible with ddt_ros2_control's rl_controller.

    Inputs
    ------
    current_obs : Tensor  [B, obs_dim]          e.g. [1, 33]
    history_obs : Tensor  [B, history_len, obs_dim]  e.g. [1, 10, 33]

    Output
    ------
    actions : Tensor  [B, num_actions]           e.g. [1, 8]

    Usage (export)::

        wrapper = DdtOnnxWrapper(runner.alg.actor_critic)
        wrapper.eval()
        dummy_current = torch.zeros(1, 33)
        dummy_history = torch.zeros(1, 10, 33)
        torch.onnx.export(
            wrapper,
            (dummy_current, dummy_history),
            "flat.onnx",
            input_names=["nn_input0", "nn_input1"],
            output_names=["nn_output"],
            opset_version=11,
        )
    """

    def __init__(self, actor_critic: ActorCriticHistoryEncoder):
        super().__init__()
        self.history_encoder = actor_critic.history_encoder
        self.actor = actor_critic.actor
        self.obs_dim = actor_critic.obs_dim

    def forward(
        self,
        current_obs: torch.Tensor,
        history_obs: torch.Tensor,
    ) -> torch.Tensor:
        # current_obs: [B, obs_dim]
        # history_obs: [B, history_len, obs_dim]  -- history BEFORE current obs (oldest at index 0)
        # Shift history left and append current obs at end (matches DDT obs_history_vec_ update logic)
        new_history = torch.cat([history_obs[:, 1:, :], current_obs.unsqueeze(1)], dim=1)  # [B, history_len, obs_dim]
        history_flat = new_history.reshape(current_obs.shape[0], -1)       # [B, history_len * obs_dim]
        latent = self.history_encoder(history_flat)                         # [B, latent_dim]
        actor_input = torch.cat([current_obs, latent], dim=-1)              # [B, obs_dim + latent_dim]
        return self.actor(actor_input)                                      # [B, num_actions]


def export_ddt_policy_as_onnx(
    actor_critic: ActorCriticHistoryEncoder,
    path: str,
    obs_dim: int = 33,
    history_len: int = 10,
    num_actions: int = 8,
) -> None:
    """Export trained policy as DDT-compatible two-input ONNX.

    The exported file will have:
      - input  "nn_input0":  shape [1, obs_dim]
      - input  "nn_input1":  shape [1, history_len, obs_dim]
      - output "nn_output":  shape [1, num_actions]

    In ddt_ros2_control's controllers.yaml set::

        output_name: "nn_output"
        num_obs: <obs_dim>
        num_actions: <num_actions>
        history_len: <history_len>
    """
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    wrapper = DdtOnnxWrapper(actor_critic)
    wrapper.eval()

    device = next(actor_critic.parameters()).device
    dummy_current = torch.zeros(1, obs_dim, device=device)
    dummy_history = torch.zeros(1, history_len, obs_dim, device=device)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_current, dummy_history),
            path,
            input_names=["nn_input0", "nn_input1"],
            output_names=["nn_output"],
            opset_version=11,
            # Fixed shapes: nn_input0=[1,obs_dim], nn_input1=[1,history_len,obs_dim], nn_output=[1,num_actions]
            # No dynamic_axes — ddt_ros2_control onnx_inferrer.cpp validates size via shape product
        )

    print(f"[DDT ONNX export] Saved to: {path}")
    print(f"  input0 'nn_input0': [1, {obs_dim}]")
    print(f"  input1 'nn_input1': [1, {history_len}, {obs_dim}]")
    print(f"  output 'nn_output': [1, {num_actions}]")
    print(
        "\n  [controllers.yaml] To use this policy, set:\n"
        f"    num_obs: {obs_dim}\n"
        f"    history_len: {history_len}\n"
        f"    num_actions: {num_actions}\n"
        "    output_name: \"nn_output\"\n"
        "    observations_name: [\"ang_vel\", \"gravity\", \"commands\", \"dof_pos\", \"dof_vel\", \"last_actions\"]\n"
    )
