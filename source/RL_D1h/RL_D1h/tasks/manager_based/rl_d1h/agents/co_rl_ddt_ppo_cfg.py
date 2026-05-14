# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO runner configuration for the DDT-compatible D1H flat-velocity task.

Network architecture
--------------------
    Training input (flat): [current_obs_33 || history_flat_330]  = 363 dims
    History encoder:        330 → [256, 128] → latent_32
    Actor:                  (33 + 32) = 65  → [256, 128, 64] → 8
    Critic:                 (33 + 32) = 65  → [256, 128, 64] → 1

Uses co-rl's OnPolicyRunner (not CoRlVecEnvWrapper) via ddt_train.py.
Fixed obs scales in obs terms – no empirical normalisation – so that the
exported ONNX policy can be deployed directly without an external normaliser.

ONNX export
-----------
    input0 "nn_input0": [1, 33]
    input1 "nn_input1": [1, 10, 33]
    output "nn_output": [1, 8]

controllers.yaml settings to use this ONNX
-------------------------------------------
    output_name:  "nn_output"
    num_obs:       33
    num_actions:    8
    history_len:   10
    observations_name: ["ang_vel", "gravity", "commands", "dof_pos", "dof_vel", "last_actions"]
    commands_scale: [2.0, 2.0, 0.25]
    dof_pos_scale:  1.0
    dof_vel_scale:  0.05
    ang_vel_scale:  0.25
"""

from isaaclab.utils import configclass

from scripts.co_rl.core.wrapper import CoRlPolicyRunnerCfg, CoRlPpoActorCriticCfg, CoRlPpoAlgorithmCfg


@configclass
class CoRlDdtPPORunnerCfg(CoRlPolicyRunnerCfg):
    """Runner config for the DDT-compatible task.

    This config is loaded by ``ddt_train.py`` via the ``co_rl_ddt_cfg_entry_point``
    gym keyword.  It is intentionally separate from ``CoRlPPORunnerCfg`` so the
    original task is unaffected.
    """

    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 50
    experiment_name = "d1h_ddt_flat_velocity"
    experiment_description = (
        "D1H DDT-compatible flat velocity: 33-dim obs, 10-frame history encoder, 8-dim action."
    )
    empirical_normalization = False   # Obs are already fixed-scaled → no runtime normaliser needed.

    policy = CoRlPpoActorCriticCfg(
        # Use the history-encoder actor-critic defined in actor_critic_history.py
        class_name="ActorCriticHistoryEncoder",
        init_noise_std=1.0,
        # These are passed as **kwargs to ActorCriticHistoryEncoder.__init__
        # (co-rl converts the configclass to a dict and pops "class_name")
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = CoRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    # num_policy_stacks / num_critic_stacks stay at 0 (default from parent).
    # DdtHistoryVecEnvWrapper manages history externally, so no stacking needed here.
    #
    # DDT-specific hyper-parameters consumed by ddt_train.py at runtime:
    #   obs_dim      = 33   (single-frame obs)       → injected into policy kwargs
    #   history_len  = 10   (DdtHistoryVecEnvWrapper) → injected into policy kwargs
    #   num_actions  = 8   (action space)
    #   latent_dim   = 32   (history encoder output)  → injected into policy kwargs
    # These are intentionally NOT declared as configclass fields to avoid dataclass
    # inheritance ordering issues; ddt_train.py reads them as module-level constants.
