# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrappers and utilities to configure an :class:`ManagerBasedRLEnv` for RSL-RL library."""

try:
    from .exporter import export_env_as_pdf, export_policy_as_jit, export_policy_as_onnx, export_srm_as_onnx
except ModuleNotFoundError as exc:
    _EXPORTER_IMPORT_ERROR = exc

    def _missing_export_dependency(*args, **kwargs):
        raise ModuleNotFoundError(
            "Optional export dependencies are missing. Install reportlab/onnx-related packages before using CO-RL exporter helpers."
        ) from _EXPORTER_IMPORT_ERROR

    export_env_as_pdf = _missing_export_dependency
    export_policy_as_jit = _missing_export_dependency
    export_policy_as_onnx = _missing_export_dependency
    export_srm_as_onnx = _missing_export_dependency

from .rl_cfg import (
    CoRlPolicyRunnerCfg,
    CoRlPpoActorCriticCfg,
    CoRlPpoAlgorithmCfg,
    CoRlSrmPpoAlgorithmCfg,
    CoRlOffPolicyCfg,
)
from .vecenv_wrapper import CoRlVecEnvWrapper