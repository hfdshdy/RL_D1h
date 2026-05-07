from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from isaaclab.markers import VisualizationMarkers
from isaaclab.sensors.ray_caster import RayCaster


@dataclass
class LiftMaskData:
    pos_w: torch.Tensor | None = None
    quat_w: torch.Tensor | None = None
    ray_hits_w: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    mask_history: torch.Tensor | None = None


class LiftMask(RayCaster):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data = LiftMaskData()
        self._height_map_w = int(round(self.cfg.pattern_cfg.size[0] / self.cfg.pattern_cfg.resolution) + 1)
        self._height_map_h = int(round(self.cfg.pattern_cfg.size[1] / self.cfg.pattern_cfg.resolution) + 1)
        self._last_zero_index = round((self._height_map_h - self.cfg.last_zero_num) / 2)

    @property
    def data(self) -> LiftMaskData:
        self._update_outdated_buffers()
        return self._data

    def reset(self, env_ids: Sequence[int] | None = None):
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        if self._data.mask is not None:
            self._data.mask[env_ids] = 0.0
        if self.cfg.history_length > 0 and self._data.mask_history is not None:
            self._data.mask_history[env_ids] = 0.0

    def _initialize_impl(self):
        super()._initialize_impl()

        self._gradient_mask = torch.ones(
            (self._height_map_h, self._height_map_w - 1), dtype=torch.float32, device=self._device
        )
        for index in range(self._last_zero_index):
            self._gradient_mask[: index + 1, (self._height_map_w - 1) - self._last_zero_index + index] = 0.0
            self._gradient_mask[self._height_map_h - (index + 1) :, (self._height_map_w - 1) - self._last_zero_index + index] = 0.0

        self._data.mask = torch.zeros(self._view.count, 1, device=self._device)
        if self.cfg.history_length > 0:
            self._data.mask_history = torch.zeros(self._view.count, self.cfg.history_length, 1, device=self._device)

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        super()._update_buffers_impl(env_ids)
        self._update_lift_mask()

        if self.cfg.history_length > 0:
            self._data.mask_history[env_ids, 1:] = self._data.mask_history[env_ids, :-1].clone()
            current_mask = self._data.mask if self._data.mask.dim() > 1 else self._data.mask.unsqueeze(-1)
            self._data.mask_history[env_ids, 0] = current_mask[env_ids]

    def _update_lift_mask(self):
        heights = self._data.ray_hits_w[..., 2]
        grid = heights.reshape(-1, self._height_map_h, self._height_map_w)
        gradients = (grid[:, :, 1:] - grid[:, :, :-1]) * self._gradient_mask
        row_max = torch.max(gradients, dim=2).values
        self._data.mask = (torch.max(row_max, dim=1).values > self.cfg.gradient_threshold).float().unsqueeze(-1)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "ray_visualizer_green") or not hasattr(self, "ray_visualizer_red"):
                self.ray_visualizer_green = VisualizationMarkers(self.cfg.green_visualizer_cfg)
                self.ray_visualizer_red = VisualizationMarkers(self.cfg.red_visualizer_cfg)
            self.ray_visualizer_green.set_visibility(True)
            self.ray_visualizer_red.set_visibility(True)
        else:
            if hasattr(self, "ray_visualizer_green"):
                self.ray_visualizer_green.set_visibility(False)
            if hasattr(self, "ray_visualizer_red"):
                self.ray_visualizer_red.set_visibility(False)

    def _debug_vis_callback(self, event):
        del event
        if self._data.mask is None or self._data.ray_hits_w is None:
            return

        indices_zero = (self._data.mask.squeeze(-1) == 0).nonzero(as_tuple=True)[0]
        indices_one = (self._data.mask.squeeze(-1) == 1).nonzero(as_tuple=True)[0]

        if indices_zero.numel() > 0:
            self.ray_visualizer_green.visualize(self._data.ray_hits_w[indices_zero, :].view(-1, 3))
        if indices_one.numel() > 0:
            self.ray_visualizer_red.visualize(self._data.ray_hits_w[indices_one, :].view(-1, 3))