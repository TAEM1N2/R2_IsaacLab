from __future__ import annotations

from dataclasses import MISSING

import numpy as np
import trimesh
from isaaclab.terrains.terrain_generator_cfg import SubTerrainBaseCfg
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.utils import configclass


def _box_segment(x0: float, x1: float, y0: float, y1: float, height: float) -> trimesh.Trimesh | None:
    if height <= 0.0 or x1 <= x0 or y1 <= y0:
        return None

    dims = (x1 - x0, y1 - y0, height)
    pos = ((x0 + x1) * 0.5, (y0 + y1) * 0.5, height * 0.5)
    return trimesh.creation.box(dims, trimesh.transformations.translation_matrix(pos))


def one_way_stairs_terrain(
    difficulty: float, cfg: MeshOneWayStairsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate one-way stairs along +x using the standard terrain curriculum difficulty."""
    if cfg.num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {cfg.num_steps}.")
    if cfg.step_width <= 0.0:
        raise ValueError(f"step_width must be positive, got {cfg.step_width}.")
    if cfg.border_width < 0.0:
        raise ValueError(f"border_width must be non-negative, got {cfg.border_width}.")

    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
    if step_height <= 0.0:
        raise ValueError(f"Resolved step_height must be positive, got {step_height}.")

    usable_x = cfg.size[0] - 2.0 * cfg.border_width
    usable_y = cfg.size[1] - 2.0 * cfg.border_width
    required_x = (
        cfg.approach_width + 2.0 * cfg.num_steps * cfg.step_width + cfg.top_platform_width + cfg.exit_width
    )
    if usable_x <= 0.0 or usable_y <= 0.0:
        raise ValueError(f"border_width={cfg.border_width} is too large for terrain size={cfg.size}.")
    if required_x > usable_x:
        raise ValueError(
            "One-way stairs do not fit inside the terrain tile: "
            f"required_x={required_x:.3f}, usable_x={usable_x:.3f}, size={cfg.size}, "
            f"border_width={cfg.border_width}, num_steps={cfg.num_steps}, step_width={cfg.step_width}, "
            f"approach_width={cfg.approach_width}, top_platform_width={cfg.top_platform_width}, "
            f"exit_width={cfg.exit_width}."
        )

    meshes = [make_plane(cfg.size, 0.0, center_zero=False)]
    x = cfg.border_width + cfg.approach_width
    y0 = cfg.border_width
    y1 = cfg.size[1] - cfg.border_width

    for step_idx in range(1, cfg.num_steps + 1):
        mesh = _box_segment(x, x + cfg.step_width, y0, y1, step_idx * step_height)
        if mesh is not None:
            meshes.append(mesh)
        x += cfg.step_width

    top_height = cfg.num_steps * step_height
    mesh = _box_segment(x, x + cfg.top_platform_width, y0, y1, top_height)
    if mesh is not None:
        meshes.append(mesh)
    x += cfg.top_platform_width

    for step_idx in range(cfg.num_steps - 1, -1, -1):
        mesh = _box_segment(x, x + cfg.step_width, y0, y1, step_idx * step_height)
        if mesh is not None:
            meshes.append(mesh)
        x += cfg.step_width

    if cfg.origin_at_start:
        origin_x = cfg.border_width + 0.5 * cfg.approach_width
    else:
        origin_x = 0.5 * cfg.size[0]
    origin = np.array([origin_x, 0.5 * cfg.size[1], 0.0])
    return meshes, origin


@configclass
class MeshOneWayStairsTerrainCfg(SubTerrainBaseCfg):
    """One-direction stair course: flat, up stairs, top platform, down stairs, flat."""

    function = one_way_stairs_terrain

    border_width: float = 0.0
    """Flat border left around the stair course."""

    step_height_range: tuple[float, float] = MISSING
    """Minimum and maximum height of each stair step."""

    step_width: float = MISSING
    """Length of each stair step along x."""

    num_steps: int = MISSING
    """Number of upward steps. The downward section uses the same count."""

    approach_width: float = 2.0
    """Flat approach length before the upward stairs."""

    top_platform_width: float = 2.0
    """Flat platform length at the top of the stairs."""

    exit_width: float = 2.0
    """Minimum flat length after descending back to ground level."""

    origin_at_start: bool = True
    """If True, spawn/reference origin is placed on the approach flat before the upward stairs."""
