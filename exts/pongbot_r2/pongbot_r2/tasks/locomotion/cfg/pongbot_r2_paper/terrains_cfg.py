"""Terrain distribution for the R2 barrier-reward paper task.

@version 0.0.3
@update 2026-07-13: Use 16 terrain columns so every configured rough-trot proportion is represented exactly.
@update 2026-07-13: Restrict the paper task to rough-trot terrain without high-step boxes.
"""

import math

from isaaclab.terrains import (
    HfInvertedPyramidSlopedTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfRandomUniformTerrainCfg,
    MeshInvertedPyramidStairsTerrainCfg,
    MeshPlaneTerrainCfg,
    MeshPyramidStairsTerrainCfg,
    TerrainGeneratorCfg,
)


PAPER_BARRIER_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=11,
    num_cols=16,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    curriculum=True,
    difficulty_range=(0.0, 1.0),
    sub_terrains={
        # The paper reports rough-trot training on flat, 6-cm bumps,
        # 27-degree slopes, 20-cm stairs, and 34.5-cm discrete steps.
        # This task intentionally stops at the requested low-stair scope;
        # the discrete-step terrain is excluded rather than reinterpreted.
        "flat": MeshPlaneTerrainCfg(proportion=0.25),
        "bumpy": HfRandomUniformTerrainCfg(
            proportion=0.25,
            noise_range=(0.0, 0.06),
            noise_step=0.005,
            border_width=0.25,
        ),
        "slope_up": HfPyramidSlopedTerrainCfg(
            proportion=0.125,
            slope_range=(0.0, math.tan(math.radians(27.0))),
            platform_width=2.0,
            border_width=0.25,
        ),
        "slope_down": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.125,
            slope_range=(0.0, math.tan(math.radians(27.0))),
            platform_width=2.0,
            border_width=0.25,
        ),
        "stairs_up": MeshPyramidStairsTerrainCfg(
            proportion=0.125,
            step_height_range=(0.0, 0.20),
            step_width=0.30,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.125,
            step_height_range=(0.0, 0.20),
            step_width=0.30,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
    },
)


__all__ = ["PAPER_BARRIER_ROUGH_TERRAINS_CFG"]
