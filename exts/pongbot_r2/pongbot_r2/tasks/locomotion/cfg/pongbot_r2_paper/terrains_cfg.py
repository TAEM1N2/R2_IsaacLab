"""Terrain distribution for the R2 barrier-reward paper task.

@version 0.0.1
@update 2026-07-13: Add flat-to-rough terrain rows matching the paper's reported limits.
"""

import math

from isaaclab.terrains import (
    HfInvertedPyramidSlopedTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfRandomUniformTerrainCfg,
    MeshBoxTerrainCfg,
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
    num_cols=10,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    curriculum=True,
    difficulty_range=(0.0, 1.0),
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(proportion=0.20),
        "bumpy": HfRandomUniformTerrainCfg(
            proportion=0.20,
            noise_range=(0.0, 0.06),
            noise_step=0.005,
            border_width=0.25,
        ),
        "slope_up": HfPyramidSlopedTerrainCfg(
            proportion=0.10,
            slope_range=(0.0, math.tan(math.radians(27.0))),
            platform_width=2.0,
            border_width=0.25,
        ),
        "slope_down": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.10,
            slope_range=(0.0, math.tan(math.radians(27.0))),
            platform_width=2.0,
            border_width=0.25,
        ),
        "stairs_up": MeshPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.0, 0.20),
            step_width=0.30,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "stairs_down": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.0, 0.20),
            step_width=0.30,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "steps": MeshBoxTerrainCfg(
            proportion=0.20,
            box_height_range=(0.0, 0.345),
            platform_width=2.0,
            double_box=True,
        ),
    },
)


__all__ = ["PAPER_BARRIER_ROUGH_TERRAINS_CFG"]
