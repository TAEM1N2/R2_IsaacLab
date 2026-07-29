"""Terrain generators used by the basic Flat/Rough task family.

@version 0.0.1
@update 2026-07-29: Isolate the rough train/play terrain definitions.
"""

from isaaclab.terrains import (
    HfInvertedPyramidSlopedTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfRandomUniformTerrainCfg,
    HfWaveTerrainCfg,
    MeshPlaneTerrainCfg,
    MeshRandomGridTerrainCfg,
    TerrainGeneratorCfg,
)


BLIND_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=18,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(proportion=0.3),
        "waves": HfWaveTerrainCfg(
            proportion=0.3,
            amplitude_range=(0.01, 0.06),
            num_waves=10,
            border_width=0.25,
        ),
        "boxes": MeshRandomGridTerrainCfg(
            proportion=0.3,
            grid_width=0.3,
            grid_height_range=(0.01, 0.2),
            platform_width=2.0,
        ),
        "random_rough": HfRandomUniformTerrainCfg(
            proportion=0.3,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            border_width=0.25,
        ),
        "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
            proportion=0.3,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
        "hf_pyramid_slope_inv": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.3,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
    },
    curriculum=True,
    difficulty_range=(0.0, 1.0),
)

BLIND_ROUGH_TERRAINS_PLAY_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=4,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    sub_terrains={
        "waves": HfWaveTerrainCfg(
            proportion=0.3,
            amplitude_range=(0.01, 0.06),
            num_waves=10,
            border_width=0.25,
        ),
        "boxes": MeshRandomGridTerrainCfg(
            proportion=0.2,
            grid_width=0.3,
            grid_height_range=(0.01, 0.04),
            platform_width=2.0,
        ),
        "random_rough": HfRandomUniformTerrainCfg(
            proportion=0.3,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            border_width=0.25,
        ),
        "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.0, 0.2),
            platform_width=2.0,
            border_width=0.25,
        ),
        "hf_pyramid_slope_inv": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.0, 0.2),
            platform_width=2.0,
            border_width=0.25,
        ),
    },
    curriculum=False,
    difficulty_range=(1.0, 1.0),
)
