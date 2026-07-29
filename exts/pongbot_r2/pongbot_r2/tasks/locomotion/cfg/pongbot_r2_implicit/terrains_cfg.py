from isaaclab.terrains import (
    HfInvertedPyramidSlopedTerrainCfg,
    HfPyramidSlopedTerrainCfg,
    HfRandomUniformTerrainCfg,
    HfWaveTerrainCfg,
    MeshInvertedPyramidStairsTerrainCfg,
    MeshPlaneTerrainCfg,
    MeshPyramidStairsTerrainCfg,
    MeshRandomGridTerrainCfg,
    TerrainGeneratorCfg,
)
from pongbot_r2.tasks.locomotion.mdp.one_way_stairs import MeshOneWayStairsTerrainCfg

#############################
# Rough Terrain Configuration
#############################

BLIND_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=25, ##
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(proportion=0.2),
        # "waves": HfWaveTerrainCfg(proportion=0.0, amplitude_range=(0.02, 0.06), num_waves=10, border_width=0.25),
        "boxes": MeshRandomGridTerrainCfg(
            proportion=0.1, grid_width=0.3, grid_height_range=(0.02, 0.15), platform_width=2.0
        ),
        "random_rough": HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.1), noise_step=0.01, border_width=0.25
        ),
        "pyramid_stairs": MeshPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.02, 0.20),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.02, 0.20),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
    curriculum=True,
    difficulty_range=(0.0, 1.0),
)

IMPLICIT_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10, ##level
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(proportion=0.15),
        # "boxes_easy": MeshRandomGridTerrainCfg(
        #     proportion=0.2,
        #     grid_width=0.3,
        #     grid_height_range=(0.02, 0.08),
        #     platform_width=2.0,
        # ),
        "boxes_hard": MeshRandomGridTerrainCfg(
            proportion=0.15,
            grid_width=0.3,
            grid_height_range=(0.02, 0.2),
            platform_width=2.0,
        ),
        "random_rough": HfRandomUniformTerrainCfg(
            proportion=0.1,
            noise_range=(0.02, 0.1),
            noise_step=0.01,
            border_width=0.25,
        ),
        # "pyramid_stairs_easy": MeshPyramidStairsTerrainCfg(
        #     proportion=0.1,
        #     step_height_range=(0.02, 0.10),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        "pyramid_stairs_hard": MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.02, 0.20),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # "pyramid_stairs_inv_easy": MeshInvertedPyramidStairsTerrainCfg(
        #     proportion=0.1,
        #     step_height_range=(0.02, 0.10),
        #     step_width=0.3,
        #     platform_width=3.0,
        #     border_width=1.0,
        #     holes=False,
        # ),
        "pyramid_stairs_inv_hard": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.02, 0.20),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
        "hf_pyramid_slope_inv": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
        # "one_way_stairs": MeshOneWayStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.04, 0.20),
        #     step_width=0.3,
        #     num_steps=5,
        #     approach_width=1.0,
        #     top_platform_width=1.0,
        #     exit_width=1.0,
        #     border_width=0.5,
        #     origin_at_start=True,
        # ),
        # "one_way_wide_step": MeshOneWayStairsTerrainCfg(
        #     proportion=0.2,
        #     step_height_range=(0.04, 0.20),
        #     step_width=0.7,
        #     num_steps=3,
        #     approach_width=1.0,
        #     top_platform_width=0.7,
        #     exit_width=1.0,
        #     border_width=0.5,
        #     origin_at_start=True,
        # ),
    },
    curriculum=True,
    difficulty_range=(0.0, 1.0),
)


# Frozen terrain contract from the 2026-07-07 ``사뿐사뿐`` run.  Keep this
# separate from IMPLICIT_ROUGH_TERRAINS_CFG so later rough-terrain experiments
# cannot silently change the legacy reproduction task.
LEGACY_IMPLICIT_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=20,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(proportion=0.1, size=(8.0, 8.0)),
        "boxes_easy": MeshRandomGridTerrainCfg(
            proportion=0.1,
            size=(8.0, 8.0),
            grid_width=0.3,
            grid_height_range=(0.02, 0.08),
            platform_width=2.0,
        ),
        "boxes_hard": MeshRandomGridTerrainCfg(
            proportion=0.1,
            size=(8.0, 8.0),
            grid_width=0.3,
            grid_height_range=(0.02, 0.15),
            platform_width=2.0,
        ),
        "random_rough": HfRandomUniformTerrainCfg(
            proportion=0.1,
            size=(8.0, 8.0),
            noise_range=(0.02, 0.1),
            noise_step=0.01,
            border_width=0.25,
            slope_threshold=0.75,
        ),
        "pyramid_stairs_easy": MeshPyramidStairsTerrainCfg(
            proportion=0.1,
            size=(8.0, 8.0),
            step_height_range=(0.02, 0.10),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_hard": MeshPyramidStairsTerrainCfg(
            proportion=0.1,
            size=(8.0, 8.0),
            step_height_range=(0.02, 0.20),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv_easy": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            size=(8.0, 8.0),
            step_height_range=(0.02, 0.10),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv_hard": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            size=(8.0, 8.0),
            step_height_range=(0.02, 0.20),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
            proportion=0.1,
            size=(8.0, 8.0),
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
            slope_threshold=0.75,
        ),
        "hf_pyramid_slope_inv": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1,
            size=(8.0, 8.0),
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
            slope_threshold=0.75,
        ),
    },
    curriculum=True,
    difficulty_range=(0.0, 1.0),
)

BLIND_ROUGH_TERRAINS_PLAY_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=1,
    num_cols=5,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    sub_terrains={
        "waves": HfWaveTerrainCfg(proportion=0.1, amplitude_range=(0.01, 0.06), num_waves=10, border_width=0.25),
        "boxes": MeshRandomGridTerrainCfg(
            proportion=0.1, grid_width=0.3, grid_height_range=(0.01, 0.1), platform_width=2.0
        ),
        "random_rough": HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        ),
        "pyramid_stairs": MeshPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.1),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.0,
            step_height_range=(0.05, 0.2),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.0, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
    curriculum=True,
    difficulty_range=(1.0, 1.0),
)


BLIND_ROUGH_PCA_TERRAINS_PLAY_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=1,
    num_cols=6,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(proportion=0.1),
        "pyramid_stairs_easy": MeshPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.1),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_hard": MeshPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.2),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "random_rough": HfRandomUniformTerrainCfg(
            proportion=0.1,
            noise_range=(0.01, 0.06),
            noise_step=0.01,
            border_width=0.25,
        ),
        "hf_pyramid_slope_easy": HfPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.0, 0.2),
            platform_width=2.0,
            border_width=0.25,
        ),
        "hf_pyramid_slope_hard": HfPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
    },
    curriculum=True,
    difficulty_range=(1.0, 1.0),
)


TCP_LATENT_EVAL_TERRAINS_PLAY_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=1,
    num_cols=6,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(proportion=1.0),
        "random_rough": HfRandomUniformTerrainCfg(
            proportion=1.0,
            noise_range=(0.02, 0.10),
            noise_step=0.01,
            border_width=0.25,
        ),
        "pyramid_stairs": MeshPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.05, 0.05), #0.12),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": MeshInvertedPyramidStairsTerrainCfg(
            proportion=1.0,
            step_height_range=(0.05, 0.05), #0.12),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
            proportion=1.0,
            slope_range=(0.10, 0.2), #0.40),
            platform_width=2.0,
            border_width=0.25,
        ),
        "hf_pyramid_slope_inv": HfInvertedPyramidSlopedTerrainCfg(
            proportion=1.0,
            slope_range=(0.10, 0.2), #0.40),
            platform_width=2.0,
            border_width=0.25,
        ),
    },
    curriculum=True,
    difficulty_range=(1.0, 1.0),
)


##################################
# Hard Rough Terrain Configuration
##################################

# BLIND_HARD_ROUGH_TERRAINS_CFG = BLIND_ROUGH_TERRAINS_CFG.copy()
# BLIND_HARD_ROUGH_TERRAINS_CFG.sub_terrains["waves"].num_waves = 8
# BLIND_HARD_ROUGH_TERRAINS_CFG.sub_terrains["waves"].amplitude_range = (0.02, 0.10)
# BLIND_HARD_ROUGH_TERRAINS_CFG.sub_terrains["boxes"].grid_height_range = (0.02, 0.08)
# BLIND_HARD_ROUGH_TERRAINS_CFG.sub_terrains["random_rough"].noise_range = (0.02, 0.10)
# BLIND_HARD_ROUGH_TERRAINS_CFG.sub_terrains["random_rough"].noise_step = 0.02

# BLIND_HARD_ROUGH_TERRAINS_PLAY_CFG = BLIND_ROUGH_TERRAINS_PLAY_CFG.copy()
# BLIND_HARD_ROUGH_TERRAINS_PLAY_CFG.sub_terrains["waves"].num_waves = 8
# BLIND_HARD_ROUGH_TERRAINS_PLAY_CFG.sub_terrains["waves"].amplitude_range = (0.02, 0.10)
# BLIND_HARD_ROUGH_TERRAINS_PLAY_CFG.sub_terrains["boxes"].grid_height_range = (0.02, 0.08)
# BLIND_HARD_ROUGH_TERRAINS_PLAY_CFG.sub_terrains["random_rough"].noise_range = (0.02, 0.10)
# BLIND_HARD_ROUGH_TERRAINS_PLAY_CFG.sub_terrains["random_rough"].noise_step = 0.02


##############################
# Stairs Terrain Configuration
##############################

STAIRS_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(16.0, 16.0),
    border_width=20.0,
    num_rows=8,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    sub_terrains={
        "pyramid_stairs": MeshPyramidStairsTerrainCfg(
            proportion=0.4,
            step_height_range=(0.05, 0.20),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.4,
            step_height_range=(0.05, 0.20),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
    curriculum=True,
    difficulty_range=(0.0, 1.0),
)

STAIRS_TERRAINS_PLAY_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(16.0, 16.0),
    border_width=20.0,
    num_rows=4,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    sub_terrains={
        "pyramid_stairs": MeshPyramidStairsTerrainCfg(
            proportion=0.4,
            step_height_range=(0.05, 0.15),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.4,
            step_height_range=(0.05, 0.15),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
    curriculum=True,
    difficulty_range=(1.0, 1.0),
)


########################################
# Implicit Stair-Only Terrain Configuration
########################################

IMPLICIT_STAIR_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(16.0, 16.0),
    border_width=20.0,
    num_rows=25,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": MeshPlaneTerrainCfg(proportion=0.05),
        "pyramid_stairs": MeshPyramidStairsTerrainCfg(
            proportion=0.35,
            step_height_range=(0.04, 0.20),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.04, 0.20),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "one_way_stairs": MeshOneWayStairsTerrainCfg(
            proportion=0.35,
            step_height_range=(0.04, 0.2),
            step_width=0.3,
            num_steps=10,
            approach_width=2.0,
            top_platform_width=2.0,
            exit_width=2.0,
            border_width=1.0,
            origin_at_start=True,
        ),
        "one_way_stairs2": MeshOneWayStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.04, 0.2),
            step_width=0.5,
            num_steps=5,
            approach_width=2.0,
            top_platform_width=2.0,
            exit_width=2.0,
            border_width=1.0,
            origin_at_start=True,
        ),
        "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
            proportion=0.05, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.05, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
    curriculum=True,
    difficulty_range=(0.0, 1.0),
)

IMPLICIT_STAIR_TERRAINS_PLAY_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(16.0, 16.0),
    border_width=20.0,
    num_rows=4,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": MeshPyramidStairsTerrainCfg(
            proportion=0.35,
            step_height_range=(0.05, 0.15),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.35,
            step_height_range=(0.05, 0.15),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "one_way_stairs": MeshOneWayStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.08, 0.08),
            step_width=0.3,
            num_steps=10,
            approach_width=2.0,
            top_platform_width=2.0,
            exit_width=2.0,
            border_width=1.0,
            origin_at_start=True,
        ),
        "hf_pyramid_slope": HfPyramidSlopedTerrainCfg(
            proportion=0.05, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.05, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
    curriculum=True,
    difficulty_range=(1.0, 1.0),
)
