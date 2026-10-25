#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


class GameConfig:
    # Set the weight of each reward item and use it in reward_manager
    # 设置各个回报项的权重，在reward_manager中使用
    # 训练前期ep小一些增加技能释放率，kill和death小一些提升对战实力，防御塔给低一些
    REWARD_WEIGHT_DICT = {
        "forward": 0.01,
        "money": 0.008,
        "exp": 0.008,
        "hp_point": 2.0,    
        "ep_rate": 0.55, # 0.75
        "kill": -0.5,   # -0.6
        "death": -0.8,  # -1.0
        "tower_hp_point": 10.0,  # 5.0
        "last_hit": 0.5,
        "monster_kill": 0.25,    # 击杀野怪的奖励
        "monster_first_hit" : 0.0,
        "wave_hit_enemy_hero": 0.1, # 小兵波次进攻
        "direnjie_ult_kill_after_hit": 0.25,
        "direnjie_ult_miss": 0.05,
        "rage_kill_after_use": 0.25,
        "rage_no_kill_after_use": 0.0,
        "rage_use_penalty": 0.2,
        "grass_ambush_kill": 0.25,
        "recover_use_penalty": 0.1,
        "direnjie_ult_hit_under_tower": 0.15,
        "direnjie_ult_tower_poke": 0.0,
        "direnjie_ult_level4_kill": 0.0,
    }
    WIN_REWARD = 2.5
    LOSE_REWARD = -2.5
    # Time decay factor, used in reward_manager
    # 时间衰减因子，在reward_manager中使用
    # 前后期同样的奖励真实价值不同，前期一些经济领先可以导致后期很大的经济优势
    TIME_SCALE_ARG = 8000

    # 是否启用 TIME_SCALE_ARG 调度
    TIME_SCALE_SCHEDULE_ENABLE = False

    # 从第几局开始调度
    TIME_SCALE_SCHEDULE_START_EPISODE = 1

    # 到第几局达到目标值
    TIME_SCALE_SCHEDULE_END_EPISODE = 200

    # 初始 time_scale_arg
    # 小一些：前期奖励保留更多，后期奖励衰减更强
    TIME_SCALE_SCHEDULE_START_VALUE = 4500

    # 目标 time_scale_arg
    # 大一些：后期奖励衰减变弱，更关注中后期
    TIME_SCALE_SCHEDULE_END_VALUE = 5500


    # 奖励调度
    # 是否启用奖励调度
    REWARD_SCHEDULE_ENABLE = False

    # 从第几局开始调度
    REWARD_SCHEDULE_START_EPISODE = 70

    # 到第几局达到目标值
    REWARD_SCHEDULE_END_EPISODE = 400

    # 每隔多少局打印一次当前奖励系数
    REWARD_SCHEDULE_LOG_INTERVAL = 100

    # 只写需要调度的奖励项
    # start 是初始值，end 是目标值
    REWARD_SCHEDULE_DICT = {
        
        

    }

    # Model save interval configuration, used in workflow
    # 模型保存间隔配置，在workflow中使用
    MODEL_SAVE_INTERVAL = 600

    DIRENJIE_ULT_TOWER_POKE_SAFE_MARGIN = 1000


    


# Dimension configuration, used when building the model
# 维度配置，构建模型时使用
class DimConfig:
    PUBLIC_MAIN_HERO_DIM = 81
    PUBLIC_ENEMY_HERO_DIM = 81
    MAIN_ORGAN_DIM = 18
    ENEMY_ORGAN_DIM = 18
    MONSTER_DIM = 18
    MAIN_SOLDIER_DIM = 19*4
    ENEMY_SOLDIER_DIM = 19*4
    GAME_DIM = 9
    MAIN_LUBAN_DIM = 15
    ENEMY_LUBAN_DIM = 15
    MAIN_DIRENJIE_DIM = 6
    ENEMY_DIRENJIE_DIM = 6
    MAIN_HERO_DIM = 13


    SUM_DIM = (
        PUBLIC_MAIN_HERO_DIM
        + PUBLIC_ENEMY_HERO_DIM
        + MAIN_ORGAN_DIM
        + ENEMY_ORGAN_DIM
        + MONSTER_DIM
        + MAIN_SOLDIER_DIM
        + ENEMY_SOLDIER_DIM
        + GAME_DIM
        + MAIN_LUBAN_DIM
        + ENEMY_LUBAN_DIM
        + MAIN_DIRENJIE_DIM
        + ENEMY_DIRENJIE_DIM
        + MAIN_HERO_DIM
    )

    DIM_OF_FEATURE = [SUM_DIM]


# Configuration related to model and algorithms used
# 模型和算法使用的相关配置
class Config:
    NETWORK_NAME = "network"
    LSTM_TIME_STEPS = 16
    LSTM_UNIT_SIZE = 512

    FEATURE_DIM = DimConfig.SUM_DIM
    LEGAL_ACTION_DIM = 85

    DATA_SPLIT_SHAPE = [
        FEATURE_DIM + LEGAL_ACTION_DIM,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        12,
        16,
        16,
        16,
        16,
        9,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        LSTM_UNIT_SIZE,
        LSTM_UNIT_SIZE,
    ]

    SERI_VEC_SPLIT_SHAPE = [(FEATURE_DIM,), (LEGAL_ACTION_DIM,)]

    INIT_LEARNING_RATE_START = 1e-4
    TARGET_LR = 1e-5
    TARGET_STEP = 6000
    BETA_START = 0.0025
    LOG_EPSILON = 1e-6

    # Entropy coefficient schedule
    # 熵系数线性衰减
    ENTROPY_BETA_DECAY_ENABLE = True
    BETA_START = 0.025
    BETA_END = 0.0025
    BETA_DECAY_STEPS = 6000

    LABEL_SIZE_LIST = [12, 16, 16, 16, 16, 9]

    IS_REINFORCE_TASK_LIST = [
        True,
        True,
        True,
        True,
        True,
        True,
    ]

    CLIP_PARAM = 0.2
    MIN_POLICY = 0.00001
    TARGET_EMBED_DIM = 32

    data_shapes = [
        [(FEATURE_DIM + LEGAL_ACTION_DIM) * LSTM_TIME_STEPS],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [192],
        [256],
        [256],
        [256],
        [256],
        [144],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [16],
        [512],
        [512],
    ]

    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()
    LEGAL_ACTION_SIZE_LIST[-1] = LEGAL_ACTION_SIZE_LIST[-1] * LEGAL_ACTION_SIZE_LIST[0]

    GAMMA = 0.997
    LAMDA = 0.95

    # gamma 调度
    USE_GAMMA_SCHEDULE = False

    GAMMA_START = 0.995
    GAMMA_END = 0.997

    # 多少局从 GAMMA_START 线性增长到 GAMMA_END
    GAMMA_SCHEDULE_EPISODES = 1000

    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5

    SAMPLE_DIM = sum(DATA_SPLIT_SHAPE[:-2]) * LSTM_TIME_STEPS + sum(DATA_SPLIT_SHAPE[-2:])
