#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from agent_ppo.feature.feature_process.hero_process import HeroProcess
from agent_ppo.feature.feature_process.monster_process import MonsterProcess
from agent_ppo.feature.feature_process.organ_process import OrganProcess
from agent_ppo.feature.feature_process.soldier_process import SoldierProcess
from agent_ppo.feature.feature_process.game_process import GameProcess
from agent_ppo.feature.feature_process.private_hero_luban import LubanProcess
from agent_ppo.feature.feature_process.private_hero_direnjie import DirenjieProcess
from agent_ppo.feature.feature_process.main_hero_process import MainHeroProcess


class FeatureProcess:
    def __init__(self, camp):
        self.camp = camp
        self.hero_process = HeroProcess(camp)
        self.organ_process = OrganProcess(camp)
        self.monster_process = MonsterProcess(camp)
        self.soldier_process = SoldierProcess(camp)
        self.game_process = GameProcess(camp)
        self.luban = LubanProcess(camp)
        self.direnjie = DirenjieProcess(camp)
        self.main_hero = MainHeroProcess(camp)

    def reset(self, camp):
        self.camp = camp
        self.hero_process = HeroProcess(camp)
        self.organ_process = OrganProcess(camp)
        self.monster_process = MonsterProcess(camp)
        self.soldier_process = SoldierProcess(camp)
        self.game_process = GameProcess(camp)
        self.luban = LubanProcess(camp)
        self.direnjie = DirenjieProcess(camp)
        self.main_hero = MainHeroProcess(camp)

    def process_main_hero_feature(self, frame_state):
        main_hero_feature = self.main_hero.process_vec_main_hero(frame_state)

        return main_hero_feature

    def process_luban_feature(self, frame_state):
        main_luban_feature, enemy_luban_feature = self.luban.process_vec_luban(frame_state)

        return main_luban_feature, enemy_luban_feature

    def process_direnjie_feature(self, frame_state):
        main_direnjie_feature, enemy_direnjie_feature = self.direnjie.process_vec_direnjie(frame_state)

        return main_direnjie_feature, enemy_direnjie_feature

    def process_game_feature(self, frame_state):
        game_feature = self.game_process.process_vec_game(frame_state)

        return game_feature
    
    def process_monster_feature(self, frame_state):

        monster_feature = self.monster_process.process_vec_monster(frame_state)

        return monster_feature
    
    def process_soldier_feature(self, frame_state):
        
        main_camp_soldier_vector_feature, enemy_camp_soldier_vector_feature = self.soldier_process.process_vec_soldier(frame_state)

        return main_camp_soldier_vector_feature, enemy_camp_soldier_vector_feature

    def process_organ_feature(self, frame_state):
        
        main_local_vector_feature, enemy_local_vector_feature = self.organ_process.process_vec_organ(frame_state)

        return main_local_vector_feature, enemy_local_vector_feature

    def process_hero_feature(self, frame_state):

        main_camp_hero_vector_feature, enemy_camp_hero_vector_feature = self.hero_process.process_vec_hero(frame_state)
        
        return main_camp_hero_vector_feature, enemy_camp_hero_vector_feature

    def process_feature(self, observation):
        frame_state = observation["frame_state"]

        main_camp_hero_vector_feature, enemy_camp_hero_vector_feature = self.process_hero_feature(frame_state)

        main_organ_vector_feature, enemy_organ_vector_feature = self.process_organ_feature(frame_state)

        monster_vector_feature = self.process_monster_feature(frame_state)

        main_camp_soldier_vector_feature, enemy_camp_soldier_vector_feature = self.process_soldier_feature(frame_state)

        game_vector_feature = self.process_game_feature(frame_state)

        main_luban_vector_feature, enemy_luban_vector_feature = self.process_luban_feature(frame_state)

        main_direnjie_vector_feature, enemy_direnjie_vector_feature = self.process_direnjie_feature(frame_state)

        main_hero_vector_feature = self.process_main_hero_feature(frame_state)

        # 特征拼接（分行写法）
        feature = main_camp_hero_vector_feature + \
                enemy_camp_hero_vector_feature + \
                main_organ_vector_feature + \
                enemy_organ_vector_feature + \
                monster_vector_feature + \
                main_camp_soldier_vector_feature + \
                enemy_camp_soldier_vector_feature + \
                game_vector_feature + \
                main_luban_vector_feature + \
                enemy_luban_vector_feature + \
                main_direnjie_vector_feature + \
                enemy_direnjie_vector_feature + \
                main_hero_vector_feature
        
        return feature
