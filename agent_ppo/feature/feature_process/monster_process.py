#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

from agent_ppo.feature.feature_process.feature_normalizer import FeatureNormalizer
import configparser
import os
import math


class MonsterProcess:
    def __init__(self, camp):
        self.normalizer = FeatureNormalizer()
        self.main_camp = camp
        if camp == "PLAYERCAMP_1":
            self.main_camp_id = 1
        elif camp == "PLAYERCAMP_2":
            self.main_camp_id = 2
        else:
            self.main_camp_id = camp    
        self.main_camp_hero_dict = {}
        self.enemy_camp_hero_dict = {}
        self.main_camp_organ_dict = {}
        self.enemy_camp_organ_dict = {}
        self.monster_dict = {}

        self.transform_camp2_to_camp1 = self.main_camp_id == 2
        self.get_organ_config()
        self.map_feature_to_norm = self.normalizer.parse_config(self.monster_feature_config)
        self.view_dist = 15000
        self.one_unit_feature_num = 18
        self.unit_buff_num = 1

    def get_organ_config(self):
        self.config = configparser.ConfigParser()
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "monster_feature_config.ini")
        self.config.read(config_path)

        # Get normalized configuration
        # 获取归一化的配置
        self.monster_feature_config = []
        for feature, config in self.config["feature_config"].items():
            self.monster_feature_config.append(f"{feature}:{config}")

        # Get feature function configuration
        # 获取特征函数的配置
        self.feature_func_map = {}
        for feature, func_name in self.config["feature_functions"].items():
            if hasattr(self, func_name):
                self.feature_func_map[feature] = getattr(self, func_name)
            else:
                raise ValueError(f"Unsupported function: {func_name}")

    def process_vec_monster(self, frame_state):
        self.generate_organ_info_dict(frame_state)
        self.generate_hero_info_list(frame_state)
        self.generate_monster_info_dict(frame_state)

        local_vector_feature = []

        
        # 生成野怪特征
        enemy_camp_organ_vector_feature = self.generate_one_type_organ_feature(self.monster_dict, None)
        local_vector_feature.extend(enemy_camp_organ_vector_feature)

        vector_feature = local_vector_feature
        return vector_feature

    def generate_hero_info_list(self, frame_state):
        self.main_camp_hero_dict.clear()
        self.enemy_camp_hero_dict.clear()
        for hero in frame_state["hero_states"]:
            if hero["camp"] == self.main_camp:
                self.main_camp_hero_dict[hero["config_id"]] = hero
                self.main_hero_info = hero
            else:
                self.enemy_camp_hero_dict[hero["config_id"]] = hero
                self.enemy_hero_info = hero

    def generate_organ_info_dict(self, frame_state):
        self.main_camp_organ_dict.clear()
        self.enemy_camp_organ_dict.clear()

        for organ in frame_state["npc_states"]:
            organ_camp = organ["camp"]
            organ_subtype = organ["sub_type"]
            if organ_camp == self.main_camp:
                if organ_subtype == 21:
                    self.main_camp_organ_dict["tower"] = organ
            else:
                if organ_subtype == 21:
                    self.enemy_camp_organ_dict["tower"] = organ

    def generate_monster_info_dict(self, frame_state):

        for npc in frame_state["npc_states"]:
            npc_camp = npc["camp"]

            if npc_camp == 0:
                self.monster_dict['monster'] = npc
                self.monster_info = npc
            


    def generate_one_type_organ_feature(self, one_type_organ_info, camp):
        vector_feature = []
        num_monster_considered = 0

        def process_monster(monster):
            nonlocal num_monster_considered
            # Generate each specific feature through feature_func_map
            # 通过 feature_func_map 生成每个具体特征
            for feature_name, feature_func in self.feature_func_map.items():
                value = []
                self.feature_func_map[feature_name](monster, value)
                # Normalize the specific features
                # 对具体特征进行正则化
                if feature_name not in self.map_feature_to_norm:
                    assert False
                for k in value:
                    norm_func, *params = self.map_feature_to_norm[feature_name]
                    normalized_value = norm_func(k, *params)
                    if isinstance(normalized_value, list):
                        vector_feature.extend(normalized_value)
                    else:
                        vector_feature.append(normalized_value)
            num_monster_considered += 1

        if "monster" in one_type_organ_info:
            monster = one_type_organ_info["monster"]
            if monster["hp"] > 0:
                process_monster(monster)

        if num_monster_considered < self.unit_buff_num:
            self.no_monster_feature(vector_feature, num_monster_considered)
        return vector_feature

    def no_monster_feature(self, vector_feature, num_monster_considered):
        for _ in range((self.unit_buff_num - num_monster_considered) * self.one_unit_feature_num):
            vector_feature.append(0)

    def is_alive(self, monster, vector_feature):
        value = 0.0
        if monster["hp"] > 0:
            value = 1.0
        vector_feature.append(value)
    
    def is_visible(self, monster, vector_feature):

        value = monster['camp_visible'][self.main_camp - 1]
        vector_feature.append(value)
    

    def get_normal_organ_location_x(self, monster, vector_feature):
        value = monster["location"]["x"]
        if self.transform_camp2_to_camp1:
            value = 0 - value
        vector_feature.append(value)

    def get_normal_organ_location_z(self, monster, vector_feature):
        value = monster["location"]["z"]
        if self.transform_camp2_to_camp1:
            value = 0 - value
        vector_feature.append(value)

    def main_relative_location_x(self, monster, vector_feature):
        monster_location_x = monster["location"]["x"]
        location_x = self.main_hero_info["location"]["x"]
        x_diff = monster_location_x - location_x
        if self.transform_camp2_to_camp1 and monster_location_x != 100000:
            x_diff = -x_diff
        value = (x_diff + 15000) / 30000.0
        vector_feature.append(value)

    def enemy_relative_location_x(self, monster, vector_feature):
        monster_location_x = monster["location"]["x"]
        location_x = self.enemy_hero_info["location"]["x"]
        x_diff = monster_location_x - location_x
        if self.transform_camp2_to_camp1 and monster_location_x != 100000:
            x_diff = -x_diff
        value = (x_diff + 15000) / 30000.0
        vector_feature.append(value)

    def main_relative_location_z(self, monster, vector_feature):
        monster_location_z = monster["location"]["z"]
        location_z = self.main_hero_info["location"]["z"]
        z_diff = monster_location_z - location_z
        if self.transform_camp2_to_camp1 and monster_location_z != 100000:
            z_diff = -z_diff
        value = (z_diff + 15000) / 30000.0
        vector_feature.append(value)

    def enemy_relative_location_z(self, monster, vector_feature):
        monster_location_z = monster["location"]["z"]
        location_z = self.enemy_hero_info["location"]["z"]
        z_diff = monster_location_z - location_z
        if self.transform_camp2_to_camp1 and monster_location_z != 100000:
            z_diff = -z_diff
        value = (z_diff + 15000) / 30000.0
        vector_feature.append(value)

    def dist_to_main_hero(self, monster, vector_feature):

        monster_location = monster["location"]
        monster_location_x = monster_location["x"]
        monster_location_z = monster_location["z"]

        main_hero_location = self.main_hero_info["location"]
        main_hero_location_x = main_hero_location["x"]
        main_hero_location_z = main_hero_location["z"]

        dist = math.sqrt((main_hero_location_x - monster_location_x) ** 2 + (main_hero_location_z - monster_location_z) ** 2)
        value = dist
        vector_feature.append(value)
    
    def dist_to_enemy_hero(self, monster, vector_feature):
        
        monster_location = monster["location"]
        monster_location_x = monster_location["x"]
        monster_location_z = monster_location["z"]

        enemy_hero_location = self.enemy_hero_info["location"]
        enemy_hero_location_x = enemy_hero_location["x"]
        enemy_hero_location_z = enemy_hero_location["z"]

        dist = math.sqrt((enemy_hero_location_x - monster_location_x) ** 2 + (enemy_hero_location_z - monster_location_z) ** 2)
        value = dist
        vector_feature.append(value)
    

    def hp(self, monster, vector_feature):
        value = monster["hp"]
        vector_feature.append(value)
    
    def hp_rate(self, monster, vector_feature):
        hp = monster["hp"]
        max_hp = monster["max_hp"]
        value = 0.0
        if max_hp > 0:
            value = hp / max_hp
        vector_feature.append(value)
    
    def max_hp(self, monster, vector_feature):
        value = monster["max_hp"]
        vector_feature.append(value)
    
    def kill_income(self, monster, vector_feature):
        value = monster["kill_income"]
        vector_feature.append(value)
    

    def main_is_attacking_this_unit(self, monster, vector_feature):
        if self.main_hero_info['attack_target'] == monster['runtime_id']:
            value = 1
        else:
            value = 0
        
        vector_feature.append(value)

    def enemy_is_attacking_this_unit(self, monster, vector_feature):
        if self.enemy_hero_info['attack_target'] == monster['runtime_id']:
            value = 1
        else:
            value = 0
        
        vector_feature.append(value)

    
    def main_hero_in_attack_range(self, monster, vector_feature):
        """
        野怪是否在我方主英雄攻击范围内。
        判断对象：main_hero_info -> monster
        """
        monster_location = monster["location"]
        main_hero_location = self.main_hero_info["location"]

        monster_x = monster_location["x"]
        monster_z = monster_location["z"]
        main_hero_x = main_hero_location["x"]
        main_hero_z = main_hero_location["z"]

        dist = math.sqrt(
            (main_hero_x - monster_x) ** 2 +
            (main_hero_z - monster_z) ** 2
        )

        value = 0
        if dist <= self.main_hero_info["attack_range"]:
            value = 1

        vector_feature.append(value)

    def enemy_hero_in_attack_range(self, monster, vector_feature):
        """
        野怪是否在敌方英雄攻击范围内。
        判断对象：enemy_hero_info -> monster
        """
        monster_location = monster["location"]
        enemy_hero_location = self.enemy_hero_info["location"]

        monster_x = monster_location["x"]
        monster_z = monster_location["z"]
        enemy_hero_x = enemy_hero_location["x"]
        enemy_hero_z = enemy_hero_location["z"]

        dist = math.sqrt(
            (enemy_hero_x - monster_x) ** 2 +
            (enemy_hero_z - monster_z) ** 2
        )

        value = 0
        if dist <= self.enemy_hero_info["attack_range"]:
            value = 1

        vector_feature.append(value)
