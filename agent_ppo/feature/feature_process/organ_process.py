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


class OrganProcess:
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

        self.transform_camp2_to_camp1 = self.main_camp_id == 2
        self.get_organ_config()
        self.map_feature_to_norm = self.normalizer.parse_config(self.organ_feature_config)
        self.view_dist = 15000
        self.one_unit_feature_num = 18
        self.unit_buff_num = 1

    def get_organ_config(self):
        self.config = configparser.ConfigParser()
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "organ_feature_config.ini")
        self.config.read(config_path)

        # Get normalized configuration
        # 获取归一化的配置
        self.organ_feature_config = []
        for feature, config in self.config["feature_config"].items():
            self.organ_feature_config.append(f"{feature}:{config}")

        # Get feature function configuration
        # 获取特征函数的配置
        self.feature_func_map = {}
        for feature, func_name in self.config["feature_functions"].items():
            if hasattr(self, func_name):
                self.feature_func_map[feature] = getattr(self, func_name)
            else:
                raise ValueError(f"Unsupported function: {func_name}")

    def process_vec_organ(self, frame_state):
        self.generate_organ_info_dict(frame_state)
        self.generate_hero_info_list(frame_state)

        main_local_vector_feature = []
        enemy_local_vector_feature = []
        # Generate features for enemy team's towers
        # 生成敌方阵营的防御塔特征
        enemy_camp_organ_vector_feature = self.generate_one_type_organ_feature(self.enemy_camp_organ_dict, "enemy_camp")
        enemy_local_vector_feature.extend(enemy_camp_organ_vector_feature)

        # 生成我方阵营的防御塔特征
        main_camp_organ_vector_feature = self.generate_one_type_organ_feature(self.main_camp_organ_dict, "main_camp")
        main_local_vector_feature.extend(main_camp_organ_vector_feature)

        
        return main_local_vector_feature, enemy_local_vector_feature

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

    def generate_one_type_organ_feature(self, one_type_organ_info, camp):
        vector_feature = []
        num_organs_considered = 0

        def process_organ(organ):
            nonlocal num_organs_considered
            # Generate each specific feature through feature_func_map
            # 通过 feature_func_map 生成每个具体特征
            for feature_name, feature_func in self.feature_func_map.items():
                value = []
                self.feature_func_map[feature_name](organ, value)
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
            num_organs_considered += 1

        if "tower" in one_type_organ_info:
            organ = one_type_organ_info["tower"]
            process_organ(organ)

        if num_organs_considered < self.unit_buff_num:
            self.no_organ_feature(vector_feature, num_organs_considered)
        return vector_feature

    def no_organ_feature(self, vector_feature, num_organs_considered):
        for _ in range((self.unit_buff_num - num_organs_considered) * self.one_unit_feature_num):
            vector_feature.append(0)

    def get_hp_rate(self, organ, vector_feature):
        value = 0
        if organ["max_hp"] > 0:
            value = organ["hp"] / organ["max_hp"]
        vector_feature.append(value)

    def judge_in_view(self, main_hero_location, obj_location):
        if (
            (main_hero_location["x"] - obj_location["x"] >= 0 - self.view_dist)
            and (main_hero_location["x"] - obj_location["x"] <= self.view_dist)
            and (main_hero_location["z"] - obj_location["z"] >= 0 - self.view_dist)
            and (main_hero_location["z"] - obj_location["z"] <= self.view_dist)
        ):
            return True
        return False


    def is_alive(self, organ, vector_feature):
        value = 0.0
        if organ["hp"] > 0:
            value = 1.0
        vector_feature.append(value)

    def belong_to_main_camp(self, organ, vector_feature):
        value = 0.0
        if organ["camp"] == self.main_hero_info["camp"]:
            value = 1.0
        vector_feature.append(value)

    def get_normal_organ_location_x(self, organ, vector_feature):
        value = organ["location"]["x"]
        if self.transform_camp2_to_camp1:
            value = 0 - value
        vector_feature.append(value)

    def get_normal_organ_location_z(self, organ, vector_feature):
        value = organ["location"]["z"]
        if self.transform_camp2_to_camp1:
            value = 0 - value
        vector_feature.append(value)

    def self_relative_location_x(self, organ, vector_feature):
        """
        防御塔相对于自己阵营英雄的 x 相对位置。
        我方塔 -> 相对我方英雄；
        敌方塔 -> 相对敌方英雄。
        """
        organ_location_x = organ["location"]["x"]

        if organ["camp"] == self.main_hero_info["camp"]:
            hero_location_x = self.main_hero_info["location"]["x"]
        else:
            hero_location_x = self.enemy_hero_info["location"]["x"]

        x_diff = organ_location_x - hero_location_x

        if self.transform_camp2_to_camp1 and organ_location_x != 100000:
            x_diff = -x_diff

        value = (x_diff + 15000) / 30000.0
        vector_feature.append(value)

    def self_relative_location_z(self, organ, vector_feature):
        """
        防御塔相对于自己阵营英雄的 z 相对位置。
        我方塔 -> 相对我方英雄；
        敌方塔 -> 相对敌方英雄。
        """
        organ_location_z = organ["location"]["z"]

        if organ["camp"] == self.main_hero_info["camp"]:
            hero_location_z = self.main_hero_info["location"]["z"]
        else:
            hero_location_z = self.enemy_hero_info["location"]["z"]

        z_diff = organ_location_z - hero_location_z

        if self.transform_camp2_to_camp1 and organ_location_z != 100000:
            z_diff = -z_diff

        value = (z_diff + 15000) / 30000.0
        vector_feature.append(value)

    def opponent_relative_location_x(self, organ, vector_feature):
        """
        防御塔相对于敌对阵营英雄的 x 相对位置。
        我方塔 -> 相对敌方英雄；
        敌方塔 -> 相对我方英雄。
        """
        organ_location_x = organ["location"]["x"]

        if organ["camp"] == self.main_hero_info["camp"]:
            hero_location_x = self.enemy_hero_info["location"]["x"]
        else:
            hero_location_x = self.main_hero_info["location"]["x"]

        x_diff = organ_location_x - hero_location_x

        if self.transform_camp2_to_camp1 and organ_location_x != 100000:
            x_diff = -x_diff

        value = (x_diff + 15000) / 30000.0
        vector_feature.append(value)

    def opponent_relative_location_z(self, organ, vector_feature):
        """
        防御塔相对于敌对阵营英雄的 z 相对位置。
        我方塔 -> 相对敌方英雄；
        敌方塔 -> 相对我方英雄。
        """
        organ_location_z = organ["location"]["z"]

        if organ["camp"] == self.main_hero_info["camp"]:
            hero_location_z = self.enemy_hero_info["location"]["z"]
        else:
            hero_location_z = self.main_hero_info["location"]["z"]

        z_diff = organ_location_z - hero_location_z

        if self.transform_camp2_to_camp1 and organ_location_z != 100000:
            z_diff = -z_diff

        value = (z_diff + 15000) / 30000.0
        vector_feature.append(value)

    def hp(self, organ, vector_feature):
        value = organ["hp"]
        vector_feature.append(value)

    def hp_max(self, organ, vector_feature):
        value = organ["max_hp"]
        vector_feature.append(value)

    def atk(self, organ, vector_feature):
        value = organ["phy_atk"]
        vector_feature.append(value)

    def attack_range(self, organ, vector_feature):
        value = organ["attack_range"]
        vector_feature.append(value)

    # 相对于自己camp的英雄的距离
    def dist_to_self_hero(self, organ, vector_feature): 
        camp = organ["camp"]
        if camp == self.main_hero_info["camp"]:
            hero_location = self.main_hero_info["location"]
        else:
            hero_location = self.enemy_hero_info["location"]
        organ_location = organ["location"]
        
        hero_location_x = hero_location["x"]
        hero_location_z = hero_location["z"]
        organ_location_x = organ_location["x"]
        organ_location_z = organ_location["z"]
        dist = math.sqrt((hero_location_x - organ_location_x) ** 2 + (hero_location_z - organ_location_z) ** 2)
        value = dist
        vector_feature.append(value)

    # 相对于非自己camp的英雄的距离
    def dist_to_opponent_hero(self, organ, vector_feature):
        camp = organ["camp"]
        if camp == self.main_hero_info["camp"]:
            hero_location = self.enemy_hero_info["location"]
        else:
            hero_location = self.main_hero_info["location"]
        organ_location = organ["location"]
        
        hero_location_x = hero_location["x"]
        hero_location_z = hero_location["z"]
        organ_location_x = organ_location["x"]
        organ_location_z = organ_location["z"]
        dist = math.sqrt((hero_location_x - organ_location_x) ** 2 + (hero_location_z - organ_location_z) ** 2)
        value = dist
        vector_feature.append(value)

    
    # 塔是否在攻击对方英雄
    def is_attacking_opponent_hero(self, organ, vector_feature):
        camp = organ['camp']
        if camp == self.main_hero_info['camp']:
            opponent_hero = self.enemy_hero_info
        else:
            opponent_hero = self.main_hero_info
        
        if organ['attack_target'] == opponent_hero['runtime_id']:
            value = 1
        else:
            value = 0
        vector_feature.append(value)
    
    # 塔是否被对方英雄攻击
    def is_attacking_this_unit(self, organ, vector_feature):
        camp = organ['camp']
        if camp == self.main_hero_info['camp']:
            opponent_hero = self.enemy_hero_info
        else:
            opponent_hero = self.main_hero_info
        
        if opponent_hero['attack_target'] == organ['runtime_id']:
            value = 1
        else:
            value = 0
        vector_feature.append(value)

    # 塔是否在对方英雄普通攻击范围内
    def opponent_hero_in_attack_range(self, organ, vector_feature):
        """
        当前防御塔是否在对方英雄的普通攻击范围内。

        我方塔 -> 判断是否在敌方英雄攻击范围内；
        敌方塔 -> 判断是否在我方英雄攻击范围内。
        """
        if organ["camp"] == self.main_hero_info["camp"]:
            opponent_hero = self.enemy_hero_info
        else:
            opponent_hero = self.main_hero_info

        organ_location = organ["location"]
        opponent_hero_location = opponent_hero["location"]

        organ_x = organ_location["x"]
        organ_z = organ_location["z"]
        opponent_x = opponent_hero_location["x"]
        opponent_z = opponent_hero_location["z"]

        dist = math.sqrt(
            (opponent_x - organ_x) ** 2 +
            (opponent_z - organ_z) ** 2
        )

        value = 0
        if dist <= opponent_hero["attack_range"]:
            value = 1

        vector_feature.append(value)

    
    
