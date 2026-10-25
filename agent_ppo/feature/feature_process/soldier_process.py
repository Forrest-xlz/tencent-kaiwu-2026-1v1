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


class SoldierProcess:
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
        self.main_camp_soldier_dict = {}
        self.enemy_camp_soldier_dict = {}
        self.main_camp_soldier_list = []
        self.enemy_camp_soldier_list = []

        self.transform_camp2_to_camp1 = self.main_camp_id == 2
        self.get_soldier_config()
        self.map_feature_to_norm = self.normalizer.parse_config(self.soldier_feature_config)
        self.view_dist = 15000
        self.one_unit_feature_num = 19
        self.unit_buff_num = 4

    def get_soldier_config(self):
        self.config = configparser.ConfigParser()
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "soldier_feature_config.ini")
        self.config.read(config_path)

        # Get normalized configuration
        # 获取归一化的配置
        self.soldier_feature_config = []
        for feature, config in self.config["feature_config"].items():
            self.soldier_feature_config.append(f"{feature}:{config}")

        # Get feature function configuration
        # 获取特征函数的配置
        self.feature_func_map = {}
        for feature, func_name in self.config["feature_functions"].items():
            if hasattr(self, func_name):
                self.feature_func_map[feature] = getattr(self, func_name)
            else:
                raise ValueError(f"Unsupported function: {func_name}")

    def process_vec_soldier(self, frame_state):
        self.generate_hero_info_list(frame_state)
        self.generate_soldier_info_list(frame_state)

        # 生成我方小兵特征：最多 4 个，不足补 0
        main_camp_soldier_vector_feature = self.generate_one_type_soldier_feature(
            self.main_camp_soldier_list, None
        )

        # 生成敌方小兵特征：最多 4 个，不足补 0
        enemy_camp_soldier_vector_feature = self.generate_one_type_soldier_feature(
            self.enemy_camp_soldier_list, None
        )
        
        return main_camp_soldier_vector_feature, enemy_camp_soldier_vector_feature

    def generate_hero_info_list(self, frame_state):
        self.main_camp_hero_dict.clear()
        self.enemy_camp_hero_dict.clear()
        for hero in frame_state["hero_states"]:
            if hero["camp"] == self.main_camp_id:
                self.main_camp_hero_dict[hero["config_id"]] = hero
                self.main_hero_info = hero
            else:
                self.enemy_camp_hero_dict[hero["config_id"]] = hero
                self.enemy_hero_info = hero

    def generate_soldier_info_list(self, frame_state):
        self.main_camp_soldier_dict.clear()
        self.enemy_camp_soldier_dict.clear()
        self.main_camp_soldier_list.clear()
        self.enemy_camp_soldier_list.clear()

        for npc in frame_state["npc_states"]:
            npc_camp = npc["camp"]
            npc_actor_type = npc["actor_type"]
            npc_subtype = npc["sub_type"]

            # 小兵: actor_type=1, sub_type=11
            if npc_actor_type != 1 or npc_subtype != 11:
                continue

            # 死亡小兵不进入最近 4 个
            if npc["hp"] <= 0:
                continue

            if npc_camp == self.main_camp_id:
                self.main_camp_soldier_dict[npc["runtime_id"]] = npc
                self.main_camp_soldier_list.append(npc)
            else:
                self.enemy_camp_soldier_dict[npc["runtime_id"]] = npc
                self.enemy_camp_soldier_list.append(npc)

        # 我方小兵多于 4 个时，选择离我方英雄最近的 4 个
        self.main_camp_soldier_list.sort(key=lambda soldier: self.get_dist_to_main_hero(soldier))
        self.main_camp_soldier_list = self.main_camp_soldier_list[:self.unit_buff_num]

        # 敌方小兵多于 4 个时，选择离我方英雄最近的 4 个
        self.enemy_camp_soldier_list.sort(key=lambda soldier: self.get_dist_to_main_hero(soldier))
        self.enemy_camp_soldier_list = self.enemy_camp_soldier_list[:self.unit_buff_num]

    def is_visible_to_main_camp(self, npc):
        camp_visible = npc.get("camp_visible", [])
        idx = self.main_camp_id - 1
        if idx < 0 or idx >= len(camp_visible):
            return False
        return bool(camp_visible[idx])

    def get_dist_to_main_hero(self, soldier):
        soldier_location = soldier["location"]
        soldier_location_x = soldier_location["x"]
        soldier_location_z = soldier_location["z"]

        main_hero_location = self.main_hero_info["location"]
        main_hero_location_x = main_hero_location["x"]
        main_hero_location_z = main_hero_location["z"]

        dist = math.sqrt((main_hero_location_x - soldier_location_x) ** 2 + (main_hero_location_z - soldier_location_z) ** 2)
        return dist

    def generate_one_type_soldier_feature(self, one_type_soldier_info, camp):
        vector_feature = []
        num_soldier_considered = 0

        def process_soldier(soldier):
            nonlocal num_soldier_considered
            # Generate each specific feature through feature_func_map
            # 通过 feature_func_map 生成每个具体特征
            for feature_name, feature_func in self.feature_func_map.items():
                value = []
                self.feature_func_map[feature_name](soldier, value)
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
            num_soldier_considered += 1

        for soldier in one_type_soldier_info:
            process_soldier(soldier)

        if num_soldier_considered < self.unit_buff_num:
            self.no_soldier_feature(vector_feature, num_soldier_considered)
        return vector_feature

    def no_soldier_feature(self, vector_feature, num_soldier_considered):
        for _ in range((self.unit_buff_num - num_soldier_considered) * self.one_unit_feature_num):
            vector_feature.append(0)

    def is_alive(self, soldier, vector_feature):
        value = 0.0
        if soldier["hp"] > 0:
            value = 1.0
        vector_feature.append(value)

    def is_visible(self, soldier, vector_feature):
        value = soldier["camp_visible"][self.main_camp_id - 1]
        vector_feature.append(value)

    def belong_to_main_camp(self, soldier, vector_feature):
        value = 0.0
        if soldier["camp"] == self.main_camp_id:
            value = 1.0
        vector_feature.append(value)

    def soldier_type(self, soldier, vector_feature):
        config_id = soldier["config_id"]

        # 近战兵
        if config_id in [6800, 6803]:
            value = 0

        # 远程兵
        elif config_id in [6801, 6804]:
            value = 1

        # 跑车 / 炮车兵
        elif config_id in [6802, 6805]:
            value = 2

        else:
            value = 0

        vector_feature.append(value)

    def get_normal_soldier_location_x(self, soldier, vector_feature):
        value = soldier["location"]["x"]
        if self.transform_camp2_to_camp1:
            value = 0 - value
        vector_feature.append(value)

    def get_normal_soldier_location_z(self, soldier, vector_feature):
        value = soldier["location"]["z"]
        if self.transform_camp2_to_camp1:
            value = 0 - value
        vector_feature.append(value)

    def self_relative_location_x(self, soldier, vector_feature):
        """
        小兵相对于自己阵营英雄的 x 相对位置。
        我方小兵 -> 相对我方英雄；
        敌方小兵 -> 相对敌方英雄。
        """
        soldier_location_x = soldier["location"]["x"]

        if soldier["camp"] == self.main_camp_id:
            hero_location_x = self.main_hero_info["location"]["x"]
        else:
            hero_location_x = self.enemy_hero_info["location"]["x"]

        x_diff = soldier_location_x - hero_location_x

        if self.transform_camp2_to_camp1 and soldier_location_x != 100000:
            x_diff = -x_diff

        value = (x_diff + 15000) / 30000.0
        vector_feature.append(value)

    def self_relative_location_z(self, soldier, vector_feature):
        """
        小兵相对于自己阵营英雄的 z 相对位置。
        我方小兵 -> 相对我方英雄；
        敌方小兵 -> 相对敌方英雄。
        """
        soldier_location_z = soldier["location"]["z"]

        if soldier["camp"] == self.main_camp_id:
            hero_location_z = self.main_hero_info["location"]["z"]
        else:
            hero_location_z = self.enemy_hero_info["location"]["z"]

        z_diff = soldier_location_z - hero_location_z

        if self.transform_camp2_to_camp1 and soldier_location_z != 100000:
            z_diff = -z_diff

        value = (z_diff + 15000) / 30000.0
        vector_feature.append(value)

    def opponent_relative_location_x(self, soldier, vector_feature):
        """
        小兵相对于敌方阵营英雄的 x 相对位置。
        我方小兵 -> 相对敌方英雄；
        敌方小兵 -> 相对我方英雄。
        """
        soldier_location_x = soldier["location"]["x"]

        if soldier["camp"] == self.main_camp_id:
            hero_location_x = self.enemy_hero_info["location"]["x"]
        else:
            hero_location_x = self.main_hero_info["location"]["x"]

        x_diff = soldier_location_x - hero_location_x

        if self.transform_camp2_to_camp1 and soldier_location_x != 100000:
            x_diff = -x_diff

        value = (x_diff + 15000) / 30000.0
        vector_feature.append(value)

    def opponent_relative_location_z(self, soldier, vector_feature):
        """
        小兵相对于敌方阵营英雄的 z 相对位置。
        我方小兵 -> 相对敌方英雄；
        敌方小兵 -> 相对我方英雄。
        """
        soldier_location_z = soldier["location"]["z"]

        if soldier["camp"] == self.main_camp_id:
            hero_location_z = self.enemy_hero_info["location"]["z"]
        else:
            hero_location_z = self.main_hero_info["location"]["z"]

        z_diff = soldier_location_z - hero_location_z

        if self.transform_camp2_to_camp1 and soldier_location_z != 100000:
            z_diff = -z_diff

        value = (z_diff + 15000) / 30000.0
        vector_feature.append(value)

    def dist_to_self_hero(self, soldier, vector_feature):
        """
        小兵到自己阵营英雄的距离。
        我方小兵 -> 到我方英雄；
        敌方小兵 -> 到敌方英雄。
        """
        soldier_location = soldier["location"]

        if soldier["camp"] == self.main_camp_id:
            hero_location = self.main_hero_info["location"]
        else:
            hero_location = self.enemy_hero_info["location"]

        dist = math.sqrt(
            (hero_location["x"] - soldier_location["x"]) ** 2 +
            (hero_location["z"] - soldier_location["z"]) ** 2
        )
        vector_feature.append(dist)

    def dist_to_opponent_hero(self, soldier, vector_feature):
        """
        小兵到敌方阵营英雄的距离。
        我方小兵 -> 到敌方英雄；
        敌方小兵 -> 到我方英雄。
        """
        soldier_location = soldier["location"]

        if soldier["camp"] == self.main_camp_id:
            hero_location = self.enemy_hero_info["location"]
        else:
            hero_location = self.main_hero_info["location"]

        dist = math.sqrt(
            (hero_location["x"] - soldier_location["x"]) ** 2 +
            (hero_location["z"] - soldier_location["z"]) ** 2
        )
        vector_feature.append(dist)

    
    def hp(self, soldier, vector_feature):
        value = soldier["hp"]
        vector_feature.append(value)

    def hp_rate(self, soldier, vector_feature):
        hp = soldier["hp"]
        max_hp = soldier["max_hp"]
        value = 0.0
        if max_hp > 0:
            value = hp / max_hp
        vector_feature.append(value)

    def max_hp(self, soldier, vector_feature):
        value = soldier["max_hp"]
        vector_feature.append(value)

    def attack_range(self, soldier, vector_feature):
        value = soldier["attack_range"]
        vector_feature.append(value)

    def kill_income(self, soldier, vector_feature):
        value = soldier["kill_income"]
        vector_feature.append(value)

    
    # 是否在攻击对方英雄
    def is_attacking_opponent_hero(self, soldier,vector_feature):
        camp = soldier['camp']
        if camp == self.main_hero_info['camp']:
            opponent_hero = self.enemy_hero_info
        else:
            opponent_hero = self.main_hero_info
        
        if soldier['attack_target'] == opponent_hero['runtime_id']:
            value = 1
        else:
            value = 0
        vector_feature.append(value)
    
    # 是否正在被对方英雄攻击
    def is_attacking_this_unit(self, soldier, vector_feaure):
        camp = soldier['camp']
        if camp == self.main_hero_info['camp']:
            opponent_hero = self.enemy_hero_info
        else:
            opponent_hero = self.main_hero_info
        
        if opponent_hero['attack_target'] == soldier['runtime_id']:
            value = 1
        else:
            value = 0
        vector_feaure.append(value)

    
    def opponent_hero_in_attack_range(self, soldier, vector_feature):
        """
        小兵是否在敌方英雄攻击范围内。

        我方小兵 -> 判断是否在敌方英雄攻击范围内；
        敌方小兵 -> 判断是否在我方英雄攻击范围内。
        """
        soldier_location = soldier["location"]

        if soldier["camp"] == self.main_camp_id:
            opponent_hero = self.enemy_hero_info
        else:
            opponent_hero = self.main_hero_info

        opponent_hero_location = opponent_hero["location"]

        dist = math.sqrt(
            (opponent_hero_location["x"] - soldier_location["x"]) ** 2 +
            (opponent_hero_location["z"] - soldier_location["z"]) ** 2
        )

        value = 0
        if dist <= opponent_hero["attack_range"]:
            value = 1

        vector_feature.append(value)
        