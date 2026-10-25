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


class MainHeroProcess:
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
        self.get_main_hero_config()
        self.map_feature_to_norm = self.normalizer.parse_config(self.main_hero_feature_config)
        self.view_dist = 15000
        self.one_unit_feature_num = 13
        self.unit_buff_num = 1

    def get_main_hero_config(self):
        self.config = configparser.ConfigParser()
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "main_hero_feature_config.ini")
        self.config.read(config_path)

        # Get normalized configuration
        # 获取归一化的配置
        self.main_hero_feature_config = []
        for feature, config in self.config["feature_config"].items():
            self.main_hero_feature_config.append(f"{feature}:{config}")

        # Get feature function configuration
        # 获取特征函数的配置
        self.feature_func_map = {}
        for feature, func_name in self.config["feature_functions"].items():
            if hasattr(self, func_name):
                self.feature_func_map[feature] = getattr(self, func_name)
            else:
                raise ValueError(f"Unsupported function: {func_name}")

    def process_vec_main_hero(self, frame_state):

        self.generate_hero_info_list(frame_state)

        main_hero_vector_feature = self.generate_one_type_main_hero_feature(
            frame_state, None
        )


        return main_hero_vector_feature
    
    def generate_one_type_main_hero_feature(self, frame_state, camp):
        vector_feature = []
        num_main_heros_considered = 0

        def process_main_hero(main_hero):
            nonlocal num_main_heros_considered
            # Generate each specific feature through feature_func_map
            # 通过 feature_func_map 生成每个具体特征
            for feature_name, feature_func in self.feature_func_map.items():
                value = []
                self.feature_func_map[feature_name](main_hero, value)
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
            num_main_heros_considered += 1

       
        process_main_hero(frame_state)

        if num_main_heros_considered < self.unit_buff_num:
            self.no_main_hero_feature(vector_feature, num_main_heros_considered)
        return vector_feature

    def no_main_hero_feature(self, vector_feature, num_main_heros_considered):
        for _ in range((self.unit_buff_num - num_main_heros_considered) * self.one_unit_feature_num):
            vector_feature.append(0)

    def generate_hero_info_list(self, frame_state):
        self.main_camp_hero_dict.clear()
        self.enemy_camp_hero_dict.clear()

        self.main_hero_info = None
        self.enemy_hero_info = None

        for hero in frame_state["hero_states"]:
            if hero["camp"] == self.main_camp:
                self.main_camp_hero_dict[hero["config_id"]] = hero
                self.main_hero_info = hero
            else:
                self.enemy_camp_hero_dict[hero["config_id"]] = hero
                self.enemy_hero_info = hero

    def _get_nearest_enemy_bullet_relative_info(self, frame_state):
        """
        获取最近敌方子弹相对我方英雄的位置。
        没有敌方子弹时，默认返回 0, 0, 0。
        """
        main_hero = self.main_hero_info
        hero_x = main_hero["location"]["x"]
        hero_z = main_hero["location"]["z"]

        nearest_x_diff = 0.0
        nearest_z_diff = 0.0
        nearest_dist = 0.0
        has_enemy_bullet = False

        for bullet in frame_state.get("bullets", []):
            bullet_camp = bullet.get("camp")

            # 只看敌方子弹
            if bullet_camp == self.main_camp_id:
                continue

            # camp=0 的子弹先跳过，避免中立/异常子弹干扰
            if bullet_camp == 0:
                continue

            bullet_location = bullet.get("location", {})
            bullet_x = bullet_location.get("x", 0)
            bullet_z = bullet_location.get("z", 0)

            x_diff = bullet_x - hero_x
            z_diff = bullet_z - hero_z

            if self.transform_camp2_to_camp1:
                x_diff = -x_diff
                z_diff = -z_diff

            dist = math.sqrt(x_diff * x_diff + z_diff * z_diff)

            if not has_enemy_bullet or dist < nearest_dist:
                nearest_dist = dist
                nearest_x_diff = x_diff
                nearest_z_diff = z_diff
                has_enemy_bullet = True

        if not has_enemy_bullet:
            return 0.0, 0.0, 0.0

        nearest_x_diff = max(-15000.0, min(15000.0, nearest_x_diff))
        nearest_z_diff = max(-15000.0, min(15000.0, nearest_z_diff))
        nearest_dist = max(0.0, min(15000.0, nearest_dist))

        rel_x_norm = (nearest_x_diff + 15000.0) / 30000.0
        rel_z_norm = (nearest_z_diff + 15000.0) / 30000.0
        dist_norm = nearest_dist / 15000.0

        return rel_x_norm, rel_z_norm, dist_norm


    def nearest_enemy_bullet_relative_x(self, frame_state, vector_feature):
        rel_x_norm, _, _ = self._get_nearest_enemy_bullet_relative_info(frame_state)
        vector_feature.append(rel_x_norm)


    def nearest_enemy_bullet_relative_z(self, frame_state, vector_feature):
        _, rel_z_norm, _ = self._get_nearest_enemy_bullet_relative_info(frame_state)
        vector_feature.append(rel_z_norm)


    def nearest_enemy_bullet_distance(self, frame_state, vector_feature):
        _, _, dist_norm = self._get_nearest_enemy_bullet_relative_info(frame_state)
        vector_feature.append(dist_norm)

    
    def _get_main_enemy_dist(self):
        """
        计算我方英雄和敌方英雄之间的 xz 平面距离。
        """
        main_hero = getattr(self, "main_hero_info", None)
        enemy_hero = getattr(self, "enemy_hero_info", None)

        if main_hero is None or enemy_hero is None:
            return None

        main_loc = main_hero.get("location", {})
        enemy_loc = enemy_hero.get("location", {})

        dx = enemy_loc.get("x", 0) - main_loc.get("x", 0)
        dz = enemy_loc.get("z", 0) - main_loc.get("z", 0)

        return math.sqrt(dx * dx + dz * dz)

    def _get_main_skill_range(self, skill_idx):
        """
        根据我方英雄 config_id 返回技能范围。

        狄仁杰 133:
            1/2/3 技能都按 10000

        鲁班七号 112:
            1技能 9000
            2技能 无限距离
            3技能 6000
        """
        main_hero = getattr(self, "main_hero_info", None)
        if main_hero is None:
            return 0.0

        hero_id = main_hero.get("config_id")

        # 狄仁杰
        if hero_id == 133:
            return 10000.0

        # 鲁班七号
        if hero_id == 112:
            if skill_idx == 1:
                return 9000.0
            elif skill_idx == 2:
                return float("inf")
            elif skill_idx == 3:
                return 6000.0

        # 未知英雄，默认不给范围
        return 0.0

    def _enemy_in_main_skill_range(self, skill_idx):
        dist = self._get_main_enemy_dist()
        if dist is None:
            return 0

        skill_range = self._get_main_skill_range(skill_idx)
        return 1 if dist <= skill_range else 0

    def enemy_in_skill_1_range(self, frame_state, vector_feature):
        value = self._enemy_in_main_skill_range(1)
        vector_feature.append(value)

    def enemy_in_skill_2_range(self, frame_state, vector_feature):
        value = self._enemy_in_main_skill_range(2)
        vector_feature.append(value)

    def enemy_in_skill_3_range(self, frame_state, vector_feature):
        value = self._enemy_in_main_skill_range(3)
        vector_feature.append(value)

    def enemy_in_common_attack_range(self, frame_state, vector_feature):
        """
        敌方英雄是否在我方普攻范围内。
        使用 main_hero["attack_range"] 判断。
        """
        main_hero = getattr(self, "main_hero_info", None)
        if main_hero is None:
            vector_feature.append(0)
            return

        dist = self._get_main_enemy_dist()
        if dist is None:
            vector_feature.append(0)
            return

        attack_range = main_hero.get("attack_range", 0)
        value = 1 if dist <= attack_range else 0
        vector_feature.append(value)

    def main_enemy_dist_bin(self, frame_state, vector_feature):
        """
        我方英雄和敌方英雄距离的分桶 one-hot 原始类别。

        类别：
            0: dist > 10000
            1: 9000 < dist <= 10000
            2: 8000 < dist <= 9000
            3: 7000 < dist <= 8000
            4: 5000 < dist <= 7000
            5: dist <= 5000

        这里 append 类别编号，后面在 ini 里用 one_hot 展开成 6 维。
        """
        dist = self._get_main_enemy_dist()

        if dist is None:
            value = 0
        elif dist > 10000:
            value = 0
        elif dist > 9000:
            value = 1
        elif dist > 8000:
            value = 2
        elif dist > 7000:
            value = 3
        elif dist > 5000:
            value = 4
        else:
            value = 5

        vector_feature.append(value)
    
