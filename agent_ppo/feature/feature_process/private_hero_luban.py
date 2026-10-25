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


class LubanProcess:
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
        self.main_camp_luban_dict = {}
        self.enemy_camp_luban_dict = {}
        self.bullets = []

        self.transform_camp2_to_camp1 = self.main_camp_id == 2
        self.get_luban_config()
        self.map_feature_to_norm = self.normalizer.parse_config(self.luban_feature_config)
        self.view_dist = 15000
        self.one_unit_feature_num = 15
        self.unit_buff_num = 1

        # =========================
        # 鲁班大招剩余时间计时
        # =========================

        # 鲁班大招槽位，按你当前 bullet 逻辑，slot_type=3 视为大招
        self.luban_ult_slot_type = 3

        # 鲁班大招持续 14s
        # 1 frame ≈ 33ms，约等于 30 frame/s
        # 14s ≈ 14 * 30 = 420 frame
        self.luban_ult_duration_frames = 420

        # 如果拿不到 frame_no，则默认一次 process 对应 1 step = 6 frame
        self.default_frame_delta = 6
        self.last_frame_no = None

        # 记录双方鲁班大招上一次释放次数
        self.last_main_luban_ult_used_times = None
        self.last_enemy_luban_ult_used_times = None

        # 记录双方鲁班自己的大招剩余时间
        self.main_luban_ult_left_frames = 0
        self.enemy_luban_ult_left_frames = 0

    def _get_opponent_hero(self, hero):
        if hero is None:
            return None

        if hero.get("camp") == self.main_hero_info.get("camp"):
            return self.enemy_hero_info

        return self.main_hero_info

    def _get_luban_ult_bullet(self, hero):
        """
        找当前 hero 所属阵营释放的鲁班大招 bullet。

        条件：
            1. bullet.camp == hero.camp
            2. bullet.slot_type == 3
            3. bullet 有 location

        如果同一帧有多个同阵营 slot_type=3 bullet，取 runtime_id 最大的那个。
        """
        if hero is None:
            return None

        hero_camp = hero.get("camp")
        if hero_camp is None:
            return None

        candidates = []

        for bullet in self.bullets:
            if bullet.get("camp") != hero_camp:
                continue

            if bullet.get("slot_type") != 3:
                continue

            if "location" not in bullet:
                continue

            candidates.append(bullet)

        if len(candidates) == 0:
            return None

        return max(candidates, key=lambda x: x.get("runtime_id", 0))

    def get_luban_config(self):
        self.config = configparser.ConfigParser()
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "luban_feature_config.ini")
        self.config.read(config_path)

        # Get normalized configuration
        # 获取归一化的配置
        self.luban_feature_config = []
        for feature, config in self.config["feature_config"].items():
            self.luban_feature_config.append(f"{feature}:{config}")

        # Get feature function configuration
        # 获取特征函数的配置
        self.feature_func_map = {}
        for feature, func_name in self.config["feature_functions"].items():
            if hasattr(self, func_name):
                self.feature_func_map[feature] = getattr(self, func_name)
            else:
                raise ValueError(f"Unsupported function: {func_name}")

    def _get_frame_delta(self, frame_state):
        """
        计算距离上一次处理经过了多少 frame。

        如果有 frame_no，就用真实帧号差；
        如果没有 frame_no，就默认一次 process 对应 1 step = 6 frame。
        """
        current_frame_no = frame_state.get("frame_no", None)

        if current_frame_no is None:
            return self.default_frame_delta

        if self.last_frame_no is None:
            self.last_frame_no = current_frame_no
            return self.default_frame_delta

        frame_delta = current_frame_no - self.last_frame_no
        self.last_frame_no = current_frame_no

        # frame_no 可能在新一局重置，或者偶尔出现异常
        if frame_delta <= 0:
            return self.default_frame_delta

        return frame_delta

    def update_luban_ult_left_timer(self, frame_delta):
        """
        根据鲁班大招 usedTimes 的增量，更新鲁班大招剩余时间。

        语义：
            main_luban_ult_left_frames:
                我方鲁班释放大招后，我方鲁班大招还剩多少帧。

            enemy_luban_ult_left_frames:
                敌方鲁班释放大招后，敌方鲁班大招还剩多少帧。
        """

        # 先扣减已有大招持续时间
        self.main_luban_ult_left_frames = max(
            0,
            self.main_luban_ult_left_frames - frame_delta
        )
        self.enemy_luban_ult_left_frames = max(
            0,
            self.enemy_luban_ult_left_frames - frame_delta
        )

        # =========================
        # 我方鲁班大招释放检测
        # =========================
        if self.main_hero_luban and self.main_hero_info is not None:
            main_ult_skill = self._get_skill_by_slot_type(
                self.main_hero_info,
                self.luban_ult_slot_type
            )

            current_used_times = 0
            if main_ult_skill is not None:
                current_used_times = main_ult_skill.get("usedTimes", 0)

            if self.last_main_luban_ult_used_times is None:
                self.last_main_luban_ult_used_times = current_used_times
            else:
                if current_used_times > self.last_main_luban_ult_used_times:
                    # 我方鲁班刚释放大招，重置为 14s 对应的帧数
                    self.main_luban_ult_left_frames = self.luban_ult_duration_frames

                elif current_used_times < self.last_main_luban_ult_used_times:
                    # 新一局 usedTimes 可能重置，避免继承上一局计时
                    self.main_luban_ult_left_frames = 0

                self.last_main_luban_ult_used_times = current_used_times
        else:
            self.last_main_luban_ult_used_times = None
            self.main_luban_ult_left_frames = 0

        # =========================
        # 敌方鲁班大招释放检测
        # =========================
        if self.enemy_hero_luban and self.enemy_hero_info is not None:
            enemy_ult_skill = self._get_skill_by_slot_type(
                self.enemy_hero_info,
                self.luban_ult_slot_type
            )

            current_used_times = 0
            if enemy_ult_skill is not None:
                current_used_times = enemy_ult_skill.get("usedTimes", 0)

            if self.last_enemy_luban_ult_used_times is None:
                self.last_enemy_luban_ult_used_times = current_used_times
            else:
                if current_used_times > self.last_enemy_luban_ult_used_times:
                    # 敌方鲁班刚释放大招，重置为 14s 对应的帧数
                    self.enemy_luban_ult_left_frames = self.luban_ult_duration_frames

                elif current_used_times < self.last_enemy_luban_ult_used_times:
                    # 新一局 usedTimes 可能重置，避免继承上一局计时
                    self.enemy_luban_ult_left_frames = 0

                self.last_enemy_luban_ult_used_times = current_used_times
        else:
            self.last_enemy_luban_ult_used_times = None
            self.enemy_luban_ult_left_frames = 0

    def process_vec_luban(self, frame_state):
        self.generate_hero_info_list(frame_state)
        self.generate_luban_info_dict(frame_state)
        self.generate_bullet_info_list(frame_state)

        # 根据 frame_no 计算距离上次处理经过了多少帧
        frame_delta = self._get_frame_delta(frame_state)

        # 根据 usedTimes 更新鲁班大招剩余时间
        self.update_luban_ult_left_timer(frame_delta)
    
        # Generate features for enemy team's luban
        # 生成敌方阵营的鲁班私有特征
        enemy_camp_luban_vector_feature = self.generate_one_type_luban_feature(
            self.enemy_hero_luban, self.enemy_hero_info, None
        )

        main_camp_luban_vector_feature = self.generate_one_type_luban_feature(
            self.main_hero_luban, self.main_hero_info, None
        )

        return main_camp_luban_vector_feature, enemy_camp_luban_vector_feature
    
    def generate_bullet_info_list(self, frame_state):
        self.bullets = frame_state.get("bullets", [])

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

    def generate_luban_info_dict(self, frame_state):
        
        if self.main_hero_info['config_id'] == 112:
            self.main_hero_luban = True
        else:
            self.main_hero_luban = False
        
        if self.enemy_hero_info['config_id'] == 112:
            self.enemy_hero_luban = True
        else:
            self.enemy_hero_luban = False
        

    def generate_one_type_luban_feature(self, is_luban, hero, camp):
        vector_feature = []
        num_lubans_considered = 0

        def process_luban(luban):
            nonlocal num_lubans_considered

            # Generate each specific feature through feature_func_map
            # 通过 feature_func_map 生成每个具体特征
            for feature_name, feature_func in self.feature_func_map.items():
                value = []
                self.feature_func_map[feature_name](luban, value)

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

            num_lubans_considered += 1

        if is_luban:
            process_luban(hero)

        if num_lubans_considered < self.unit_buff_num:
            self.no_luban_feature(vector_feature, num_lubans_considered)

        return vector_feature

    def no_luban_feature(self, vector_feature, num_lubans_considered):
        for _ in range((self.unit_buff_num - num_lubans_considered) * self.one_unit_feature_num):
            vector_feature.append(0)

    def is_luban(self, hero, vector_feature):
        vector_feature.append(1)
    
    def _get_skill_by_slot_type(self, hero, slot_type):
        skill_states = hero.get("skill_state", {}).get("slot_states", [])

        for skill in skill_states:
            if skill.get("slot_type") == slot_type:
                return skill

        return None

    def normal_attack_stage(self, hero, vector_feature):
        """
        鲁班普通攻击阶段。

        从技能槽 slot_type=0 判断当前普攻阶段。
        当前已观察到鲁班普攻相关 configId 包括：
            11200
            11201
            11202
            11203
            11204

        这里函数只输出 configId，具体 one-hot 交给 ini。
        """
        skill = self._get_skill_by_slot_type(hero, 0)

        if skill is None:
            value = 0
        else:
            value = skill.get("configId", 0)

        vector_feature.append(value)

    
    def ult_bullet_location_x(self, hero, vector_feature):
        bullet = self._get_luban_ult_bullet(hero)

        if bullet is None:
            value = 0
        else:
            value = bullet.get("location", {}).get("x", 0)

            if self.transform_camp2_to_camp1:
                value = 0 - value

        vector_feature.append(value)
    

    def ult_bullet_location_z(self, hero, vector_feature):
        bullet = self._get_luban_ult_bullet(hero)

        if bullet is None:
            value = 0
        else:
            value = bullet.get("location", {}).get("z", 0)

            if self.transform_camp2_to_camp1:
                value = 0 - value

        vector_feature.append(value)


    
    def ult_bullet_opponent_relative_location_x(self, hero, vector_feature):
        bullet = self._get_luban_ult_bullet(hero)
        opponent_hero = self._get_opponent_hero(hero)

        if bullet is None or opponent_hero is None:
            value = 0
        else:
            bullet_x = bullet.get("location", {}).get("x", 0)
            opponent_x = opponent_hero.get("location", {}).get("x", 0)

            x_diff = bullet_x - opponent_x

            if self.transform_camp2_to_camp1:
                x_diff = 0 - x_diff

            value = (x_diff + 15000) / 30000.0

        vector_feature.append(value)

    
    def ult_bullet_opponent_relative_location_z(self, hero, vector_feature):
        bullet = self._get_luban_ult_bullet(hero)
        opponent_hero = self._get_opponent_hero(hero)

        if bullet is None or opponent_hero is None:
            value = 0
        else:
            bullet_z = bullet.get("location", {}).get("z", 0)
            opponent_z = opponent_hero.get("location", {}).get("z", 0)

            z_diff = bullet_z - opponent_z

            if self.transform_camp2_to_camp1:
                z_diff = 0 - z_diff

            value = (z_diff + 15000) / 30000.0

        vector_feature.append(value)

    
    def ult_bullet_opponent_dist(self, hero, vector_feature):
        bullet = self._get_luban_ult_bullet(hero)
        opponent_hero = self._get_opponent_hero(hero)

        if bullet is None or opponent_hero is None:
            value = 0
        else:
            bullet_location = bullet.get("location", {})
            opponent_location = opponent_hero.get("location", {})

            bullet_x = bullet_location.get("x", 0)
            bullet_z = bullet_location.get("z", 0)

            opponent_x = opponent_location.get("x", 0)
            opponent_z = opponent_location.get("z", 0)

            value = math.sqrt(
                (bullet_x - opponent_x) ** 2 +
                (bullet_z - opponent_z) ** 2
            )

        vector_feature.append(value)
    
    

    def ult_bullet_exist(self, hero, vector_feature):
        bullet = self._get_luban_ult_bullet(hero)

        if bullet is None:
            value = 0
        else:
            value = 1

        vector_feature.append(value)

    def ult_bullet_opponent_in_4900(self, hero, vector_feature):
        """
        鲁班大招子弹与对方英雄距离是否在 4900 以内。

        规则：
            有大招子弹，且子弹到对方英雄距离 <= 4900 -> 1
            没有大招子弹 -> 0
            距离大于 4900 -> 0
        """
        bullet = self._get_luban_ult_bullet(hero)
        opponent_hero = self._get_opponent_hero(hero)

        if bullet is None or opponent_hero is None:
            value = 0
        else:
            bullet_location = bullet.get("location", {})
            opponent_location = opponent_hero.get("location", {})

            bullet_x = bullet_location.get("x", 0)
            bullet_z = bullet_location.get("z", 0)

            opponent_x = opponent_location.get("x", 0)
            opponent_z = opponent_location.get("z", 0)

            dist = math.sqrt(
                (bullet_x - opponent_x) ** 2
                + (bullet_z - opponent_z) ** 2
            )

            value = 1 if dist <= 4900 else 0

        vector_feature.append(value)

    def ult_left_time(self, hero, vector_feature):
        """
        当前这个鲁班自己的大招剩余时间。

        如果当前 hero 是我方鲁班：
            输出我方鲁班大招剩余帧数。

        如果当前 hero 是敌方鲁班：
            输出敌方鲁班大招剩余帧数。
        """
        if hero is None:
            value = 0
        elif hero.get("camp") == self.main_camp_id:
            value = self.main_luban_ult_left_frames
        else:
            value = self.enemy_luban_ult_left_frames

        vector_feature.append(value)