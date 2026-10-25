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


class HeroProcess:
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
        self.main_cake_info = None
        self.enemy_cake_info = None
        self.transform_camp2_to_camp1 = self.main_camp_id == 2
    
        self.get_hero_config()
        self.map_feature_to_norm = self.normalizer.parse_config(self.hero_feature_config)
        self.view_dist = 15000

        # 原来是 80，新增 1 个 rage_left_time 特征，所以改成 81
        self.one_unit_feature_num = 81
        self.unit_buff_num = 1

        # =========================
        # 狂暴剩余持续时间
        # =========================
        # 召唤师技能槽位，一般 slot_type=6
        self.summon_skill_slot_type = 6

        # 狂暴召唤师技能 configId
        self.rage_skill_config_id = 80110

        # 狂暴持续 8s；1 frame ≈ 33ms，所以 8s ≈ 240 frame
        self.rage_duration_frames = 240

        # 如果拿不到 frame_no，则默认一次 process 对应 1 step = 6 frame
        self.default_frame_delta = 6
        self.last_frame_no_for_rage = None

        # 记录上一帧双方狂暴 usedTimes
        self.last_main_rage_used_times = None
        self.last_enemy_rage_used_times = None

        # 记录双方狂暴剩余帧数
        self.main_rage_left_frames = 0
        self.enemy_rage_left_frames = 0

    def get_hero_config(self):
        self.config = configparser.ConfigParser()
        self.config.optionxform = str
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "hero_feature_config.ini")
        self.config.read(config_path)

        # Get normalized configuration
        # 获取归一化的配置
        self.hero_feature_config = []
        for feature, config in self.config["feature_config"].items():
            self.hero_feature_config.append(f"{feature}:{config}")

        # Get feature function configuration
        # 获取特征函数的配置
        self.feature_func_map = {}
        for feature, func_name in self.config["feature_functions"].items():
            if hasattr(self, func_name):
                self.feature_func_map[feature] = getattr(self, func_name)
            else:
                raise ValueError(f"Unsupported function: {func_name}")

    def process_vec_hero(self, frame_state):
        self.generate_hero_info_list(frame_state)
        self.generate_organ_info_dict(frame_state)
        self.generate_cakes_info_dict(frame_state)

        # 先根据 usedTimes 更新狂暴剩余持续时间，再生成英雄特征
        self.update_rage_left_time(frame_state)

        # Generate hero features for our camp
        # 生成我方阵营的英雄特征
        main_camp_hero_vector_feature = self.generate_one_type_hero_feature(self.main_camp_hero_dict, "main_camp")
        
        # 生成敌方阵营的英雄特征
        enemy_camp_hero_vector_feature = self.generate_one_type_hero_feature(self.enemy_camp_hero_dict, "enemy_camp")

        return main_camp_hero_vector_feature, enemy_camp_hero_vector_feature
    

    def generate_cakes_info_dict(self, frame_state):
        self.main_cake_info = None
        self.enemy_cake_info = None

        cakes = frame_state.get("cakes", [])

        for cake in cakes:

            collider = cake.get("collider", {})
            location = collider.get("location", {})

            x = location.get("x")
            z = location.get("z")

            if x is None or z is None:
                continue

            # 根据墨家机关道血包固定位置判断阵营
            # camp 1: 负坐标侧
            # camp 2: 正坐标侧
            if x < 0 and z < 0:
                cake_camp = 1
            elif x > 0 and z > 0:
                cake_camp = 2
            else:
                continue

            cake_info = {
                "x": x,
                "z": z,
            }

            if cake_camp == self.main_camp:
                self.main_cake_info = cake_info
            else:
                self.enemy_cake_info = cake_info
            

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
                    self.main_camp_organ_info = organ
            else:
                if organ_subtype == 21:
                    self.enemy_camp_organ_dict["tower"] = organ
                    self.enemy_camp_organ_info = organ

    def generate_one_type_hero_feature(self, one_type_hero_info, camp):
        vector_feature = []
        num_heros_considered = 0
        for hero in one_type_hero_info.values():
            if num_heros_considered >= self.unit_buff_num:  # 最多考虑 unit_buff_num 个英雄
                break

            # Generate each specific feature through feature_func_map
            # 通过 feature_func_map 生成每个具体特征
            for feature_name, feature_func in self.feature_func_map.items():
                value = []
                self.feature_func_map[feature_name](hero, value, feature_name)
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
            num_heros_considered += 1

        if num_heros_considered < self.unit_buff_num:
            self.no_hero_feature(vector_feature, num_heros_considered)
        return vector_feature

    def no_hero_feature(self, vector_feature, num_heros_considered):
        for _ in range((self.unit_buff_num - num_heros_considered) * self.one_unit_feature_num):
            vector_feature.append(0)

    def is_alive(self, hero, vector_feature, feature_name):
        value = 0.0
        if hero["hp"] > 0:
            value = 1.0
        vector_feature.append(value)

    def level(self, hero, vector_feature, feature_name):
        value = hero["level"]
        vector_feature.append(value)

    def hp(self, hero, vector_feature, feature_name):
        value = hero["hp"]
        vector_feature.append(value)
    
    def hp_rate(self, hero, vector_feature, feature_name):
        value = 0
        if hero["max_hp"] > 0:
            value = hero["hp"] / hero["max_hp"]
        vector_feature.append(value)

    def max_hp(self, hero, vector_feature, feature_name):
        value = hero["max_hp"]
        vector_feature.append(value)

    def hp_recover(self, hero, vector_feature, feature_name):
        value = hero["hp_recover"]
        vector_feature.append(value)
    
    def ep(self, hero, vector_feature, feature_name):
        value = hero["ep"]
        vector_feature.append(value)
    
    def ep_rate(self, hero, vector_feature, feature_name):
        value = 0
        if hero["max_ep"] > 0:
            value = hero["ep"] / hero["max_ep"]
        vector_feature.append(value)

    def max_ep(self, hero, vector_feature, feature_name):
        value = hero["max_ep"]
        vector_feature.append(value)
    
    def ep_recover(self, hero, vector_feature, feature_name):
        value = hero["ep_recover"]
        vector_feature.append(value)
    
    def phy_atk(self, hero, vector_feature, feature_name):
        value = hero["phy_atk"]
        vector_feature.append(value)
    
    def mgc_atk(self, hero, vector_feature, feature_name):
        value = hero["mgc_atk"]
        vector_feature.append(value)
    
    def phy_def(self, hero, vector_feature, feature_name):
        value = hero["phy_def"]
        vector_feature.append(value)
    
    def mgc_def(self, hero, vector_feature, feature_name):
        value = hero["mgc_def"]
        vector_feature.append(value)
    
    def kill_cnt(self, hero, vector_feature, feature_name):
        value = hero["kill_cnt"]
        vector_feature.append(value)

    def dead_cnt(self, hero, vector_feature, feature_name):
        value = hero["dead_cnt"]
        vector_feature.append(value)

    def money_cnt(self, hero, vector_feature, feature_name):
        value = hero["money_cnt"]
        vector_feature.append(value)

    def get_location_x(self, hero, vector_feature, feature_name):
        value = hero["location"]["x"]
        if self.transform_camp2_to_camp1 and value != 100000:
            value = 0 - value
        vector_feature.append(value)

    def get_location_z(self, hero, vector_feature, feature_name):
        value = hero["location"]["z"]
        if self.transform_camp2_to_camp1 and value != 100000:
            value = 0 - value
        vector_feature.append(value)

    def dist_to_hero(self, hero, vector_feature, feature_name):
        main_hero_location = self.main_hero_info["location"]
        enemy_hero_location = self.enemy_hero_info["location"]
        main_hero_x = main_hero_location["x"]
        main_hero_z = main_hero_location["z"]
        enemy_hero_x = enemy_hero_location["x"]
        enemy_hero_z = enemy_hero_location["z"]
        dist = ((main_hero_x - enemy_hero_x) ** 2 + (main_hero_z - enemy_hero_z) ** 2) ** 0.5
        vector_feature.append(dist)

    def mov_spd(self, hero, vector_feature, feature_name):
        value = hero["mov_spd"]
        vector_feature.append(value)
    
    def attack_range(self, hero, vector_feature, feature_name):
        value = hero["attack_range"]
        vector_feature.append(value)
    
    def atk_spd(self, hero, vector_feature, feature_name):
        value = hero["atk_spd"]
        vector_feature.append(value)
    
    def phy_armor_hurt(self, hero, vector_feature, feature_name):
        value = hero["phy_armor_hurt"]
        vector_feature.append(value)
    
    def mgc_armor_hurt(self, hero, vector_feature, feature_name):
        value = hero["mgc_armor_hurt"]
        vector_feature.append(value)

    def crit_rate(self, hero, vector_feature, feature_name):
        value = hero["crit_rate"]
        vector_feature.append(value)
    
    def crit_effe(self, hero, vector_feature, feature_name):
        value = hero["crit_effe"]
        vector_feature.append(value)

    def phy_vamp(self, hero, vector_feature, feature_name):
        value = hero["phy_vamp"]
        vector_feature.append(value)
    
    def mgc_vamp(self, hero, vector_feature, feature_name):
        value = hero["mgc_vamp"]
        vector_feature.append(value)
    
    def cd_reduce(self, hero, vector_feature, feature_name):
        value = hero["cd_reduce"]
        vector_feature.append(value)
    
    def ctrl_reduce(self, hero, vector_feature, feature_name):
        value = hero["ctrl_reduce"]
        vector_feature.append(value)
    
    def exp(self, hero, vector_feature, feature_name):
        value = hero["exp"]
        vector_feature.append(value)
    
    def money(self, hero, vector_feature, feature_name):
        value = hero["money"]
        vector_feature.append(value)
    
    def revive_time(self, hero, vector_feature, feature_name):
        value = hero["revive_time"]
        vector_feature.append(value)
        
    
    def kill_income(self, hero, vector_feature, feature_name):
        value = hero["kill_income"]
        vector_feature.append(value)

    def _get_skill_by_slot_type(self, hero, slot_type):
        skill_states = hero["skill_state"]["slot_states"]

        for skill in skill_states:
            if skill["slot_type"] == slot_type:
                return skill

        raise KeyError(f"slot_type {slot_type} not found in skill_states")

    def _get_rage_frame_delta(self, frame_state):
        """
        计算距离上一次处理经过了多少 frame。
        如果 frame_state 中没有 frame_no，则默认一次 process 对应 1 step = 6 frame。
        """
        current_frame_no = frame_state.get("frame_no", None)

        if current_frame_no is None:
            return self.default_frame_delta

        if self.last_frame_no_for_rage is None:
            self.last_frame_no_for_rage = current_frame_no
            return self.default_frame_delta

        frame_delta = current_frame_no - self.last_frame_no_for_rage
        self.last_frame_no_for_rage = current_frame_no

        if frame_delta <= 0:
            return self.default_frame_delta

        return frame_delta

    def _get_rage_skill(self, hero):
        """
        获取当前英雄的狂暴召唤师技能槽。
        如果没有携带狂暴，返回 None。
        """
        if hero is None:
            return None

        try:
            skill = self._get_skill_by_slot_type(hero, self.summon_skill_slot_type)
        except Exception:
            return None

        if skill.get("configId", 0) != self.rage_skill_config_id:
            return None

        return skill

    def update_rage_left_time(self, frame_state):
        """
        根据狂暴 usedTimes 的增量，更新双方狂暴剩余持续时间。

        规则：
            1. 当前帧 usedTimes > 上一帧 usedTimes，说明使用了狂暴；
            2. 使用后把剩余时间设置为 rage_duration_frames；
            3. 后续每次处理按 frame_delta 递减；
            4. 没有携带狂暴，或者持续时间扣完，则为 0。
        """
        frame_delta = self._get_rage_frame_delta(frame_state)

        # 先扣减已有剩余时间
        self.main_rage_left_frames = max(0, self.main_rage_left_frames - frame_delta)
        self.enemy_rage_left_frames = max(0, self.enemy_rage_left_frames - frame_delta)

        # 我方英雄狂暴计时
        self._update_one_rage_left_time(
            hero=self.main_hero_info,
            last_used_attr="last_main_rage_used_times",
            left_frames_attr="main_rage_left_frames",
        )

        # 敌方英雄狂暴计时
        self._update_one_rage_left_time(
            hero=self.enemy_hero_info,
            last_used_attr="last_enemy_rage_used_times",
            left_frames_attr="enemy_rage_left_frames",
        )

    def _update_one_rage_left_time(self, hero, last_used_attr, left_frames_attr):
        """
        更新单个英雄的狂暴剩余时间。
        """
        rage_skill = self._get_rage_skill(hero)

        # 没有携带狂暴，直接清零
        if rage_skill is None:
            setattr(self, last_used_attr, None)
            setattr(self, left_frames_attr, 0)
            return

        current_used_times = rage_skill.get("usedTimes", 0)
        last_used_times = getattr(self, last_used_attr)

        # 第一次进入只建立基准，不触发计时
        if last_used_times is None:
            setattr(self, last_used_attr, current_used_times)
            return

        # usedTimes 变小，通常说明新一局或状态重置
        if current_used_times < last_used_times:
            setattr(self, last_used_attr, current_used_times)
            setattr(self, left_frames_attr, 0)
            return

        # 当前帧使用了狂暴
        if current_used_times > last_used_times:
            setattr(self, left_frames_attr, self.rage_duration_frames)

        setattr(self, last_used_attr, current_used_times)

    def rage_left_time(self, hero, vector_feature, feature_name):
        """
        当前 hero 的狂暴剩余持续时间，单位：frame。

        如果没有携带狂暴，或者当前不在狂暴持续时间内，则为 0。
        归一化建议：rage_left_time = min_max:0:240
        """
        if hero is None:
            value = 0
        elif hero.get("camp") == self.main_camp_id:
            value = self.main_rage_left_frames
        else:
            value = self.enemy_rage_left_frames

        vector_feature.append(value)

    def skill1_usable(self, hero, vector_feature, feature_name):
        
        skill = self._get_skill_by_slot_type(hero, 1)
        value = skill["usable"]
        vector_feature.append(value)
                
    def skill1_cd(self, hero, vector_feature, feature_name):
        
        skill = self._get_skill_by_slot_type(hero, 1)
        value = skill["cooldown"]
        vector_feature.append(value)
    
    def skill1_cd_rate(self, hero, vector_feature, feature_name):
        
        skill = self._get_skill_by_slot_type(hero, 1)

        value = 0
        if skill["cooldown_max"] > 0:
            value = skill["cooldown"] / skill["cooldown_max"]

        vector_feature.append(value)

    def skill2_usable(self, hero, vector_feature, feature_name):
        skill = self._get_skill_by_slot_type(hero, 2)
        value = skill["usable"]
        vector_feature.append(value)


    def skill2_cd(self, hero, vector_feature, feature_name):
        skill = self._get_skill_by_slot_type(hero, 2)
        value = skill["cooldown"]
        vector_feature.append(value)


    def skill2_cd_rate(self, hero, vector_feature, feature_name):
        skill = self._get_skill_by_slot_type(hero, 2)

        value = 0
        if skill["cooldown_max"] > 0:
            value = skill["cooldown"] / skill["cooldown_max"]

        vector_feature.append(value)

    def skill3_usable(self, hero, vector_feature, feature_name):
        skill = self._get_skill_by_slot_type(hero, 3)
        value = skill["usable"]
        vector_feature.append(value)


    def skill3_cd(self, hero, vector_feature, feature_name):
        skill = self._get_skill_by_slot_type(hero, 3)
        value = skill["cooldown"]
        vector_feature.append(value)


    def skill3_cd_rate(self, hero, vector_feature, feature_name):
        skill = self._get_skill_by_slot_type(hero, 3)

        value = 0
        if skill["cooldown_max"] > 0:
            value = skill["cooldown"] / skill["cooldown_max"]

        vector_feature.append(value)

    
    def skill5_usable(self, hero, vector_feature, feature_name):
        skill = self._get_skill_by_slot_type(hero, 5)
        value = skill["usable"]
        vector_feature.append(value)


    def skill5_cd(self, hero, vector_feature, feature_name):
        skill = self._get_skill_by_slot_type(hero, 5)
        value = skill["cooldown"]
        vector_feature.append(value)


    def skill5_cd_rate(self, hero, vector_feature, feature_name):
        skill = self._get_skill_by_slot_type(hero, 5)

        value = 0
        if skill["cooldown_max"] > 0:
            value = skill["cooldown"] / skill["cooldown_max"]

        vector_feature.append(value)

    def skill6_usable(self, hero, vector_feature, feature_name):
        skill = self._get_skill_by_slot_type(hero, 6)
        
        value = skill["usable"]
        vector_feature.append(value)


    def skill6_cd(self, hero, vector_feature, feature_name):
        skill = self._get_skill_by_slot_type(hero, 6)
        value = skill["cooldown"]
        vector_feature.append(value)


    def skill6_cd_rate(self, hero, vector_feature, feature_name):
        skill = self._get_skill_by_slot_type(hero, 6)

        value = 0
        if skill["cooldown_max"] > 0:
            value = skill["cooldown"] / skill["cooldown_max"]

        vector_feature.append(value)


    def skill0_usable(self, hero, vector_feature, feature_name):
        skill = self._get_skill_by_slot_type(hero, 0)
        value = skill["usable"]
        vector_feature.append(value)

    def hero_in_self_camp_tower_atk_range(self, hero, vector_feature, feature_name):
        """
        当前 hero 是否在自己阵营防御塔攻击范围内。

        我方英雄 -> 判断是否在我方塔范围内；
        敌方英雄 -> 判断是否在敌方塔范围内。
        """
        if hero["camp"] == self.main_camp_id:
            self_organ = self.main_camp_organ_info
        else:
            self_organ = self.enemy_camp_organ_info

        hero_location = hero["location"]
        organ_location = self_organ["location"]

        hero_x = hero_location["x"]
        hero_z = hero_location["z"]
        organ_x = organ_location["x"]
        organ_z = organ_location["z"]

        dist = ((hero_x - organ_x) ** 2 + (hero_z - organ_z) ** 2) ** 0.5

        value = 0
        if dist <= self_organ["attack_range"]:
            value = 1

        vector_feature.append(value)

    def hero_in_opponent_camp_tower_atk_range(self, hero, vector_feature, feature_name):
        """
        当前 hero 是否在敌方阵营防御塔攻击范围内。

        我方英雄 -> 判断是否在敌方塔范围内；
        敌方英雄 -> 判断是否在我方塔范围内。
        """
        if hero["camp"] == self.main_camp_id:
            opponent_organ = self.enemy_camp_organ_info
        else:
            opponent_organ = self.main_camp_organ_info

        hero_location = hero["location"]
        organ_location = opponent_organ["location"]

        hero_x = hero_location["x"]
        hero_z = hero_location["z"]
        organ_x = organ_location["x"]
        organ_z = organ_location["z"]

        dist = ((hero_x - organ_x) ** 2 + (hero_z - organ_z) ** 2) ** 0.5

        value = 0
        if dist <= opponent_organ["attack_range"]:
            value = 1

        vector_feature.append(value)

    def is_hero_under_opponent_tower_atk(self, hero, vector_feature, feature_name):
        """
        当前 hero 是否正在被敌方阵营防御塔攻击。

        我方英雄 -> 判断敌方塔是否正在攻击我方英雄；
        敌方英雄 -> 判断我方塔是否正在攻击敌方英雄。
        """
        if hero["camp"] == self.main_camp_id:
            opponent_organ = self.enemy_camp_organ_info
        else:
            opponent_organ = self.main_camp_organ_info

        value = 0
        if opponent_organ["attack_target"] == hero["runtime_id"]:
            value = 1

        vector_feature.append(value)
    # cake 相关特征：是否存在，和英雄的距离，dx，dz，这里做成两个血包的，都是相对于自己阵营英雄的视角。

    def self_cake_exist(self, hero, vector_feature, feature_name):
        if hero['camp'] == self.main_hero_info["camp"]:
            self_cake_info = self.main_cake_info
        else:
            self_cake_info = self.enemy_cake_info
        
        value = 0
        if self_cake_info is not None:
            value = 1
        vector_feature.append(value)
    
    def opponent_cake_exist(self, hero, vector_feature, feature_name):
        if hero['camp'] == self.main_hero_info["camp"]:
            opponent_cake_info = self.enemy_cake_info
        else:
            opponent_cake_info = self.main_cake_info
        
        value = 0
        if opponent_cake_info is not None:
            value = 1
        vector_feature.append(value)
    
    def self_cake_dist(self, hero, vector_feature, feature_name):
        if hero['camp'] == self.main_hero_info["camp"]:
            
            self_cake_info = self.main_cake_info
        else:
            
            self_cake_info = self.enemy_cake_info
        
        value = 0
        if self_cake_info is not None:
            self_hero_location = hero["location"]
            self_hero_x = self_hero_location["x"]
            self_hero_z = self_hero_location["z"]
            self_cake_x = self_cake_info["x"]
            self_cake_z = self_cake_info["z"]
            value = ((self_hero_x - self_cake_x) ** 2 + (self_hero_z - self_cake_z) ** 2) ** 0.5
        vector_feature.append(value)

    def opponent_cake_dist(self, hero, vector_feature, feature_name):
        if hero['camp'] == self.main_hero_info["camp"]:
            
            opponent_cake_info = self.enemy_cake_info
        else:
            
            opponent_cake_info = self.main_cake_info
        
        value = 0
        if opponent_cake_info is not None:
            opponent_hero_location = hero["location"]
            opponent_hero_x = opponent_hero_location["x"]
            opponent_hero_z = opponent_hero_location["z"]
            opponent_cake_x = opponent_cake_info["x"]
            opponent_cake_z = opponent_cake_info["z"]
            value = ((opponent_hero_x - opponent_cake_x) ** 2 + (opponent_hero_z - opponent_cake_z) ** 2) ** 0.5
        vector_feature.append(value)

    
    def self_cake_relative_location_x(self, hero, vector_feature, feature_name):
        if hero['camp'] == self.main_hero_info["camp"]:
            self_cake = self.main_cake_info
            
        else:
            self_cake = self.enemy_cake_info
            
        
        value = 0
        if self_cake is not None:
            self_cake_x = self_cake["x"]
            self_hero_x = hero["location"]["x"]
            x_diff = self_cake_x - self_hero_x
            if self.transform_camp2_to_camp1 and self_cake_x != 100000:
                x_diff = -x_diff
            value = (x_diff + 15000) / 30000.0
        vector_feature.append(value)

    def self_cake_relative_location_z(self, hero, vector_feature, feature_name):
        if hero['camp'] == self.main_hero_info["camp"]:
            self_cake = self.main_cake_info
            
        else:
            self_cake = self.enemy_cake_info
            
        
        value = 0
        if self_cake is not None:
            self_cake_z = self_cake["z"]
            self_hero_z = hero["location"]["z"]
            z_diff = self_cake_z - self_hero_z
            if self.transform_camp2_to_camp1 and self_cake_z != 100000:
                z_diff = -z_diff
            value = (z_diff + 15000) / 30000.0
        vector_feature.append(value)

    def opponent_cake_relative_location_x(self, hero, vector_feature, feature_name):
        if hero['camp'] == self.main_hero_info["camp"]:
            opponent_cake = self.enemy_cake_info
            
        else:
            opponent_cake = self.main_cake_info
            
        
        value = 0
        if opponent_cake is not None:
            opponent_cake_x = opponent_cake["x"]
            location_x = hero["location"]["x"]
            x_diff = opponent_cake_x - location_x
            if self.transform_camp2_to_camp1 and opponent_cake_x != 100000:
                x_diff = -x_diff
            value = (x_diff + 15000) / 30000.0
        vector_feature.append(value)

    def opponent_cake_relative_location_z(self, hero, vector_feature, feature_name):
        if hero['camp'] == self.main_hero_info["camp"]:
            opponent_cake = self.enemy_cake_info
            
        else:
            opponent_cake = self.main_cake_info
            
        value = 0
        if opponent_cake is not None:
            opponent_cake_z = opponent_cake["z"]
            location_z = hero["location"]["z"]
            z_diff = opponent_cake_z - location_z
            if self.transform_camp2_to_camp1 and opponent_cake_z != 100000:
                z_diff = -z_diff
            value = (z_diff + 15000) / 30000.0
        vector_feature.append(value)

    def _get_summon_skill_id(self, hero):
        """
        召唤师技能槽 slot_type = 6。
        返回当前英雄携带的召唤师技能 configId。
        """
        try:
            skill = self._get_skill_by_slot_type(hero, 6)
            return skill.get("configId", 0)
        except Exception:
            return 0

    def summon_skill_heal(self, hero, vector_feature, feature_name):
        # 80102 治疗
        value = self._get_summon_skill_id(hero)
        vector_feature.append(value)

    def summon_skill_sprint(self, hero, vector_feature, feature_name):
        # 80109 疾跑
        value = self._get_summon_skill_id(hero)
        vector_feature.append(value)

    def summon_skill_punish(self, hero, vector_feature, feature_name):
        # 80104 惩击
        value = self._get_summon_skill_id(hero)
        vector_feature.append(value)

    def summon_skill_execute(self, hero, vector_feature, feature_name):
        # 80108 终结 / 斩杀
        value = self._get_summon_skill_id(hero)
        vector_feature.append(value)

    def summon_skill_rage(self, hero, vector_feature, feature_name):
        # 80110 狂暴
        value = self._get_summon_skill_id(hero)
        vector_feature.append(value)

    def summon_skill_disrupt(self, hero, vector_feature, feature_name):
        # 80105 干扰
        value = self._get_summon_skill_id(hero)
        vector_feature.append(value)

    def summon_skill_daze(self, hero, vector_feature, feature_name):
        # 80103 眩晕
        value = self._get_summon_skill_id(hero)
        vector_feature.append(value)

    def summon_skill_purify(self, hero, vector_feature, feature_name):
        # 80107 净化
        value = self._get_summon_skill_id(hero)
        vector_feature.append(value)

    def summon_skill_weak(self, hero, vector_feature, feature_name):
        # 80121 弱化
        value = self._get_summon_skill_id(hero)
        vector_feature.append(value)

    def summon_skill_flash(self, hero, vector_feature, feature_name):
        # 80115 闪现
        value = self._get_summon_skill_id(hero)
        vector_feature.append(value)

    # 是否在草丛
    def is_in_grass(self, hero, vector_feature, feature_name):
        value = hero['is_in_grass']
        vector_feature.append(value)

    # 是否在对方的攻击范围内
    def hero_in_opponent_normal_attack_range(self, hero, vector_feature, feature_name):
        """
        当前 hero 是否在对方英雄的普通攻击范围内。

        我方英雄 -> 判断是否在敌方英雄普攻范围内；
        敌方英雄 -> 判断是否在我方英雄普攻范围内。
        """
        if hero["camp"] == self.main_camp_id:
            opponent_hero = self.enemy_hero_info
        else:
            opponent_hero = self.main_hero_info

        hero_location = hero["location"]
        opponent_location = opponent_hero["location"]

        hero_x = hero_location["x"]
        hero_z = hero_location["z"]
        opponent_x = opponent_location["x"]
        opponent_z = opponent_location["z"]

        dist = ((hero_x - opponent_x) ** 2 + (hero_z - opponent_z) ** 2) ** 0.5

        value = 0
        if dist <= opponent_hero["attack_range"]:
            value = 1

        vector_feature.append(value)

    # 是否正在攻击对方
    def is_attacking_opponent_hero(self, hero, vector_feature, feature_name):
        """
        当前 hero 是否正在攻击对方英雄。

        我方英雄 -> 判断 attack_target 是否是敌方英雄；
        敌方英雄 -> 判断 attack_target 是否是我方英雄。
        """
        if hero["camp"] == self.main_camp_id:
            opponent_hero = self.enemy_hero_info
        else:
            opponent_hero = self.main_hero_info

        value = 0
        if hero.get("attack_target", 0) == opponent_hero["runtime_id"]:
            value = 1

        vector_feature.append(value)
        
    # 是否正在被对方攻击
    def is_attacked_by_opponent_hero(self, hero, vector_feature, feature_name):
        """
        当前 hero 是否正在被对方英雄攻击。

        我方英雄 -> 判断敌方英雄 attack_target 是否是我方英雄；
        敌方英雄 -> 判断我方英雄 attack_target 是否是敌方英雄。
        """
        if hero["camp"] == self.main_camp_id:
            opponent_hero = self.enemy_hero_info
        else:
            opponent_hero = self.main_hero_info

        value = 0
        if opponent_hero.get("attack_target", 0) == hero["runtime_id"]:
            value = 1

        vector_feature.append(value)

    



