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


class DirenjieProcess:
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

        self.main_hero_info = None
        self.enemy_hero_info = None

        self.main_hero_direnjie = False
        self.enemy_hero_direnjie = False

        self.transform_camp2_to_camp1 = self.main_camp_id == 2

        self.get_direnjie_config()
        self.map_feature_to_norm = self.normalizer.parse_config(self.direnjie_feature_config)

        self.view_dist = 15000

        # 如果 ini 里只有：
        # is_direnjie = 1维
        # normal_attack_stage = 3维 one-hot
        # 那么最终维度是 4
        self.one_unit_feature_num = 6
        self.unit_buff_num = 1


        # 狄仁杰大招槽位，Skill 3 一般是 slot_type=3
        self.direnjie_ult_slot_type = 3

        # 1 frame ≈ 33 ms，1s ≈ 30 frame
        self.direnjie_ult_stun_max_frames = 30

        # 如果拿不到 frame_no，则默认一次 process 对应 1 step = 6 frame
        self.default_frame_delta = 6
        self.last_frame_no = None

        # 记录双方狄仁杰大招上一次命中英雄次数
        self.last_main_direnjie_ult_hit_hero_times = None
        self.last_enemy_direnjie_ult_hit_hero_times = None

        # 记录“自己的大招命中敌方后，敌方剩余眩晕时间”
        self.main_direnjie_ult_hit_stun_left_frames = 0
        self.enemy_direnjie_ult_hit_stun_left_frames = 0

    def get_direnjie_config(self):
        self.config = configparser.ConfigParser()
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "direnjie_feature_config.ini")
        self.config.read(config_path)

        # Get normalized configuration
        # 获取归一化的配置
        self.direnjie_feature_config = []
        for feature, config in self.config["feature_config"].items():
            self.direnjie_feature_config.append(f"{feature}:{config}")

        # Get feature function configuration
        # 获取特征函数的配置
        self.feature_func_map = {}
        for feature, func_name in self.config["feature_functions"].items():
            if hasattr(self, func_name):
                self.feature_func_map[feature] = getattr(self, func_name)
            else:
                raise ValueError(f"Unsupported function: {func_name}")
                
    # 计算当前帧和上一帧的帧号差值
    def _get_frame_delta(self, frame_state):
        current_frame_no = frame_state.get("frame_no", None)

        if current_frame_no is None:
            return self.default_frame_delta

        if self.last_frame_no is None:
            self.last_frame_no = current_frame_no
            return self.default_frame_delta

        frame_delta = current_frame_no - self.last_frame_no
        self.last_frame_no = current_frame_no

        if frame_delta <= 0:
            return self.default_frame_delta

        return frame_delta
    
    
        
    def update_direnjie_ult_hit_stun_timer(self, frame_delta):
        """
        更新狄仁杰大招命中敌方后造成的眩晕剩余时间。

        main_direnjie_ult_hit_stun_left_frames:
            我方狄仁杰大招命中敌方后，敌方剩余眩晕时间。

        enemy_direnjie_ult_hit_stun_left_frames:
            敌方狄仁杰大招命中我方后，我方剩余眩晕时间。
        """

        # 先扣减已有眩晕时间
        self.main_direnjie_ult_hit_stun_left_frames = max(
            0,
            self.main_direnjie_ult_hit_stun_left_frames - frame_delta
        )
        self.enemy_direnjie_ult_hit_stun_left_frames = max(
            0,
            self.enemy_direnjie_ult_hit_stun_left_frames - frame_delta
        )

        # 我方狄仁杰大招命中敌方
        if self.main_hero_direnjie and self.main_hero_info is not None:
            main_ult_skill = self._get_skill_by_slot_type(
                self.main_hero_info,
                self.direnjie_ult_slot_type
            )

            current_hit_times = 0
            if main_ult_skill is not None:
                current_hit_times = main_ult_skill.get("hitHeroTimes", 0)

            if self.last_main_direnjie_ult_hit_hero_times is None:
                self.last_main_direnjie_ult_hit_hero_times = current_hit_times
            else:
                if current_hit_times > self.last_main_direnjie_ult_hit_hero_times:
                    self.main_direnjie_ult_hit_stun_left_frames = self.direnjie_ult_stun_max_frames

                self.last_main_direnjie_ult_hit_hero_times = current_hit_times
        else:
            self.last_main_direnjie_ult_hit_hero_times = None
            self.main_direnjie_ult_hit_stun_left_frames = 0

        # 敌方狄仁杰大招命中我方
        if self.enemy_hero_direnjie and self.enemy_hero_info is not None:
            enemy_ult_skill = self._get_skill_by_slot_type(
                self.enemy_hero_info,
                self.direnjie_ult_slot_type
            )

            current_hit_times = 0
            if enemy_ult_skill is not None:
                current_hit_times = enemy_ult_skill.get("hitHeroTimes", 0)

            if self.last_enemy_direnjie_ult_hit_hero_times is None:
                self.last_enemy_direnjie_ult_hit_hero_times = current_hit_times
            else:
                if current_hit_times > self.last_enemy_direnjie_ult_hit_hero_times:
                    self.enemy_direnjie_ult_hit_stun_left_frames = self.direnjie_ult_stun_max_frames

                self.last_enemy_direnjie_ult_hit_hero_times = current_hit_times
        else:
            self.last_enemy_direnjie_ult_hit_hero_times = None
            self.enemy_direnjie_ult_hit_stun_left_frames = 0

    def process_vec_direnjie(self, frame_state):
        self.generate_hero_info_list(frame_state)
        self.generate_direnjie_info_dict(frame_state)

        frame_delta = self._get_frame_delta(frame_state)
        self.update_direnjie_ult_hit_stun_timer(frame_delta)

        # 生成敌方阵营的狄仁杰私有特征
        enemy_camp_direnjie_vector_feature = self.generate_one_type_direnjie_feature(
            self.enemy_hero_direnjie, self.enemy_hero_info, None
        )

        # 生成我方阵营的狄仁杰私有特征
        main_camp_direnjie_vector_feature = self.generate_one_type_direnjie_feature(
            self.main_hero_direnjie, self.main_hero_info, None
        )

        return main_camp_direnjie_vector_feature, enemy_camp_direnjie_vector_feature

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

    def generate_direnjie_info_dict(self, frame_state):
        if self.main_hero_info is not None and self.main_hero_info["config_id"] == 133:
            self.main_hero_direnjie = True
        else:
            self.main_hero_direnjie = False

        if self.enemy_hero_info is not None and self.enemy_hero_info["config_id"] == 133:
            self.enemy_hero_direnjie = True
        else:
            self.enemy_hero_direnjie = False

    def generate_one_type_direnjie_feature(self, is_direnjie, hero, camp):
        vector_feature = []
        num_direnjies_considered = 0

        def process_direnjie(direnjie):
            nonlocal num_direnjies_considered

            # Generate each specific feature through feature_func_map
            # 通过 feature_func_map 生成每个具体特征
            for feature_name, feature_func in self.feature_func_map.items():
                value = []
                self.feature_func_map[feature_name](direnjie, value)

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

            num_direnjies_considered += 1

        if is_direnjie:
            process_direnjie(hero)

        if num_direnjies_considered < self.unit_buff_num:
            self.no_direnjie_feature(vector_feature, num_direnjies_considered)

        return vector_feature

    def no_direnjie_feature(self, vector_feature, num_direnjies_considered):
        for _ in range((self.unit_buff_num - num_direnjies_considered) * self.one_unit_feature_num):
            vector_feature.append(0)

    def _get_skill_by_slot_type(self, hero, slot_type):
        skill_states = hero.get("skill_state", {}).get("slot_states", [])

        for skill in skill_states:
            if skill.get("slot_type") == slot_type:
                return skill

        return None

    def is_direnjie(self, hero, vector_feature):
        vector_feature.append(1)

    def normal_attack_stage(self, hero, vector_feature):
        """
        狄仁杰普通攻击阶段。

        从技能槽 slot_type=0 判断当前普攻阶段。
        当前已观察到狄仁杰普攻相关 configId 包括：
            13300
            13301
            13302

        这里函数只输出 configId，具体 one-hot 交给 ini。
        """
        skill = self._get_skill_by_slot_type(hero, 0)

        if skill is None:
            value = 0
        else:
            value = skill.get("configId", 0)

        vector_feature.append(value)

    
    def ult_hit_stun_left_time(self, hero, vector_feature):
        """
        当前这个狄仁杰自己的大招命中敌方后，敌方剩余眩晕时间。

        如果 hero["camp"] == self.main_camp:
            返回我方狄仁杰大招造成的眩晕剩余时间。

        否则：
            返回敌方狄仁杰大招造成的眩晕剩余时间。
        """

        if hero is None:
            value = 0
        elif hero.get("camp") == self.main_camp:
            value = self.main_direnjie_ult_hit_stun_left_frames
        else:
            value = self.enemy_direnjie_ult_hit_stun_left_frames

        vector_feature.append(value)