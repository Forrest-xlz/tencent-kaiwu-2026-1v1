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


class GameProcess:
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
        self.get_game_config()
        self.map_feature_to_norm = self.normalizer.parse_config(self.game_feature_config)
        self.view_dist = 15000

        # 当前配置维度：
        # game_time: 5维 one-hot
        # next_soldier_spawn_time: 1维
        # river_spirit_respawn_time: 1维
        # main_cake_respawn_time: 1维
        # enemy_cake_respawn_time: 1维
        # 总计 9 维
        self.one_unit_feature_num = 9
        self.unit_buff_num = 1

        # =========================
        # 通用时间配置
        # =========================

        # 1 frame ≈ 33 ms
        self.frame_ms = 33.0

        # =========================
        # 血包刷新时间
        # =========================

        # 血包首次刷新：30s
        self.cake_first_spawn_time_s = 30.0

        # 血包吃掉后刷新：60s
        self.cake_respawn_interval_s = 60.0

        # 当前帧血包信息
        self.main_cake_info = None
        self.enemy_cake_info = None

        # 上一帧血包是否存在
        self.last_main_cake_exist = False
        self.last_enemy_cake_exist = False

        # 最近一次血包被吃掉的时间
        self.main_cake_last_eaten_time_s = None
        self.enemy_cake_last_eaten_time_s = None

        # 用于检测新一局 frame_no 重置
        self.last_frame_no_for_cake = None

        # =========================
        # 河道之灵刷新时间
        # =========================

        # 首次刷新：30s
        self.river_spirit_first_spawn_time_s = 30.0

        # 击杀后刷新：60s
        self.river_spirit_respawn_interval_s = 60.0

        # 5分钟后停止刷新：300s
        self.river_spirit_stop_spawn_time_s = 300.0

        # 记录最近一次河道之灵死亡时间，单位秒
        self.river_spirit_last_death_time_s = None

        # 记录最近一次河道之灵死亡帧，防止同一帧重复处理
        self.river_spirit_last_death_frame_no = None

        # 用于检测新一局 frame_no 重置
        self.last_frame_no_for_river_spirit = None

    def get_game_config(self):
        self.config = configparser.ConfigParser()
        current_dir = os.path.dirname(__file__)
        config_path = os.path.join(current_dir, "game_feature_config.ini")
        self.config.read(config_path)

        # Get normalized configuration
        # 获取归一化的配置
        self.game_feature_config = []
        for feature, config in self.config["feature_config"].items():
            self.game_feature_config.append(f"{feature}:{config}")

        # Get feature function configuration
        # 获取特征函数的配置
        self.feature_func_map = {}
        for feature, func_name in self.config["feature_functions"].items():
            if hasattr(self, func_name):
                self.feature_func_map[feature] = getattr(self, func_name)
            else:
                raise ValueError(f"Unsupported function: {func_name}")

    def _get_current_time_s(self, frame_state):
        """
        根据 frame_no 计算当前局内时间，单位秒。
        """
        frame_no = frame_state.get("frame_no", 0)
        return frame_no * self.frame_ms / 1000.0

    def _is_neutral_monster_death(self, dead_action):
        """
        判断当前 dead_action 是否是河道之灵/中立野怪死亡事件。

        参考奖励函数 monster_kill / monster_first_hit 的写法：
            actor_type = 1
            sub_type = 0
            camp = 0
        """
        death = dead_action.get("death", {})

        return (
            death.get("actor_type") == 1
            and death.get("sub_type") == 0
            and death.get("camp") == 0
        )

    def _reset_river_spirit_respawn_state(self):
        """
        新一局开始时重置河道之灵刷新状态。
        """
        self.river_spirit_last_death_time_s = None
        self.river_spirit_last_death_frame_no = None

    def update_river_spirit_respawn_state(self, frame_state):
        """
        根据 frame_action.dead_action 更新河道之灵死亡时间。

        只要检测到中立野怪死亡事件，就记录当前死亡时间。
        """
        frame_no = frame_state.get("frame_no", 0)

        # frame_no 变小，说明进入新一局，清空上一局缓存
        if (
            self.last_frame_no_for_river_spirit is not None
            and frame_no < self.last_frame_no_for_river_spirit
        ):
            self._reset_river_spirit_respawn_state()

        self.last_frame_no_for_river_spirit = frame_no

        frame_action = frame_state.get("frame_action", {})
        dead_actions = frame_action.get("dead_action", [])

        for dead_action in dead_actions:
            if not self._is_neutral_monster_death(dead_action):
                continue

            # 防止同一帧重复记录
            if self.river_spirit_last_death_frame_no == frame_no:
                continue

            self.river_spirit_last_death_frame_no = frame_no
            self.river_spirit_last_death_time_s = self._get_current_time_s(frame_state)
            break

    def _reset_cake_respawn_state(self):
        """
        新一局开始时重置血包刷新状态。
        """
        self.main_cake_info = None
        self.enemy_cake_info = None

        self.last_main_cake_exist = False
        self.last_enemy_cake_exist = False

        self.main_cake_last_eaten_time_s = None
        self.enemy_cake_last_eaten_time_s = None

    def update_cake_respawn_state(self, frame_state):
        """
        更新血包刷新计时。

        重点：
            只有上一帧存在、当前帧不存在，才认为血包被吃掉。
            当前帧不存在但上一帧也不存在，不会重新计时。
        """
        frame_no = frame_state.get("frame_no", 0)

        # 新一局 frame_no 重置
        if (
            self.last_frame_no_for_cake is not None
            and frame_no < self.last_frame_no_for_cake
        ):
            self._reset_cake_respawn_state()

        self.last_frame_no_for_cake = frame_no

        current_time_s = self._get_current_time_s(frame_state)

        main_cake_exist = self.main_cake_info is not None
        enemy_cake_exist = self.enemy_cake_info is not None

        self._update_one_cake_respawn_state(
            cake_exist=main_cake_exist,
            last_exist_attr="last_main_cake_exist",
            eaten_time_attr="main_cake_last_eaten_time_s",
            current_time_s=current_time_s,
        )

        self._update_one_cake_respawn_state(
            cake_exist=enemy_cake_exist,
            last_exist_attr="last_enemy_cake_exist",
            eaten_time_attr="enemy_cake_last_eaten_time_s",
            current_time_s=current_time_s,
        )

    def _update_one_cake_respawn_state(
        self,
        cake_exist,
        last_exist_attr,
        eaten_time_attr,
        current_time_s,
    ):
        """
        单个阵营血包状态更新。

        只在：
            last_exist == True and cake_exist == False
        时记录 eaten_time。
        """
        last_exist = getattr(self, last_exist_attr)

        # 30s 前血包还没首次刷新，不记录吃掉事件
        if current_time_s < self.cake_first_spawn_time_s:
            setattr(self, last_exist_attr, False)
            setattr(self, eaten_time_attr, None)
            return

        # 核心逻辑：上一帧存在，这一帧不存在 -> 被吃了
        if last_exist and not cake_exist:
            setattr(self, eaten_time_attr, current_time_s)

        # 无论是否刷新计时，都要更新上一帧存在状态
        setattr(self, last_exist_attr, cake_exist)

    def generate_cakes_info_dict(self, frame_state):
        """
        解析当前帧双方血包。

        墨家机关道血包固定位置：
            x < 0 and z < 0 -> camp 1 血包
            x > 0 and z > 0 -> camp 2 血包
        """
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

            if cake_camp == self.main_camp_id:
                self.main_cake_info = cake_info
            else:
                self.enemy_cake_info = cake_info

    def process_vec_game(self, frame_state):
        self.generate_organ_info_dict(frame_state)
        self.generate_hero_info_list(frame_state)

        self.generate_cakes_info_dict(frame_state)
        self.update_cake_respawn_state(frame_state)

        self.update_river_spirit_respawn_state(frame_state)

        game_vector_feature = self.generate_game_feature(frame_state, None)

        return game_vector_feature

    def generate_hero_info_list(self, frame_state):
        self.main_camp_hero_dict.clear()
        self.enemy_camp_hero_dict.clear()
        for hero in frame_state["hero_states"]:
            if hero["camp"] == self.main_camp:
                self.main_camp_hero_dict[hero["config_id"]] = hero
                self.main_hero_info = hero
            else:
                self.enemy_camp_hero_dict[hero["config_id"]] = hero

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

    def generate_game_feature(self, frame_state, camp):
        vector_feature = []
        num_game_considered = 0

        def process_game(frame_state=frame_state, camp=camp):
            nonlocal num_game_considered
            # Generate each specific feature through feature_func_map
            # 通过 feature_func_map 生成每个具体特征
            for feature_name, feature_func in self.feature_func_map.items():
                value = []
                self.feature_func_map[feature_name](frame_state, value)
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
            num_game_considered += 1

        process_game(frame_state=frame_state, camp=camp)

        if num_game_considered < self.unit_buff_num:
            self.no_game_feature(vector_feature, num_game_considered)
        return vector_feature

    def no_game_feature(self, vector_feature, num_game_considered):
        for _ in range((self.unit_buff_num - num_game_considered) * self.one_unit_feature_num):
            vector_feature.append(0)

    def game_time(self, frame_state, vector_feature):
        frame_no = frame_state["frame_no"]
        value = 0
        if frame_no < 1818 * 2:
            value = 0
        elif frame_no < 1818 * 4:
            value = 1
        elif frame_no < 1818 * 6:
            value = 2
        elif frame_no < 1818 * 8:
            value = 3
        else:
            value = 4

        vector_feature.append(value)

    def next_soldier_spawn_time(self, frame_state, vector_feature):
        """
        距离下一次出兵的时间，单位：秒。

        规则：
            第一次出兵：开局 15s
            后续出兵：每隔 27s

        环境时间：
            1 frame ≈ 33 ms
            1 step = 6 frame

        这里直接使用 frame_state["frame_no"] 计算当前仿真秒数。
        输出范围：
            [0, 27]
        """
        frame_no = frame_state["frame_no"]

        frame_ms = 33.0
        current_time_s = frame_no * frame_ms / 1000.0

        first_spawn_time_s = 15.0
        spawn_interval_s = 27.0

        if current_time_s < first_spawn_time_s:
            value = first_spawn_time_s - current_time_s
        else:
            elapsed_after_first_spawn = current_time_s - first_spawn_time_s
            remainder = elapsed_after_first_spawn % spawn_interval_s

            if remainder == 0:
                value = 0.0
            else:
                value = spawn_interval_s - remainder

        vector_feature.append(value)

    def river_spirit_respawn_time(self, frame_state, vector_feature):
        """
        河道之灵剩余刷新时间，单位秒。

        输出原始值范围：
            0 ~ 60

        归一化建议：
            river_spirit_respawn_time = min_max:0:60

        规则：
            1. 开局 30s 首次刷新；
            2. 30s 后如果没有死亡事件，认为河道之灵活着，输出 0；
            3. 死亡后 60s 刷新；
            4. 如果死亡时间 + 60s > 300s，说明不会再刷新，输出 60；
            5. 超过 5 分钟后，没有刷新出来的野怪也不会再刷新，输出 60。
        """
        current_time_s = self._get_current_time_s(frame_state)

        first_spawn_time_s = self.river_spirit_first_spawn_time_s
        respawn_interval_s = self.river_spirit_respawn_interval_s
        stop_spawn_time_s = self.river_spirit_stop_spawn_time_s

        # 1. 首次刷新前：距离 30s 首刷还剩多少
        if current_time_s < first_spawn_time_s:
            value = first_spawn_time_s - current_time_s

        # 2. 没有发生过死亡事件
        #    30s 之后认为河道之灵已经在场，剩余刷新时间为 0
        elif self.river_spirit_last_death_time_s is None:
            value = 0.0

        else:
            next_spawn_time_s = (
                self.river_spirit_last_death_time_s
                + respawn_interval_s
            )

            # 3. 如果下一次刷新时间超过 5 分钟，就不会再刷新
            #    例如 4分10秒死亡，刷新时间是 5分10秒，超过 5分钟，所以输出 60 -> 归一化为 1
            if next_spawn_time_s > stop_spawn_time_s:
                value = respawn_interval_s

            # 4. 还没到下一次刷新时间：输出剩余秒数
            elif current_time_s < next_spawn_time_s:
                value = next_spawn_time_s - current_time_s

            # 5. 已经刷新出来：野怪活着，剩余刷新时间为 0
            else:
                value = 0.0

        # 保底限制到 [0, 60]
        value = max(0.0, min(respawn_interval_s, value))
        vector_feature.append(value)

    def _get_cake_respawn_left_time(self, cake_info, eaten_time_s, frame_state):
        """
        返回血包剩余刷新时间，单位秒。

        规则：
            30s 前：返回 30 - 当前时间；
            血包存在：返回 0；
            血包不存在且有明确的“存在 -> 不存在”事件：返回倒计时；
            血包不存在但没有明确吃掉事件：返回 0。

        注意：
            倒计时结束后如果仍未观察到血包，不会重新刷 60s；
            只有重新观察到血包存在，之后再出现“存在 -> 不存在”，才会重新计时。
        """
        current_time_s = self._get_current_time_s(frame_state)

        # 首次刷新前
        if current_time_s < self.cake_first_spawn_time_s:
            value = self.cake_first_spawn_time_s - current_time_s

        # 当前血包存在
        elif cake_info is not None:
            value = 0.0

        # 当前血包不存在，但有明确的“上一帧存在 -> 当前帧不存在”事件
        elif eaten_time_s is not None:
            next_spawn_time_s = eaten_time_s + self.cake_respawn_interval_s
            value = max(0.0, next_spawn_time_s - current_time_s)

        # 当前血包不存在，但没有明确吃掉事件，不强行推断
        else:
            value = 0.0

        value = max(0.0, min(self.cake_respawn_interval_s, value))
        return value

    def main_cake_respawn_time(self, frame_state, vector_feature):
        """
        我方阵营血包剩余刷新时间，单位秒。
        """
        value = self._get_cake_respawn_left_time(
            cake_info=self.main_cake_info,
            eaten_time_s=self.main_cake_last_eaten_time_s,
            frame_state=frame_state,
        )
        vector_feature.append(value)

    def enemy_cake_respawn_time(self, frame_state, vector_feature):
        """
        敌方阵营血包剩余刷新时间，单位秒。
        """
        value = self._get_cake_respawn_left_time(
            cake_info=self.enemy_cake_info,
            eaten_time_s=self.enemy_cake_last_eaten_time_s,
            frame_state=frame_state,
        )
        vector_feature.append(value)
