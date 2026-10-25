#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2025 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import math
from agent_ppo.conf.conf import GameConfig



class RewardStruct:
    def __init__(self, m_weight=0.0):
        self.cur_frame_value = 0.0
        self.last_frame_value = 0.0
        self.value = 0.0
        self.weight = m_weight
        self.min_value = -1
        self.is_first_arrive_center = True


def linear_schedule_value(start_value, end_value, episode_cnt, start_episode, end_episode):
    """
    根据 episode_cnt 线性插值。
    episode <= start_episode: start_value
    episode >= end_episode: end_value
    中间线性变化。
    """
    if end_episode <= start_episode:
        return end_value if episode_cnt >= end_episode else start_value

    progress = (episode_cnt - start_episode) / float(end_episode - start_episode)
    progress = max(0.0, min(1.0, progress))

    return start_value + (end_value - start_value) * progress


def get_reward_weight_by_episode(reward_name, base_weight, episode_cnt):
    """
    返回当前 episode 下某个 reward_name 应该使用的权重。
    如果没有配置调度，则返回 base_weight。
    """
    if not getattr(GameConfig, "REWARD_SCHEDULE_ENABLE", False):
        return base_weight

    schedule_dict = getattr(GameConfig, "REWARD_SCHEDULE_DICT", {})
    if reward_name not in schedule_dict:
        return base_weight

    schedule_cfg = schedule_dict[reward_name]

    start_episode = getattr(GameConfig, "REWARD_SCHEDULE_START_EPISODE", 1)
    end_episode = getattr(GameConfig, "REWARD_SCHEDULE_END_EPISODE", start_episode)

    start_value = float(schedule_cfg.get("start", base_weight))
    end_value = float(schedule_cfg.get("end", base_weight))

    return linear_schedule_value(
        start_value=start_value,
        end_value=end_value,
        episode_cnt=episode_cnt,
        start_episode=start_episode,
        end_episode=end_episode,
    )


def get_scheduled_reward_weight_dict(episode_cnt):
    weight_dict = {}

    for key, weight in GameConfig.REWARD_WEIGHT_DICT.items():
        weight_dict[key] = get_reward_weight_by_episode(
            reward_name=key,
            base_weight=weight,
            episode_cnt=episode_cnt,
        )

    return weight_dict


def init_calc_frame_map(episode_cnt=None):
    calc_frame_map = {}

    if episode_cnt is None:
        episode_cnt = 0

    for key, weight in GameConfig.REWARD_WEIGHT_DICT.items():
        scheduled_weight = get_reward_weight_by_episode(
            reward_name=key,
            base_weight=weight,
            episode_cnt=episode_cnt,
        )
        calc_frame_map[key] = RewardStruct(scheduled_weight)

    return calc_frame_map


def get_time_scale_arg_by_episode(episode_cnt):
    """
    返回当前 episode 下实际使用的 time_scale_arg。
    如果未启用调度，则返回 GameConfig.TIME_SCALE_ARG。
    """
    base_value = float(getattr(GameConfig, "TIME_SCALE_ARG", 0))

    if not getattr(GameConfig, "TIME_SCALE_SCHEDULE_ENABLE", False):
        return base_value

    start_episode = getattr(GameConfig, "TIME_SCALE_SCHEDULE_START_EPISODE", 1)
    end_episode = getattr(GameConfig, "TIME_SCALE_SCHEDULE_END_EPISODE", start_episode)

    start_value = float(getattr(GameConfig, "TIME_SCALE_SCHEDULE_START_VALUE", base_value))
    end_value = float(getattr(GameConfig, "TIME_SCALE_SCHEDULE_END_VALUE", base_value))

    time_scale_arg = linear_schedule_value(
        start_value=start_value,
        end_value=end_value,
        episode_cnt=episode_cnt,
        start_episode=start_episode,
        end_episode=end_episode,
    )

    # 小于等于 0 时，你的 result() 里会跳过时间衰减。
    # 这里保留这个语义。
    return max(0.0, time_scale_arg)

class GameRewardManager:
    def __init__(self, main_hero_runtime_id, episode_cnt):
        self.episode_cnt = episode_cnt
        self.main_hero_player_id = main_hero_runtime_id

        # 当前奖励权重字典，用于日志和 monitor
        self.current_reward_weight_dict = get_scheduled_reward_weight_dict(self.episode_cnt)

        # 当前时间衰减参数，用于日志和 monitor
        self.time_scale_arg = get_time_scale_arg_by_episode(self.episode_cnt)
        self.current_time_scale_arg = self.time_scale_arg

        self.main_hero_camp = -1
        self.main_hero_hp = -1
        self.main_hero_organ_hp = -1

        self.m_reward_value = {}
        self.m_last_frame_no = -1

        self.m_cur_calc_frame_map = init_calc_frame_map(self.episode_cnt)
        self.m_main_calc_frame_map = init_calc_frame_map(self.episode_cnt)
        self.m_enemy_calc_frame_map = init_calc_frame_map(self.episode_cnt)

        self.m_init_calc_frame_map = {}

        self.m_main_hero_config_id = -1
        self.m_each_level_max_exp = {}

        self.monster_first_hit_rewarded = False
        self.current_soldier_wave_idx = -1
        self.wave_hit_enemy_hero_rewarded = False

        # =========================
        # 狄仁杰大招奖励状态
        # =========================
        self.direnjie_config_id = 133
        self.direnjie_ult_slot_type = 3

        # 1 frame ≈ 33 ms，2s ≈ 60 frame
        self.direnjie_ult_hit_window_frames = 60
        self.direnjie_ult_kill_window_frames = 150

        # 上一帧我方狄仁杰大招累计释放次数 / 累计命中英雄次数
        self.last_direnjie_ult_used_times = None
        self.last_direnjie_ult_hit_hero_times = None

        # 上一帧我方英雄击杀次数。
        # 用 kill_cnt 增量判断命中大招后的 2 秒内是否完成击杀，
        # 不再依赖 frame_action.dead_action。
        self.last_main_hero_kill_cnt = None

        # 当前待判定的大招释放事件：记录释放帧
        self.pending_direnjie_ult_release_frame_no = None

        # 当前已命中、等待击杀判定的大招事件：记录命中帧
        self.pending_direnjie_ult_hit_frame_no = None

        # 本帧两个奖励项的原始值
        # set_cur_calc_frame_vec 会按 reward_name 遍历，
        # 这里用 frame_no 缓存避免同一帧重复更新跨帧状态。
        self.direnjie_ult_kill_after_hit_value = 0.0
        self.direnjie_ult_miss_value = 0.0

        # 狄仁杰大招命中且我方防御塔正在攻击敌方英雄
        self.direnjie_ult_hit_under_tower_value = 0.0

        # 狄仁杰站在敌方塔攻击范围外，大招命中敌方塔内英雄
        self.direnjie_ult_tower_poke_value = 0.0

        # 狄仁杰大招命中后 5 秒内击杀，并且击杀时我方 4 级、敌方 3 级
        self.direnjie_ult_level4_kill_value = 0.0

        self.direnjie_ult_reward_last_update_frame_no = None

        # =========================
        # 狂暴召唤师技能奖励状态
        # =========================
        # 召唤师技能槽位，一般 slot_type=6
        self.summon_skill_slot_type = 6

        # 狂暴召唤师技能 configId
        self.rage_skill_config_id = 80110

        # 1 frame ≈ 33 ms，8s ≈ 240 frame
        self.rage_kill_window_frames = 240

        # 上一帧狂暴累计使用次数
        self.last_rage_used_times = None

        # 上一帧我方英雄击杀次数
        self.last_rage_main_hero_kill_cnt = None

        # 当前待判定的狂暴使用事件
        self.pending_rage_use_frame_no = None

        # 本帧两个奖励项的原始值
        self.rage_kill_after_use_value = 0.0
        self.rage_no_kill_after_use_value = 0.0
        self.rage_use_penalty_value = 0.0

        # 防止同一帧两个 rage reward item 重复更新跨帧状态
        self.rage_reward_last_update_frame_no = None


        # =========================
        # 草丛诱敌击杀奖励状态
        # =========================

        # 1 frame ≈ 33 ms，3s ≈ 90 frame
        self.grass_ambush_kill_window_frames = 150

        # 当前草丛诱敌奖励所属兵线周期
        self.current_grass_ambush_wave_idx = -1

        # 当前兵线周期是否已经使用过一次草丛诱敌尝试
        self.grass_ambush_attempt_used = False

        # 上一帧是否处于状态 A：
        # 我方在草丛中，并且敌方看不到我方
        self.last_grass_ambush_state_a = False

        # 打破状态 A 的帧号
        self.pending_grass_ambush_break_frame_no = None

        # 上一帧我方击杀数
        self.last_grass_ambush_main_hero_kill_cnt = None

        # 当前帧草丛诱敌击杀奖励原始值
        self.grass_ambush_kill_value = 0.0

        # 防止同一帧重复更新状态
        self.grass_ambush_last_update_frame_no = None

        # =========================
        # 恢复召唤师技能使用惩罚状态
        # =========================
        # 恢复技能槽位：slot_type = 5
        self.recover_skill_slot_type = 5

        # 上一帧恢复累计使用次数
        self.last_recover_used_times = None

        # 当前帧恢复使用惩罚原始值
        self.recover_use_penalty_value = 0.0

        # 防止同一帧重复更新状态
        self.recover_reward_last_update_frame_no = None


    def _get_camp_visible(self, hero, camp):
        """
        判断 hero 是否对指定 camp 可见。

        camp:
            1 = 蓝方
            2 = 红方

        camp_visible:
            [0] = 蓝方是否可见
            [1] = 红方是否可见
        """
        camp_visible = hero.get("camp_visible", [])
        idx = camp - 1

        if idx < 0 or idx >= len(camp_visible):
            return False

        return bool(camp_visible[idx])

    def _clear_pending_grass_ambush_reward_state(self):
        self.pending_grass_ambush_break_frame_no = None

    def update_grass_ambush_reward_state(self, frame_data, main_hero, enemy_hero):
        """
        草丛诱敌击杀奖励。

        状态 A:
            我方英雄在草丛中，并且敌方看不到我方英雄。

        尝试规则：
            1. 每一轮兵线周期最多使用一次草丛诱敌尝试；
            2. 本轮第一次进入状态 A 时，认为已经使用本轮尝试；
            3. 之后如果从状态 A 变为非状态 A，认为打破状态 A；
            4. 打破状态 A 后 3 秒内，如果 main_hero["kill_cnt"] 增加，则 grass_ambush_kill = +1；
            5. 如果 3 秒内没有击杀，不给奖励，但本轮尝试已经消耗，不能再重复尝试。
        """
        frame_no = frame_data["frame_no"]

        # 同一帧只更新一次跨帧状态
        if self.grass_ambush_last_update_frame_no == frame_no:
            return

        self.grass_ambush_last_update_frame_no = frame_no
        self.grass_ambush_kill_value = 0.0

        wave_idx = self.get_soldier_wave_idx(frame_no)

        # 第一波兵还没出生，不处理这个奖励
        if wave_idx < 0:
            self.last_grass_ambush_state_a = False
            self.last_grass_ambush_main_hero_kill_cnt = main_hero.get("kill_cnt", 0)
            self.grass_ambush_attempt_used = False
            self._clear_pending_grass_ambush_reward_state()
            return

        # 进入新的兵线周期，刷新本轮尝试
        if wave_idx != self.current_grass_ambush_wave_idx:
            self.current_grass_ambush_wave_idx = wave_idx
            self.grass_ambush_attempt_used = False
            self.last_grass_ambush_state_a = False
            self._clear_pending_grass_ambush_reward_state()

        current_kill_cnt = main_hero.get("kill_cnt", 0)

        # 第一次进入，只建立 kill_cnt 基准
        if self.last_grass_ambush_main_hero_kill_cnt is None:
            self.last_grass_ambush_main_hero_kill_cnt = current_kill_cnt
            self.last_grass_ambush_state_a = False
            return

        # 新一局或状态重置
        if current_kill_cnt < self.last_grass_ambush_main_hero_kill_cnt:
            self.last_grass_ambush_main_hero_kill_cnt = current_kill_cnt
            self.last_grass_ambush_state_a = False
            self.grass_ambush_attempt_used = False
            self._clear_pending_grass_ambush_reward_state()
            return

        kill_increased = current_kill_cnt > self.last_grass_ambush_main_hero_kill_cnt

        main_camp = main_hero.get("camp", -1)
        enemy_camp = enemy_hero.get("camp", -1)

        # 敌方是否能看到我方英雄
        enemy_can_see_main = self._get_camp_visible(main_hero, enemy_camp)

        # 状态 A：我方在草丛，且敌方看不到我方
        current_state_a = (
            bool(main_hero.get("is_in_grass", False))
            and not enemy_can_see_main
        )

        # 本轮第一次进入状态 A，消耗本轮草丛诱敌尝试
        if current_state_a and not self.grass_ambush_attempt_used:
            self.grass_ambush_attempt_used = True

        # 只有本轮已经使用过尝试，才允许后续判断“打破状态 A”
        broke_state_a = (
            self.grass_ambush_attempt_used
            and self.last_grass_ambush_state_a
            and not current_state_a
        )

        if broke_state_a and self.pending_grass_ambush_break_frame_no is None:
            self.pending_grass_ambush_break_frame_no = frame_no

        # 如果已经打破状态 A，检查 3 秒内是否击杀
        if self.pending_grass_ambush_break_frame_no is not None:
            elapsed_frames = frame_no - self.pending_grass_ambush_break_frame_no

            if elapsed_frames <= self.grass_ambush_kill_window_frames:
                if kill_increased:
                    self.grass_ambush_kill_value = 1.0
                    self._clear_pending_grass_ambush_reward_state()
            else:
                # 超过 3 秒没有击杀，本轮尝试失败。
                # 注意：grass_ambush_attempt_used 仍然保持 True，
                # 所以本轮兵线不能再次尝试。
                self._clear_pending_grass_ambush_reward_state()

        self.last_grass_ambush_state_a = current_state_a
        self.last_grass_ambush_main_hero_kill_cnt = current_kill_cnt

    # Used to initialize the maximum experience value for each agent level
    # 用于初始化智能体各个等级的最大经验值
    def init_max_exp_of_each_hero(self):
        self.m_each_level_max_exp.clear()
        self.m_each_level_max_exp[1] = 160
        self.m_each_level_max_exp[2] = 298
        self.m_each_level_max_exp[3] = 446
        self.m_each_level_max_exp[4] = 524
        self.m_each_level_max_exp[5] = 613
        self.m_each_level_max_exp[6] = 713
        self.m_each_level_max_exp[7] = 825
        self.m_each_level_max_exp[8] = 950
        self.m_each_level_max_exp[9] = 1088
        self.m_each_level_max_exp[10] = 1240
        self.m_each_level_max_exp[11] = 1406
        self.m_each_level_max_exp[12] = 1585
        self.m_each_level_max_exp[13] = 1778
        self.m_each_level_max_exp[14] = 1984

    def get_soldier_wave_idx(self, frame_no):
        """
        根据 frame_no 计算当前处于第几波小兵周期。
        第一次出兵：15s
        后续出兵间隔：27s
        1 frame ≈ 33ms

        返回：
            -1 表示第一波兵还没出生
            0 表示第一波兵周期
            1 表示第二波兵周期
            ...
        """
        frame_ms = 33.0
        current_time_s = frame_no * frame_ms / 1000.0

        first_spawn_time_s = 15.0
        spawn_interval_s = 27.0

        if current_time_s < first_spawn_time_s:
            return -1

        return int((current_time_s - first_spawn_time_s) // spawn_interval_s)

    def _get_skill_by_slot_type(self, hero, slot_type):
        """
        从 hero["skill_state"]["slot_states"] 中取指定 slot_type 的技能槽。
        找不到时返回 None，避免奖励计算因为技能槽缺失而报错。
        """
        skill_states = hero.get("skill_state", {}).get("slot_states", [])

        for skill in skill_states:
            if skill.get("slot_type") == slot_type:
                return skill

        return None

    def _get_xz_distance(self, obj_a, obj_b):
        """
        计算两个对象在 x-z 平面上的距离。
        对象需要包含 location.x 和 location.z。
        """
        if obj_a is None or obj_b is None:
            return None

        loc_a = obj_a.get("location", {})
        loc_b = obj_b.get("location", {})

        if not loc_a or not loc_b:
            return None

        try:
            pos_a = (loc_a["x"], loc_a["z"])
            pos_b = (loc_b["x"], loc_b["z"])
        except KeyError:
            return None

        return math.dist(pos_a, pos_b)

    def _is_direnjie_enemy_tower_poke_scene(self, main_hero, enemy_hero, enemy_tower):
        """
        判断是否为“狄仁杰塔外大招消耗塔内敌人”场景。

        条件：
            1. 敌方防御塔存在且存活；
            2. 敌方防御塔 attack_range 有效；
            3. 我方狄仁杰在敌方塔攻击范围外；
            4. 敌方英雄在敌方塔攻击范围内。
        """
        if main_hero is None or enemy_hero is None or enemy_tower is None:
            return False

        if enemy_tower.get("hp", 0) <= 0:
            return False

        tower_range = float(enemy_tower.get("attack_range", 0))
        if tower_range <= 0:
            return False

        main_dist = self._get_xz_distance(main_hero, enemy_tower)
        enemy_dist = self._get_xz_distance(enemy_hero, enemy_tower)

        if main_dist is None or enemy_dist is None:
            return False

        # 安全边距：> 0 时要求我方距离敌方塔更远，避免贴着塔边界蹭奖励。
        safe_margin = float(getattr(GameConfig, "DIRENJIE_ULT_TOWER_POKE_SAFE_MARGIN", 0.0))

        main_outside_enemy_tower = main_dist > tower_range + safe_margin
        enemy_inside_enemy_tower = enemy_dist <= tower_range

        return main_outside_enemy_tower and enemy_inside_enemy_tower

    def _clear_pending_direnjie_ult_reward_state(self):
        """
        清空当前待判定的狄仁杰大招事件。
        不清空 last_direnjie_ult_used_times / last_direnjie_ult_hit_hero_times，
        因为它们是累计计数的上一帧基准。
        """
        self.pending_direnjie_ult_release_frame_no = None
        self.pending_direnjie_ult_hit_frame_no = None

    def update_direnjie_ult_reward_state(self, frame_data, main_hero, enemy_hero=None, main_tower=None, enemy_tower=None):
        """
        更新狄仁杰大招相关奖励状态。

        输出到：
            self.direnjie_ult_kill_after_hit_value
            self.direnjie_ult_miss_value
            self.direnjie_ult_hit_under_tower_value
            self.direnjie_ult_tower_poke_value
            self.direnjie_ult_level4_kill_value

        奖励规则：
            1. 我方不是狄仁杰：两个奖励都是 0；
            2. usedTimes 增加：记录大招释放帧；
            3. 释放后 2 秒内 hitHeroTimes 增加：认为大招命中；
            4. 命中后 5 秒内 main_hero["kill_cnt"] 增加：direnjie_ult_kill_after_hit = +1；
            5. 如果击杀发生时我方 4 级、敌方 3 级：direnjie_ult_level4_kill = +1；
            6. 释放后 2 秒内没有命中：direnjie_ult_miss = -1。
        """
        frame_no = frame_data["frame_no"]

        # 同一帧可能先后计算 direnjie_ult_kill_after_hit 和 direnjie_ult_miss，
        # 只允许第一次进入时更新跨帧状态。
        if self.direnjie_ult_reward_last_update_frame_no == frame_no:
            return

        self.direnjie_ult_reward_last_update_frame_no = frame_no
        self.direnjie_ult_kill_after_hit_value = 0.0
        self.direnjie_ult_miss_value = 0.0
        self.direnjie_ult_hit_under_tower_value = 0.0
        self.direnjie_ult_tower_poke_value = 0.0
        self.direnjie_ult_level4_kill_value = 0.0

        # 只处理我方狄仁杰。
        if main_hero.get("config_id") != self.direnjie_config_id:
            self.last_direnjie_ult_used_times = None
            self.last_direnjie_ult_hit_hero_times = None
            self.last_main_hero_kill_cnt = None
            self._clear_pending_direnjie_ult_reward_state()
            return

        ult_skill = self._get_skill_by_slot_type(
            main_hero,
            self.direnjie_ult_slot_type,
        )

        if ult_skill is None:
            self.last_direnjie_ult_used_times = None
            self.last_direnjie_ult_hit_hero_times = None
            self.last_main_hero_kill_cnt = None
            self._clear_pending_direnjie_ult_reward_state()
            return

        current_used_times = ult_skill.get("usedTimes", 0)
        current_hit_hero_times = ult_skill.get("hitHeroTimes", 0)
        current_kill_cnt = main_hero.get("kill_cnt", 0)

        # 第一次进入时只建立基准，不触发奖励/惩罚。
        if self.last_direnjie_ult_used_times is None:
            self.last_direnjie_ult_used_times = current_used_times
            self.last_direnjie_ult_hit_hero_times = current_hit_hero_times
            self.last_main_hero_kill_cnt = current_kill_cnt
            return

        # 如果累计计数变小，通常说明新一局或状态重置。
        if (
            current_used_times < self.last_direnjie_ult_used_times
            or current_hit_hero_times < self.last_direnjie_ult_hit_hero_times
            or (
                self.last_main_hero_kill_cnt is not None
                and current_kill_cnt < self.last_main_hero_kill_cnt
            )
        ):
            self.last_direnjie_ult_used_times = current_used_times
            self.last_direnjie_ult_hit_hero_times = current_hit_hero_times
            self.last_main_hero_kill_cnt = current_kill_cnt
            self._clear_pending_direnjie_ult_reward_state()
            return

        used_increased = current_used_times > self.last_direnjie_ult_used_times
        hit_increased = current_hit_hero_times > self.last_direnjie_ult_hit_hero_times
        kill_increased = current_kill_cnt > self.last_main_hero_kill_cnt

        # 1. 检测我方狄仁杰释放大招。
        if used_increased:
            self.pending_direnjie_ult_release_frame_no = frame_no
            self.pending_direnjie_ult_hit_frame_no = None

        if hit_increased:
            self.pending_direnjie_ult_hit_frame_no = frame_no

            # 已经命中，就不再把这次释放判定为空大。
            self.pending_direnjie_ult_release_frame_no = None

            # 狄仁杰大招命中敌方英雄时，如果我方防御塔正在攻击敌方英雄，
            # 说明这是一次塔下反击场景，给正奖励。
            enemy_runtime_id = None
            if enemy_hero is not None:
                enemy_runtime_id = enemy_hero.get("runtime_id")

            tower_attack_target = None
            if main_tower is not None:
                tower_attack_target = main_tower.get("attack_target")

            if (
                enemy_runtime_id is not None
                and tower_attack_target == enemy_runtime_id
            ):
                self.direnjie_ult_hit_under_tower_value = 1.0

            # 狄仁杰站在敌方塔攻击范围外，大招命中敌方塔内英雄，
            # 鼓励塔外安全消耗，而不是越塔抗塔。
            if self._is_direnjie_enemy_tower_poke_scene(
                main_hero=main_hero,
                enemy_hero=enemy_hero,
                enemy_tower=enemy_tower,
            ):
                self.direnjie_ult_tower_poke_value = 1.0

        # 3. 如果已经命中，检查命中后 2 秒内我方 kill_cnt 是否增加。
        #    1v1 中 main_hero["kill_cnt"] 增加即可表示我方击杀了敌方英雄，
        #    不再依赖 frame_action.dead_action。
        if self.pending_direnjie_ult_hit_frame_no is not None:
            hit_elapsed_frames = frame_no - self.pending_direnjie_ult_hit_frame_no

            if hit_elapsed_frames <= self.direnjie_ult_kill_window_frames:
                if kill_increased:
                    self.direnjie_ult_kill_after_hit_value = 1.0

                    # 早期等级压制击杀奖励：
                    # 大招命中后 5 秒内完成击杀，并且击杀发生时我方 4 级、敌方 3 级。
                    # 这个奖励依赖同一次 kill_increased，不单独开启新的 pending 事件。
                    if (
                        main_hero.get("level", 0) == 4
                        and enemy_hero is not None
                        and enemy_hero.get("level", 0) <= 3
                    ):
                        self.direnjie_ult_level4_kill_value = 1.0

                    self._clear_pending_direnjie_ult_reward_state()
            else:
                # 命中后超过 2 秒没击杀，不给奖励，清空事件。
                self._clear_pending_direnjie_ult_reward_state()

        # 4. 释放后 2 秒内没有命中，给空大惩罚。
        if (
            self.pending_direnjie_ult_release_frame_no is not None
            and self.pending_direnjie_ult_hit_frame_no is None
        ):
            release_elapsed_frames = (
                frame_no - self.pending_direnjie_ult_release_frame_no
            )

            if release_elapsed_frames > self.direnjie_ult_hit_window_frames:
                self.direnjie_ult_miss_value = -1.0
                self._clear_pending_direnjie_ult_reward_state()

        # 5. 更新上一帧累计计数。
        self.last_direnjie_ult_used_times = current_used_times
        self.last_direnjie_ult_hit_hero_times = current_hit_hero_times
        self.last_main_hero_kill_cnt = current_kill_cnt


    def _clear_pending_rage_reward_state(self):
        """
        清空当前待判定的狂暴使用事件。
        不清空 last_rage_used_times / last_rage_main_hero_kill_cnt，
        因为它们是累计计数的上一帧基准。
        """
        self.pending_rage_use_frame_no = None

    def update_recover_reward_state(self, frame_data, main_hero):
        """
        更新恢复召唤师技能使用惩罚状态。

        奖励规则：
            1. 从 slot_type=5 获取恢复技能；
            2. 如果 usedTimes 增加，说明本帧使用了恢复；
            3. 使用恢复本身给一个小惩罚 recover_use_penalty = -1；
            4. 实际惩罚强度由 GameConfig.REWARD_WEIGHT_DICT["recover_use_penalty"] 控制。
        """
        frame_no = frame_data["frame_no"]

        # 同一帧只更新一次跨帧状态
        if self.recover_reward_last_update_frame_no == frame_no:
            return

        self.recover_reward_last_update_frame_no = frame_no
        self.recover_use_penalty_value = 0.0

        recover_skill = self._get_skill_by_slot_type(
            main_hero,
            self.recover_skill_slot_type,
        )

        # 没有恢复技能槽，直接清状态
        if recover_skill is None:
            self.last_recover_used_times = None
            return

        current_used_times = recover_skill.get("usedTimes", 0)

        # 第一次进入，只建立基准，不触发惩罚
        if self.last_recover_used_times is None:
            self.last_recover_used_times = current_used_times
            return

        # usedTimes 变小，通常说明新一局或状态重置
        if current_used_times < self.last_recover_used_times:
            self.last_recover_used_times = current_used_times
            return

        used_increased = current_used_times > self.last_recover_used_times

        # 检测使用恢复
        if used_increased:
            # 使用恢复本身给一个小惩罚，鼓励智能体只在更有价值的时候使用
            self.recover_use_penalty_value = -1.0

        # 更新上一帧累计使用次数
        self.last_recover_used_times = current_used_times

    def update_rage_reward_state(self, frame_data, main_hero, enemy_hero):
        """
        更新狂暴召唤师技能相关奖励状态。

        输出到：
            self.rage_kill_after_use_value
            self.rage_no_kill_after_use_value

        奖励规则：
            1. 我方没有携带狂暴：两个奖励都是 0；
            2. 召唤师技能槽 usedTimes 增加：认为使用了狂暴；
            3. 使用后 8 秒内 main_hero["kill_cnt"] 增加：rage_kill_after_use = +1；
            4. 使用后 8 秒内没有击杀：rage_no_kill_after_use = -1。
        """
        frame_no = frame_data["frame_no"]

        # 同一帧可能先后计算 rage_kill_after_use 和 rage_no_kill_after_use，
        # 只允许第一次进入时更新跨帧状态。
        if self.rage_reward_last_update_frame_no == frame_no:
            return

        self.rage_reward_last_update_frame_no = frame_no
        self.rage_kill_after_use_value = 0.0
        self.rage_no_kill_after_use_value = 0.0
        self.rage_use_penalty_value = 0.0

        summon_skill = self._get_skill_by_slot_type(
            main_hero,
            self.summon_skill_slot_type,
        )

        # 没有召唤师技能槽，直接清状态。
        if summon_skill is None:
            self.last_rage_used_times = None
            self.last_rage_main_hero_kill_cnt = None
            self._clear_pending_rage_reward_state()
            return

        # 只处理狂暴召唤师技能。
        if summon_skill.get("configId", 0) != self.rage_skill_config_id:
            self.last_rage_used_times = None
            self.last_rage_main_hero_kill_cnt = None
            self._clear_pending_rage_reward_state()
            return

        current_used_times = summon_skill.get("usedTimes", 0)
        current_kill_cnt = main_hero.get("kill_cnt", 0)

        # 第一次进入时只建立基准，不触发奖励/惩罚。
        if self.last_rage_used_times is None:
            self.last_rage_used_times = current_used_times
            self.last_rage_main_hero_kill_cnt = current_kill_cnt
            return

        # 如果累计计数变小，通常说明新一局或状态重置。
        if (
            current_used_times < self.last_rage_used_times
            or (
                self.last_rage_main_hero_kill_cnt is not None
                and current_kill_cnt < self.last_rage_main_hero_kill_cnt
            )
        ):
            self.last_rage_used_times = current_used_times
            self.last_rage_main_hero_kill_cnt = current_kill_cnt
            self._clear_pending_rage_reward_state()
            return

        used_increased = current_used_times > self.last_rage_used_times
        kill_increased = current_kill_cnt > self.last_rage_main_hero_kill_cnt

        # 1. 检测我方使用狂暴。
        # 1. 检测我方使用狂暴。
        if used_increased:
            self.pending_rage_use_frame_no = frame_no

            # 使用狂暴本身给一个小惩罚，鼓励模型只在有价值时使用。
            self.rage_use_penalty_value = -1.0

        # 2. 如果有待判定的狂暴使用事件，检查 8 秒内是否击杀。
        if self.pending_rage_use_frame_no is not None:
            rage_elapsed_frames = frame_no - self.pending_rage_use_frame_no

            if rage_elapsed_frames <= self.rage_kill_window_frames:
                if kill_increased:
                    self.rage_kill_after_use_value = 1.0
                    self._clear_pending_rage_reward_state()
            else:
                # 使用狂暴后超过 8 秒没有击杀，给惩罚。
                self.rage_no_kill_after_use_value = -1.0
                self._clear_pending_rage_reward_state()

        # 3. 更新上一帧累计计数。
        self.last_rage_used_times = current_used_times
        self.last_rage_main_hero_kill_cnt = current_kill_cnt

    def result(self, frame_data):
        # 获取当前帧号
        frame_no = frame_data["frame_no"]
        # 初始化英雄的各个等级经验
        self.init_max_exp_of_each_hero()
        # 对英雄的特征数据进行处理
        self.frame_data_process(frame_data)
        # 得到对应的奖励信息
        self.get_reward(frame_data, self.m_reward_value)
        # 如果设置了时间折扣因子，则进行开始奖励退火
        if self.time_scale_arg > 0:
            # 遍历所有的奖励，并且对对应的值进行衰减
            for key in self.m_reward_value:
                # 下面是实现退火逻辑，默认的逻辑函数是0.6^(frame_no/scale_arg)
                self.m_reward_value[key] *= math.pow(0.6, 1.0 * frame_no / self.time_scale_arg)

        return self.m_reward_value

    # Calculate the value of each reward item in each frame
    # 计算每帧的每个奖励子项的信息
    def set_cur_calc_frame_vec(self, cul_calc_frame_map, frame_data, camp):

        # Get both agents
        # 获取双方智能体

        # 主视角英雄和敌人英雄
        main_hero, enemy_hero = None, None
        # 英雄列表
        hero_list = frame_data["hero_states"]
        for hero in hero_list:
            hero_camp = hero["camp"]
            if hero_camp == camp:
                main_hero = hero
            else:
                enemy_hero = hero
        # 主英雄当前血量
        main_hero_hp = main_hero["hp"]
        # 主英雄最大血量
        main_hero_max_hp = main_hero["max_hp"]
        # 主英雄法力值
        main_hero_ep = main_hero["ep"]
        # 主英雄最大法力值
        main_hero_max_ep = main_hero["max_ep"]

        # Get both defense towers
        # 获取双方防御塔
        main_tower, main_spring, enemy_tower, enemy_spring = None, None, None, None
        npc_list = frame_data["npc_states"]
        for organ in npc_list:
            # 防御塔阵营
            organ_camp = organ["camp"]
            # 防御塔类型,有sprint-水晶， tower-防御塔
            organ_subtype = organ["sub_type"]
            # 下面的逻辑是确定此时的防御塔阵营
            if organ_camp == camp:
                if organ_subtype == 21:  # 21 is ACTOR_SUB_TOWER, normal tower
                    main_tower = organ
                elif organ_subtype == 24:  # 24 is ACTOR_SUB_CRYSTAL, base crystal
                    main_spring = organ
            else:
                if organ_subtype == 21:  # 21 is ACTOR_SUB_TOWER, normal tower
                    enemy_tower = organ
                elif organ_subtype == 24:  # 24 is ACTOR_SUB_CRYSTAL, base crystal
                    enemy_spring = organ

        for reward_name, reward_struct in cul_calc_frame_map.items():
            # 将前一个帧的信息给last_frame_value
            reward_struct.last_frame_value = reward_struct.cur_frame_value

            # Money
            # 金钱
            if reward_name == "money":
                # 计算当前的经济
                reward_struct.cur_frame_value = main_hero["money_cnt"]
            # Health points
            # 生命值
            elif reward_name == "hp_point":
                # 计算对应的血量
                reward_struct.cur_frame_value = math.sqrt(math.sqrt(1.0 * main_hero_hp / main_hero_max_hp))
            # Energy points
            # 法力值
            elif reward_name == "ep_rate":
                # 统计法力百分比
                if main_hero_max_ep == 0 or main_hero_hp <= 0:
                    reward_struct.cur_frame_value = 0
                else:
                    reward_struct.cur_frame_value = main_hero_ep / float(main_hero_max_ep)
            # Kills
            # 击杀
            elif reward_name == "kill":
                # 统计击杀次数
                reward_struct.cur_frame_value = main_hero["kill_cnt"]
            # Deaths
            # 死亡
            elif reward_name == "death":
                # 统计死亡次数
                reward_struct.cur_frame_value = main_hero["dead_cnt"]
            # Tower health points
            # 塔血量
            elif reward_name == "tower_hp_point":
                # 统计塔血量百分比
                reward_struct.cur_frame_value = 1.0 * main_tower["hp"] / main_tower["max_hp"]
            # Last hit
            # 补刀
            elif reward_name == "last_hit":
                # 先默认初始化当前帧信息为0
                reward_struct.cur_frame_value = 0.0
                # 取出当前的死亡事件信息
                frame_action = frame_data["frame_action"]
                # 如果发生了死亡事件
                if "dead_action" in frame_action:
                    # 取出当前的死亡事件信息
                    dead_actions = frame_action["dead_action"]

                    for dead_action in dead_actions:
                        # 这个部分鼓励我方英雄去杀小兵
                        if (
                            # 杀人者-我方英雄
                            dead_action["killer"]["runtime_id"] == main_hero["runtime_id"]
                            # 死亡者-敌方小兵
                            and dead_action["death"]["sub_type"] == 11 and dead_action['death']['actor_type'] == 1
                        ):
                            # 此时当前帧的value + 1
                            reward_struct.cur_frame_value += 1.0
                        # 这个部分阻碍敌方英雄去杀小兵
                        elif (
                            # 杀人者-敌方英雄
                            dead_action["killer"]["runtime_id"] == enemy_hero["runtime_id"]
                            # 死亡者-我方小兵
                            and dead_action["death"]["sub_type"] == 11 and dead_action['death']['actor_type'] == 1
                        ):
                            # 此时当前帧value-1
                            reward_struct.cur_frame_value -= 1.0

            # Monster kill
            # 击杀野怪 / 中立单位
            elif reward_name == "monster_kill":
                reward_struct.cur_frame_value = 0.0
                frame_action = frame_data.get("frame_action", {})

                if "dead_action" in frame_action:
                    dead_actions = frame_action["dead_action"]

                    for dead_action in dead_actions:
                        death = dead_action.get("death", {})
                        killer = dead_action.get("killer", {})

                        # 野怪 / 中立单位：
                        # actor_type=1, sub_type=0, camp=0
                        is_neutral_monster = (
                            death.get("actor_type") == 1
                            and death.get("sub_type") == 0
                            and death.get("camp") == 0
                        )

                        if not is_neutral_monster:
                            continue

                        # 我方英雄击杀野怪：正奖励
                        if (
                            killer.get("runtime_id") == main_hero["runtime_id"]
                        ):
                            reward_struct.cur_frame_value += 1.0

                        # 敌方英雄击杀野怪：负奖励
                        elif (
                            killer.get("runtime_id") == enemy_hero["runtime_id"]
                        ):
                            reward_struct.cur_frame_value -= 1.0

            # 第一次打到野怪
            elif reward_name == "monster_first_hit":
                reward_struct.cur_frame_value = 0.0

                # frame_data_process 会调用两次：
                # 先算主英雄视角，再算敌方视角。
                # 这个奖励不做零和，只在主视角更新状态，避免重复修改。
                if camp != self.main_hero_camp:
                    continue

                # 1. 找当前活着的中立野怪
                neutral_monster_runtime_id = None
                for npc in frame_data.get("npc_states", []):
                    if (
                        npc.get("actor_type") == 1
                        and npc.get("sub_type") == 0
                        and npc.get("camp") == 0
                        and npc.get("hp", 0) > 0
                    ):
                        neutral_monster_runtime_id = npc.get("runtime_id")
                        break

                # 2. 如果野怪当前活着，检查我方英雄是否命中它
                if neutral_monster_runtime_id is not None:
                    for hit_info in main_hero.get("hit_target_info", []):
                        if hit_info.get("hit_target") == neutral_monster_runtime_id:
                            if not self.monster_first_hit_rewarded:
                                reward_struct.cur_frame_value += 1.0
                                self.monster_first_hit_rewarded = True
                            break


                # 4. 最后用死亡事件刷新
                # 野怪死亡后，下一次复活可以重新获得 first-hit 奖励
                frame_action = frame_data.get("frame_action", {})
                for dead_action in frame_action.get("dead_action", []):
                    death = dead_action.get("death", {})

                    is_neutral_monster_dead = (
                        death.get("actor_type") == 1
                        and death.get("sub_type") == 0
                        and death.get("camp") == 0
                    )

                    if is_neutral_monster_dead:
                        self.monster_first_hit_rewarded = False
                        break
                        # Wave hit enemy hero

            # 每一轮小兵期间，第一次打到敌方英雄给一次奖励
            elif reward_name == "wave_hit_enemy_hero":
                reward_struct.cur_frame_value = 0.0

                # frame_data_process 会调用两次：
                # 一次主视角，一次敌方视角。
                # 这个奖励不做零和，只在主视角更新状态。
                if camp != self.main_hero_camp:
                    continue

                frame_no = frame_data["frame_no"]
                wave_idx = self.get_soldier_wave_idx(frame_no)

                # 第一波兵还没出生，不给这个奖励
                if wave_idx < 0:
                    continue

                # 进入新的一波兵线周期，刷新奖励状态
                if wave_idx != self.current_soldier_wave_idx:
                    self.current_soldier_wave_idx = wave_idx
                    self.wave_hit_enemy_hero_rewarded = False

                # 当前这一波已经给过奖励，就不再给
                if self.wave_hit_enemy_hero_rewarded:
                    continue

                enemy_runtime_id = enemy_hero["runtime_id"]

                # 检查我方英雄当前帧是否命中敌方英雄
                for hit_info in main_hero.get("hit_target_info", []):
                    if hit_info.get("hit_target") == enemy_runtime_id:
                        reward_struct.cur_frame_value += 1.0
                        self.wave_hit_enemy_hero_rewarded = True
                        break

            # 狄仁杰大招命中后 2 秒内击杀敌方英雄奖励
            elif reward_name == "direnjie_ult_kill_after_hit":
                reward_struct.cur_frame_value = 0.0

                # frame_data_process 会调用两次：
                # 一次主视角，一次敌方视角。
                # 这个奖励只在主视角更新，避免重复修改跨帧状态。
                if camp != self.main_hero_camp:
                    continue

                self.update_direnjie_ult_reward_state(
                    frame_data=frame_data,
                    main_hero=main_hero,
                    enemy_hero=enemy_hero,
                    main_tower=main_tower,
                    enemy_tower=enemy_tower,
                )

                reward_struct.cur_frame_value = self.direnjie_ult_kill_after_hit_value

            # 狄仁杰大招命中后 5 秒内击杀，且击杀时我方 4 级、敌方 3 级奖励
            elif reward_name == "direnjie_ult_level4_kill":
                reward_struct.cur_frame_value = 0.0

                # 只在主视角更新，避免重复修改跨帧状态
                if camp != self.main_hero_camp:
                    continue

                self.update_direnjie_ult_reward_state(
                    frame_data=frame_data,
                    main_hero=main_hero,
                    enemy_hero=enemy_hero,
                    main_tower=main_tower,
                    enemy_tower=enemy_tower,
                )

                reward_struct.cur_frame_value = self.direnjie_ult_level4_kill_value

            # 狄仁杰释放大招后 2 秒内未命中敌方英雄惩罚
            elif reward_name == "direnjie_ult_miss":
                reward_struct.cur_frame_value = 0.0

                # frame_data_process 会调用两次：
                # 一次主视角，一次敌方视角。
                # 这个奖励只在主视角更新，避免重复修改跨帧状态。
                if camp != self.main_hero_camp:
                    continue

                self.update_direnjie_ult_reward_state(
                    frame_data=frame_data,
                    main_hero=main_hero,
                    enemy_hero=enemy_hero,
                    main_tower=main_tower,
                    enemy_tower=enemy_tower,
                )

                reward_struct.cur_frame_value = self.direnjie_ult_miss_value


            # 狂暴使用后 8 秒内击杀敌方英雄奖励
            elif reward_name == "rage_kill_after_use":
                reward_struct.cur_frame_value = 0.0

                # frame_data_process 会调用两次：
                # 一次主视角，一次敌方视角。
                # 这个奖励只在主视角更新，避免重复修改跨帧状态。
                if camp != self.main_hero_camp:
                    continue

                self.update_rage_reward_state(
                    frame_data=frame_data,
                    main_hero=main_hero,
                    enemy_hero=enemy_hero,
                )

                reward_struct.cur_frame_value = self.rage_kill_after_use_value

            # 狂暴使用后 8 秒内没有击杀敌方英雄惩罚
            elif reward_name == "rage_no_kill_after_use":
                reward_struct.cur_frame_value = 0.0

                # frame_data_process 会调用两次：
                # 一次主视角，一次敌方视角。
                # 这个奖励只在主视角更新，避免重复修改跨帧状态。
                if camp != self.main_hero_camp:
                    continue

                self.update_rage_reward_state(
                    frame_data=frame_data,
                    main_hero=main_hero,
                    enemy_hero=enemy_hero,
                )

                reward_struct.cur_frame_value = self.rage_no_kill_after_use_value
            # 使用狂暴本身的小惩罚
            elif reward_name == "rage_use_penalty":
                reward_struct.cur_frame_value = 0.0

                # frame_data_process 会调用两次：
                # 一次主视角，一次敌方视角。
                # 这个奖励只在主视角更新，避免重复修改跨帧状态。
                if camp != self.main_hero_camp:
                    continue

                self.update_rage_reward_state(
                    frame_data=frame_data,
                    main_hero=main_hero,
                    enemy_hero=enemy_hero,
                )

                reward_struct.cur_frame_value = self.rage_use_penalty_value
            
            elif reward_name == "grass_ambush_kill":
                reward_struct.cur_frame_value = 0.0

                # 只在主视角更新，不做零和奖励
                if camp != self.main_hero_camp:
                    continue

                self.update_grass_ambush_reward_state(
                    frame_data=frame_data,
                    main_hero=main_hero,
                    enemy_hero=enemy_hero,
                )

                reward_struct.cur_frame_value = self.grass_ambush_kill_value

            # 使用恢复本身的小惩罚
            elif reward_name == "recover_use_penalty":
                reward_struct.cur_frame_value = 0.0

                # frame_data_process 会调用两次：
                # 一次主视角，一次敌方视角。
                # 这个奖励只在主视角更新，避免重复扣惩罚。
                if camp != self.main_hero_camp:
                    continue

                self.update_recover_reward_state(
                    frame_data=frame_data,
                    main_hero=main_hero,
                )

                reward_struct.cur_frame_value = self.recover_use_penalty_value

            # 狄仁杰大招命中，且我方防御塔正在攻击敌方英雄，鼓励塔下反击
            elif reward_name == "direnjie_ult_hit_under_tower":
                reward_struct.cur_frame_value = 0.0

                # 只在主视角更新，避免重复修改跨帧状态
                if camp != self.main_hero_camp:
                    continue

                self.update_direnjie_ult_reward_state(
                    frame_data=frame_data,
                    main_hero=main_hero,
                    enemy_hero=enemy_hero,
                    main_tower=main_tower,
                    enemy_tower=enemy_tower,
                )

                reward_struct.cur_frame_value = self.direnjie_ult_hit_under_tower_value

            # 狄仁杰站在敌方塔攻击范围外，大招命中敌方塔内英雄，鼓励塔外消耗
            elif reward_name == "direnjie_ult_tower_poke":
                reward_struct.cur_frame_value = 0.0

                # 只在主视角更新，避免重复修改跨帧状态
                if camp != self.main_hero_camp:
                    continue

                self.update_direnjie_ult_reward_state(
                    frame_data=frame_data,
                    main_hero=main_hero,
                    enemy_hero=enemy_hero,
                    main_tower=main_tower,
                    enemy_tower=enemy_tower,
                )

                reward_struct.cur_frame_value = self.direnjie_ult_tower_poke_value

            # Experience points
            # 经验值
            elif reward_name == "exp":
                # 根据self.calculate_exp_sum获取经验value
                reward_struct.cur_frame_value = self.calculate_exp_sum(main_hero)
            # Forward
            # 前进
            elif reward_name == "forward":
                # 根据self.calculate_forward获取forward的value
                reward_struct.cur_frame_value = self.calculate_forward(main_hero, main_tower, enemy_tower)

    # Calculate the total amount of experience gained using agent level and current experience value
    # 用智能体等级和当前经验值，计算获得经验值的总量
    def calculate_exp_sum(self, this_hero_info):
        exp_sum = 0.0
        # 通过我方当前的等级去遍历
        for i in range(1, this_hero_info["level"]):
            # 统计累计经验
            exp_sum += self.m_each_level_max_exp[i]
        # 因为前面sum是加到1-当前角色等级-1，所以此处要补充当前的角色等级的经验,这样做的目的是避免当前角色没有升级满，但是加了满经验
        exp_sum += this_hero_info["exp"]
        return exp_sum

    # Calculate the forward reward based on the distance between the agent and both defensive towers
    # 用智能体到双方防御塔的距离，计算前进奖励
    def calculate_forward(self, main_hero, main_tower, enemy_tower):
        # 获取主防御塔在地图里面的x和z
        main_tower_pos = (main_tower["location"]["x"], main_tower["location"]["z"])
        # 获取敌人防御塔在地图里面的x和z
        enemy_tower_pos = (enemy_tower["location"]["x"], enemy_tower["location"]["z"])
        # 获取英雄当前在地图里面的x和z
        hero_pos = (
            main_hero["location"]["x"],
            main_hero["location"]["z"],
        )
        # 初始化forward_value
        forward_value = 0
        # 用math里面的距离函数去计算当前主英雄和敌人防御塔的距离
        dist_hero2emy = math.dist(hero_pos, enemy_tower_pos)
        # 计算我方防御塔和敌方防御塔直接的距离
        dist_main2emy = math.dist(main_tower_pos, enemy_tower_pos)
        # and左边代表英雄血量健康， and右边是此时的主英雄在两个防御塔之间的战争之外
        if main_hero["hp"] / main_hero["max_hp"] > 0.99 and dist_hero2emy > dist_main2emy:
            # 计算逻辑为（两塔之间的距离 - 主英雄与敌方防御塔的距离）/ 主英雄与敌方防御塔的距离
            forward_value = (dist_main2emy - dist_hero2emy) / dist_main2emy

        return forward_value

    # Calculate the reward item information for both sides using frame data
    # 用帧数据来计算两边的奖励子项信息
    def frame_data_process(self, frame_data):
        main_camp, enemy_camp = -1, -1

        for hero in frame_data["hero_states"]:
            if hero["runtime_id"] == self.main_hero_player_id:
                main_camp = hero["camp"]
                self.main_hero_camp = main_camp
            else:
                enemy_camp = hero["camp"]
        # 获取主英雄的各项信息
        self.set_cur_calc_frame_vec(self.m_main_calc_frame_map, frame_data, main_camp)
        # 获取敌方英雄的各项信息
        self.set_cur_calc_frame_vec(self.m_enemy_calc_frame_map, frame_data, enemy_camp)

    # Use the values obtained in each frame to calculate the corresponding reward value
    # 用每一帧得到的奖励子项信息来计算对应的奖励值
    def get_reward(self, frame_data, reward_dict):
    # 清空当前的奖励字典
        reward_dict.clear()

        # 初始化总奖励和总权重
        reward_sum, weight_sum = 0.0, 0.0
        # 从权重map取出对应的权重名和权重值来着GameConfig里面
        for reward_name, reward_struct in self.m_cur_calc_frame_map.items():

            # 英雄血量
            if reward_name == "hp_point":
                if (
                    # 前一帧主英雄和敌方英雄血量都为空的时候
                    self.m_main_calc_frame_map[reward_name].last_frame_value == 0.0
                    and self.m_enemy_calc_frame_map[reward_name].last_frame_value == 0.0
                ):
                    # 给reward_struct的前一帧和当前帧初始化为0
                    reward_struct.cur_frame_value = 0
                    reward_struct.last_frame_value = 0
                # 前一帧我方主英雄血量为空
                elif self.m_main_calc_frame_map[reward_name].last_frame_value == 0.0:
                    # 给reward_struct的前一帧和当前帧初始化为（0 - 敌方血量）
                    reward_struct.cur_frame_value = 0 - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                    reward_struct.last_frame_value = 0 - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                # 前一帧敌方英雄血量为空
                elif self.m_enemy_calc_frame_map[reward_name].last_frame_value == 0.0:
                    # 给reward_struct的前一帧和当前帧初始化为（我方血量 - 0）
                    reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value - 0
                    reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value - 0
                # 都不为空
                else:
                    # 给reward_struct的前一帧初始化为（我方血量 - 敌方血量）
                    reward_struct.cur_frame_value = (
                        self.m_main_calc_frame_map[reward_name].cur_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                    )
                    # 给reward_struct的当前帧初始化为（我方血量 - 敌方血量）
                    reward_struct.last_frame_value = (
                        self.m_main_calc_frame_map[reward_name].last_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                    )
                # 最后给reward_struct的value赋值为（当前帧的value - 前一帧的value）
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value

            # 法力值
            elif reward_name == "ep_rate":
                # 前一帧和当前帧初始化为主英雄对应的数值
                reward_struct.cur_frame_value = self.m_main_calc_frame_map[reward_name].cur_frame_value
                reward_struct.last_frame_value = self.m_main_calc_frame_map[reward_name].last_frame_value
                # 如果前一帧对应的数值是正数即为大于0
                if reward_struct.last_frame_value > 0:
                    # 给reward_struct的value赋值为（当前帧的value - 前一帧的value）
                    reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
                # 如果没了则为0
                else:
                    reward_struct.value = 0

            # 经验值
            elif reward_name == "exp":
                # 默认初始化主英雄为空
                main_hero = None
                # 遍历所有的英雄
                for hero in frame_data["hero_states"]:
                    # 如果此时满足为主英雄则将此时的主英雄指定
                    if hero["player_id"] == self.main_hero_player_id:
                        main_hero = hero
                # 主英雄已经制定了，而且主英雄的等级满了，将reward_struct的value赋值为0
                if main_hero and main_hero["level"] >= 15:
                    reward_struct.value = 0
                # 如果没有满
                else:
                    # 对应的reward_struct的当前帧为（我方经验 - 敌方经验）
                    reward_struct.cur_frame_value = (
                        self.m_main_calc_frame_map[reward_name].cur_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                    )
                    # 对应的reward_struct的前一帧为（我方经验 - 敌方经验）
                    reward_struct.last_frame_value = (
                        self.m_main_calc_frame_map[reward_name].last_frame_value
                        - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                    )
                    # 然后将reward_struct的value赋值为（当前帧的value - 前一帧的value）
                    reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value

            # 英雄与塔之间的距离，判断英雄是否进入战场，此为惩罚
            elif reward_name == "forward":
                # 具体参考奖励函数
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            # 最后一次击杀（对击杀事件的分析）
            elif reward_name == "last_hit":
                # 直接将reward_struct的value赋值给主英雄当前帧对应的值，对应的实现在初始化里面，详细参考set_cur_calc_frame_vec函数，对应179行之后
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            
            elif reward_name == "monster_kill":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value

            elif reward_name == "monster_first_hit":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value

            elif reward_name == "wave_hit_enemy_hero":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value

            elif reward_name == "direnjie_ult_kill_after_hit":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value

            elif reward_name == "direnjie_ult_level4_kill":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value

            elif reward_name == "direnjie_ult_miss":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value

            elif reward_name == "rage_kill_after_use":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value

            elif reward_name == "rage_no_kill_after_use":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            
            elif reward_name == "rage_use_penalty":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            
            elif reward_name == "grass_ambush_kill":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value
            
            elif reward_name == "recover_use_penalty":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value

            elif reward_name == "direnjie_ult_hit_under_tower":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value

            elif reward_name == "direnjie_ult_tower_poke":
                reward_struct.value = self.m_main_calc_frame_map[reward_name].cur_frame_value

            # 其他的奖励包括 money、death、kill、tower_hp_point
            else:
                # 对应的reward_struct的当前帧为（我方 - 敌方）
                reward_struct.cur_frame_value = (
                    self.m_main_calc_frame_map[reward_name].cur_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].cur_frame_value
                )
                # 对应的reward_struct的前一帧为（我方 - 敌方）
                reward_struct.last_frame_value = (
                    self.m_main_calc_frame_map[reward_name].last_frame_value
                    - self.m_enemy_calc_frame_map[reward_name].last_frame_value
                )
                # 然后将reward_struct的value赋值为（当前帧的value - 前一帧的value）
                reward_struct.value = reward_struct.cur_frame_value - reward_struct.last_frame_value
            
            # 权重相加
            weight_sum += reward_struct.weight
            # 奖励*权重 = 总奖励 ， 总奖励累加
            reward_sum += reward_struct.value * reward_struct.weight
            # 将对应的奖励名称的奖励值赋值给奖励字典
            reward_dict[reward_name] = reward_struct.value
        # 总奖励的值赋值给reward_dict
        reward_dict["reward_sum"] = reward_sum
