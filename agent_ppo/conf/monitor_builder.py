#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    """
    # This function is used to create monitoring panel configurations for custom indicators.
    # 该函数用于创建自定义指标的监控面板配置。
    """
    monitor = MonitorConfigBuilder()

    config_dict = (
        monitor.title("智能决策1v1")
        .add_group(
            group_name="算法指标",
            group_name_en="algorithm",
        )
        .add_panel(
            name="累积回报",
            name_en="reward",
            type="line",
        )
        .add_metric(
            metrics_name="reward",
            expr="round(avg(reward{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="总损失",
            name_en="total_loss",
            type="line",
        )
        .add_metric(
            metrics_name="total_loss",
            expr="round(avg(total_loss{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="价值损失",
            name_en="value_loss",
            type="line",
        )
        .add_metric(
            metrics_name="value_loss",
            expr="round(avg(value_loss{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="策略损失",
            name_en="policy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="policy_loss",
            expr="round(avg(policy_loss{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="熵损失",
            name_en="entropy_loss",
            type="line",
        )
        .add_metric(
            metrics_name="entropy_loss",
            expr="round(avg(entropy_loss{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="梯度范数",
            name_en="grad_norm",
            type="line",
        )
        .add_metric(
            metrics_name="grad_norm",
            expr="round(avg(grad_norm{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="梯度裁剪比例",
            name_en="gradclip_frac",
            type="line",
        )
        .add_metric(
            metrics_name="gradclip_frac",
            expr="round(avg(gradclip_frac{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="解释方差",
            name_en="explained_variance",
            type="line",
        )
        .add_metric(
            metrics_name="explained_variance",
            expr="round(avg(explained_variance{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="裁剪比例",
            name_en="clipfrac",
            type="line",
        )
        .add_metric(
            metrics_name="clipfrac",
            expr="round(avg(clipfrac{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="近似KL散度",
            name_en="approx_kl",
            type="line",
        )
        .add_metric(
            metrics_name="approx_kl",
            expr="round(avg(approx_kl{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="优势均值",
            name_en="adv_mean",
            type="line",
        )
        .add_metric(
            metrics_name="adv_mean",
            expr="round(avg(adv_mean{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="优势标准差",
            name_en="adv_std",
            type="line",
        )
        .add_metric(
            metrics_name="adv_std",
            expr="round(avg(adv_std{}), 0.01)",
        )
        .end_panel()
        .add_panel(
            name="局数",
            name_en="episode_cnt",
            type="line",
        )
        .add_metric(
            metrics_name="episode_cnt",
            expr="round(avg(episode_cnt{}), 0.01)",
        )
        .end_panel()

        .add_panel(
            name="前进权重",
            name_en="rw_forward",
            type="line",
        )
        .add_metric(
            metrics_name="reward_weight_forward",
            expr="round(avg(reward_weight_forward{}), 0.0001)",
        )
        .end_panel()

        .add_panel(
            name="经济权重",
            name_en="rw_money",
            type="line",
        )
        .add_metric(
            metrics_name="reward_weight_money",
            expr="round(avg(reward_weight_money{}), 0.0001)",
        )
        .end_panel()

        .add_panel(
            name="经验权重",
            name_en="rw_exp",
            type="line",
        )
        .add_metric(
            metrics_name="reward_weight_exp",
            expr="round(avg(reward_weight_exp{}), 0.0001)",
        )
        .end_panel()

        .add_panel(
            name="血量权重",
            name_en="rw_hp_point",
            type="line",
        )
        .add_metric(
            metrics_name="reward_weight_hp_point",
            expr="round(avg(reward_weight_hp_point{}), 0.0001)",
        )
        .end_panel()

        .add_panel(
            name="蓝量权重",
            name_en="rw_ep_rate",
            type="line",
        )
        .add_metric(
            metrics_name="reward_weight_ep_rate",
            expr="round(avg(reward_weight_ep_rate{}), 0.0001)",
        )
        .end_panel()

        .add_panel(
            name="击杀权重",
            name_en="rw_kill",
            type="line",
        )
        .add_metric(
            metrics_name="reward_weight_kill",
            expr="round(avg(reward_weight_kill{}), 0.0001)",
        )
        .end_panel()

        .add_panel(
            name="死亡权重",
            name_en="rw_death",
            type="line",
        )
        .add_metric(
            metrics_name="reward_weight_death",
            expr="round(avg(reward_weight_death{}), 0.0001)",
        )
        .end_panel()

        .add_panel(
            name="塔血权重",
            name_en="rw_tower_hp",
            type="line",
        )
        .add_metric(
            metrics_name="reward_weight_tower_hp_point",
            expr="round(avg(reward_weight_tower_hp_point{}), 0.0001)",
        )
        .end_panel()

        .add_panel(
            name="补刀权重",
            name_en="rw_last_hit",
            type="line",
        )
        .add_metric(
            metrics_name="reward_weight_last_hit",
            expr="round(avg(reward_weight_last_hit{}), 0.0001)",
        )
        .end_panel()

        .add_panel(
            name="野怪击杀权重",
            name_en="rw_monster_kill",
            type="line",
        )
        .add_metric(
            metrics_name="reward_weight_monster_kill",
            expr="round(avg(reward_weight_monster_kill{}), 0.0001)",
        )
        .end_panel()

        .add_panel(
            name="野怪首击权重",
            name_en="rw_monster_first_hit",
            type="line",
        )
        .add_metric(
            metrics_name="reward_weight_monster_first_hit",
            expr="round(avg(reward_weight_monster_first_hit{}), 0.0001)",
        )
        .end_panel()

        .add_panel(
            name="兵线命中权重",
            name_en="rw_wave_hit_enemy",
            type="line",
        )
        .add_metric(
            metrics_name="reward_weight_wave_hit_enemy_hero",
            expr="round(avg(reward_weight_wave_hit_enemy_hero{}), 0.0001)",
        )
        .end_panel()

        .add_panel(
            name="狄仁杰大招命中后击杀权重",
            name_en="rw_dj_ult_kill_hit",
            type="line",
        )
        .add_metric(
            metrics_name="rw_dj_ult_kill_hit",
            expr="round(avg(rw_dj_ult_kill_hit{}), 0.0001)",
        )
        .end_panel()

        .add_panel(
            name="狄仁杰大招空放权重",
            name_en="rw_dj_ult_miss",
            type="line",
        )
        .add_metric(
            metrics_name="rw_dj_ult_miss",
            expr="round(avg(rw_dj_ult_miss{}), 0.0001)",
        )
        .end_panel()

        .add_panel(
            name="狂暴后击杀权重",
            name_en="rw_rage_kill",
            type="line",
        )
        .add_metric(
            metrics_name="rw_rage_kill",
            expr="round(avg(rw_rage_kill{}), 0.0001)",
        )
        .end_panel()

        .add_panel(
            name="狂暴后未击杀权重",
            name_en="rw_rage_no_kill",
            type="line",
        )
        .add_metric(
            metrics_name="rw_rage_no_kill",
            expr="round(avg(rw_rage_no_kill{}), 0.0001)",
        )
        .end_panel()

        .add_panel(
            name="狂暴使用惩罚权重",
            name_en="rw_rage_use_pen",
            type="line",
        )
        .add_metric(
            metrics_name="rw_rage_use_pen",
            expr="round(avg(rw_rage_use_pen{}), 0.0001)",
        )
        .end_panel()

        .add_panel(
            name="草丛伏击击杀权重",
            name_en="rw_grass_ambush",
            type="line",
        )
        .add_metric(
            metrics_name="rw_grass_ambush",
            expr="round(avg(rw_grass_ambush{}), 0.0001)",
        )
        .end_panel()

        .add_panel(
            name="时间尺度",
            name_en="time_scale_arg",
            type="line",
        )
        .add_metric(
            metrics_name="time_scale_arg",
            expr="round(avg(time_scale_arg{}), 0.01)",
        )
        .end_panel()

        .add_panel(
            name="学习率",
            name_en="lr",
            type="line",
        )
        .add_metric(
            metrics_name="lr",
            expr="round(avg(lr{}), 0.01)",
        )
        .end_panel()

        .add_panel(
            name="训练步数",
            name_en="train_steps",
            type="line",
        )
        .add_metric(
            metrics_name="train_steps",
            expr="round(avg(train_steps{}), 0.01)",
        )
        .end_panel()

        .add_panel(
            name="狄仁杰塔下反击权重",
            name_en="rw_dj_ult_tower_hit",
            type="line",
        )
        .add_metric(
            metrics_name="rw_dj_ult_tower_hit",
            expr="round(avg(rw_dj_ult_tower_hit{}), 0.0001)",
        )
        .end_panel()

        .add_panel(
            name="恢复使用惩罚权重",
            name_en="rw_recover_use_pen",
            type="line",
        )
        .add_metric(
            metrics_name="rw_recover_use_pen",
            expr="round(avg(rw_recover_use_pen{}), 0.0001)",
        )
        .end_panel()

        .add_panel(
            name="狄仁杰塔外消耗权重",
            name_en="rw_dj_tower_poke",
            type="line",
        )
        .add_metric(
            metrics_name="rw_dj_tower_poke",
            expr="round(avg(rw_dj_tower_poke{}), 0.0001)",
        )
        .end_panel()

        .add_panel(
            name="狄仁杰4级击杀权重",
            name_en="rw_dj_lv4_kill",
            type="line",
        )
        .add_metric(
            metrics_name="rw_dj_lv4_kill",
            expr="round(avg(rw_dj_lv4_kill{}), 0.0001)",
        )
        .end_panel()


        .add_panel(
            name="熵系数",
            name_en="entropy_beta",
            type="line",
        )
        .add_metric(
            metrics_name="entropy_beta",
            expr="round(avg(entropy_beta{}), 0.0001)",
        )
        .end_panel()
        

        .end_group()
        .build()
    )
    return config_dict
