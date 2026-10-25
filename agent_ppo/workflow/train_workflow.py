#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import os
import time
import random
from agent_ppo.feature.definition import (
    sample_process,
    build_frame,
    FrameCollector,
    NONE_ACTION,
    lineup_iterator_roundrobin_camp_heroes,
)
from agent_ppo.conf.conf import GameConfig
from tools.env_conf_manager import EnvConfManager
from tools.model_pool_utils import get_valid_model_pool
from tools.metrics_utils import get_training_metrics
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    # Whether the agent is training, corresponding to do_predicts
    # 智能体是否进行训练
    do_learns = [True, True]
    last_save_model_time = time.time()

    # Create environment configuration manager instance
    # 创建对局配置管理器实例
    env_conf_manager = EnvConfManager(
        config_path="agent_ppo/conf/train_env_conf.toml",
        logger=logger,
    )

    # Lineup iterator (112:Luban, 133:DiRenjie)
    # 阵容迭代器 (112:鲁班， 133:狄仁杰)
    lineup_iterator = lineup_iterator_roundrobin_camp_heroes([112, 133])

    # Create EpisodeRunner instance
    # 创建 EpisodeRunner 实例
    episode_runner = EpisodeRunner(
        env=envs[0],
        agents=agents,
        logger=logger,
        monitor=monitor,
        env_conf_manager=env_conf_manager,
        lineup_iterator=lineup_iterator,
    )

    while True:
        # Run episodes and collect data
        # 运行对局并收集数据
        for g_data in episode_runner.run_episodes():
            for index, (d_learn, agent) in enumerate(zip(do_learns, agents)):
                if d_learn and len(g_data[index]) > 0:
                    # The learner trains in a while true loop, here learn actually sends samples
                    # learner 采用 while true 训练，此处 learn 实际为发送样本
                    agent.send_sample_data(g_data[index])
            g_data.clear()

            now = time.time()
            if now - last_save_model_time > GameConfig.MODEL_SAVE_INTERVAL:
                agents[0].save_model()
                last_save_model_time = now


class EpisodeRunner:
    def __init__(self, env, agents, logger, monitor, env_conf_manager, lineup_iterator):
        self.env = env
        self.agents = agents
        self.logger = logger
        self.monitor = monitor
        self.env_conf_manager = env_conf_manager
        self.lineup_iterator = lineup_iterator
        self.agent_num = len(agents)
        self.episode_cnt = 0
        self.last_report_monitor_time = 0
        # 上一次被接受训练的鲁班 vs 狄仁杰混合局采样类别。
        # None 表示还没有接受过混合局。
        # "good" 表示狄仁杰赢，或狄仁杰长局输但按目标抗压样本采样。
        # "mid_loss" 表示狄仁杰中局输，用作纠错样本。
        self.last_mixed_sample_bucket = None

        # 当前局是否使用模型池对手。
        # 模型池局只用于“狄仁杰样本 vs 模型池鲁班对手”。
        self.current_pool_opponent_agent = None
        self.current_use_pool_opponent = False
        self.current_pool_direnjie_side = None
        self.current_pool_luban_side = None

        # 10% 概率触发模型池鲁班；弱鲁班局的轨迹级过滤阈值。
        self.pool_opponent_prob = 0.04
        self.pool_win_min_frame = 10000
        self.pool_short_loss_drop_frame = 10000
        self.pool_short_loss_drop_prob = 1.0

        # 统计真正参与训练的“狄仁杰 vs 鲁班”对局来源。
        # 只统计已经通过 keep_train_samples，并且即将进入 sample_process 的对局。
        self.train_direnjie_luban_pool_cnt = 0
        self.train_direnjie_luban_selfplay_cnt = 0

    def _get_tower_hp_by_camp(self, frame_state):
        """
        获取双方防御塔血量。
        sub_type == 21 表示防御塔。
        """
        tower_hp_by_camp = {}

        for npc in frame_state.get("npc_states", []):
            if npc.get("sub_type") == 21:
                camp = npc.get("camp")
                hp = npc.get("hp", 0)
                tower_hp_by_camp[camp] = hp

        return tower_hp_by_camp


    def _calc_terminal_win_loss_rewards(self, observation, terminated, truncated):
        """
        根据终局时双方防御塔剩余血量判断胜负。
        塔血量高的一方胜。
        """
        terminal_rewards = [0.0] * self.agent_num

        if not terminated or truncated:
            return terminal_rewards

        # 任意一个 observation 的 frame_state 都包含双方建筑信息
        frame_state = observation["0"]["frame_state"]
        tower_hp_by_camp = self._get_tower_hp_by_camp(frame_state)

        for i, agent in enumerate(self.agents):
            my_camp = agent.hero_camp

            enemy_camp = None
            for j, other_agent in enumerate(self.agents):
                if j != i:
                    enemy_camp = other_agent.hero_camp
                    break

            if enemy_camp is None:
                continue

            # 如果某方防御塔在终局帧消失，按 0 处理
            my_tower_hp = tower_hp_by_camp.get(my_camp, 0)
            enemy_tower_hp = tower_hp_by_camp.get(enemy_camp, 0)

            if my_tower_hp > enemy_tower_hp:
                terminal_rewards[i] = GameConfig.WIN_REWARD
            elif my_tower_hp < enemy_tower_hp:
                terminal_rewards[i] = GameConfig.LOSE_REWARD
            else:
                terminal_rewards[i] = 0.0

        return terminal_rewards

    def _get_winner_hero_id_by_tower_hp(self, observation):
        """
        根据终局双方防御塔血量判断赢家英雄 config_id。

        返回:
            winner_hero_id: 112 / 133 / None
        """
        frame_state = observation["0"]["frame_state"]

        # 1. 先从 npc_states 里找双方防御塔血量
        tower_hp_by_camp = self._get_tower_hp_by_camp(frame_state)

        hp_camp1 = tower_hp_by_camp.get(1, 0)
        hp_camp2 = tower_hp_by_camp.get(2, 0)

        if hp_camp1 > hp_camp2:
            winner_camp = 1
        elif hp_camp2 > hp_camp1:
            winner_camp = 2
        else:
            return None

        # 2. 再从 hero_states 里找 winner_camp 对应的英雄 config_id
        for hero in frame_state.get("hero_states", []):
            if hero.get("camp") == winner_camp:
                return hero.get("config_id")

        return None

    @staticmethod
    def _get_episode_length_bucket(frame_no):
        """
        按终局帧数划分对局长度。

        long: frame_no > 10000
        mid : 8000 < frame_no <= 10000
        short: frame_no <= 8000
        """
        if frame_no > 8000:
            return "long"
        if frame_no > 7000:
            return "mid"
        return "short"

    def _get_mixed_sample_bucket(self, winner_hero_id, frame_no):
        """
        鲁班(112) vs 狄仁杰(133) 混合局采样分类。

        采样逻辑：
        1. 狄仁杰赢：good。
        2. 狄仁杰输，但终局帧数 > 10000：long loss，按 good 采样。
        3. 狄仁杰输，且 8000 < 终局帧数 <= 10000：mid_loss。
        4. 狄仁杰输，且终局帧数 <= 8000：short loss，丢弃。
        5. 平局 / 无法判断赢家：丢弃。

        返回：
            sample_bucket: "good" / "mid_loss" / None
            length_bucket: "long" / "mid" / "short"
            direnjie_win: True / False / None
            sample_reason: 日志说明
        """
        length_bucket = self._get_episode_length_bucket(frame_no)

        if winner_hero_id is None:
            return None, length_bucket, None, "draw_or_unknown_winner"

        direnjie_win = winner_hero_id == 133
        if direnjie_win:
            return "good", length_bucket, True, "direnjie_win"

        # winner_hero_id != 133 表示狄仁杰输。
        if length_bucket == "long":
            return "good", length_bucket, False, "direnjie_long_loss_as_good"
        if length_bucket == "mid":
            return "mid_loss", length_bucket, False, "direnjie_mid_loss"
        return None, length_bucket, False, "direnjie_short_loss_drop"

    def _call_init_config(self, usr_conf):
        """Call init_config on both agents to get summoner skill selections,
        then inject the results into usr_conf.
        调用双方 agent 的 init_config 获取召唤师技能选择，并注入 usr_conf。
        """
        blue_hero_ids, red_hero_ids = EnvConfManager.extract_hero_ids_from_usr_conf(usr_conf)

        # camp_keys[i] is the camp key for agents[i] based on monitor_side
        # monitor_side 的 agent 对应 blue/red 取决于 monitor_side 配置
        monitor_side = self.env_conf_manager.get_monitor_side()
        camp_keys = ["blue_camp", "red_camp"]

        for agent_idx, agent in enumerate(self.agents):
            # Determine which camp this agent controls
            # 确定该 agent 控制哪个阵营
            if agent_idx == 0:
                my_hero_ids = blue_hero_ids
                opponent_hero_ids = red_hero_ids
                camp_key = camp_keys[0]
            else:
                my_hero_ids = red_hero_ids
                opponent_hero_ids = blue_hero_ids
                camp_key = camp_keys[1]

            config_data = {
                "my_camp": camp_key,
                "my_heroes": my_hero_ids,
                "opponent_heroes": opponent_hero_ids,
            }

            select_skills = agent.init_config(config_data)
            EnvConfManager.inject_select_skills(usr_conf, camp_key, select_skills)
            self.logger.info(
                f"Agent[{agent_idx}] init_config: camp={camp_key}, select_skills={select_skills}"
            )

    def run_episodes(self):
        # Single environment process
        # 单局流程
        while True:
            # Retrieving training metrics
            # 获取训练中的指标
            training_metrics = get_training_metrics()
            if training_metrics:
                for key, value in training_metrics.items():
                    if key == "env":
                        for env_key, env_value in value.items():
                            self.logger.info(f"training_metrics {key} {env_key} is {env_value}")
                    else:
                        self.logger.info(f"training_metrics {key} is {value}")

            # Update environment configuration
            # Can use a list of length 2 to pass in the lineup id of the current game
            # 更新对局配置, 可以用长度为2的列表传入当前对局的阵容id
            # Update environment configuration
            # 更新对局配置, 可以用长度为2的列表传入当前对局的阵容id
            """
            allowed_lineups = {
                (133, 133),  # 狄仁杰 vs 狄仁杰
                (112, 112),  # 鲁班 vs 鲁班
            }

            while True:
                lineup = next(self.lineup_iterator)
                if tuple(lineup) in allowed_lineups:
                    break

            self.current_lineup = lineup
            """
            lineup = next(self.lineup_iterator)
            self.current_lineup = lineup
       
            #self.logger.info(f"episode: {self.episode_cnt} ,lineup: {self.current_lineup}")
            

            usr_conf, is_eval, monitor_side = self.env_conf_manager.update_config(lineup)

            self.current_pool_opponent_agent = None
            self.current_use_pool_opponent = False
            self.current_pool_direnjie_side = None
            self.current_pool_luban_side = None

            lineup_tuple = tuple(self.current_lineup)

            # 只有鲁班(112) vs 狄仁杰(133) 混合局，才有 10% 概率把鲁班侧换成模型池对手。
            # 模型池局里，狄仁杰侧参与采样训练；鲁班侧只做对手，不采样。
            if (
                not is_eval
                and lineup_tuple in ((112, 133), (133, 112))
                and random.random() < self.pool_opponent_prob
            ):
                candidate_model_ids = get_valid_model_pool(self.logger)

                if candidate_model_ids:
                    self.current_pool_opponent_agent = random.choice(list(candidate_model_ids))
                    self.current_use_pool_opponent = True
                    self.current_pool_direnjie_side = lineup_tuple.index(133)
                    self.current_pool_luban_side = lineup_tuple.index(112)

                    self.logger.info(
                        f"[POOL_OPPONENT_SELECT] enabled=True, "
                        f"lineup={lineup_tuple}, "
                        f"direnjie_side={self.current_pool_direnjie_side}, "
                        f"luban_side={self.current_pool_luban_side}, "
                        f"selected_opponent_agent={self.current_pool_opponent_agent}"
                    )
                else:
                    self.logger.info(
                        f"[POOL_OPPONENT_SELECT] enabled=False, "
                        f"lineup={lineup_tuple}, reason=empty_model_pool"
                    )
            else:
                self.logger.info(
                    f"[POOL_OPPONENT_SELECT] enabled=False, "
                    f"lineup={lineup_tuple}, reason=not_selected_or_not_mixed"
                )


            # Call init_config on agents to get summoner skill selections
            # 调用 agent 的 init_config 获取召唤师技能选择，注入 usr_conf
            self._call_init_config(usr_conf)
            
            # Start a new environment
            # 启动新对局，返回初始环境状态

            env_obs = self.env.reset(usr_conf=usr_conf)
            # Disaster recovery
            # 容灾
            if handle_disaster_recovery(env_obs, self.logger):
                break

            observation = env_obs["observation"]
            extra_info = env_obs["extra_info"]

            # Reset agents
            # 重置智能体
            self.reset_agents(observation, self.episode_cnt)

            # Reset environment frame collector
            # 重置环境帧收集器
            frame_collector = FrameCollector(
                self.agent_num,
                episode_cnt=self.episode_cnt,
            )

            self.logger.info(
                f"[GAMMA_SCHEDULE] episode_cnt={self.episode_cnt}, gamma={frame_collector.gamma:.6f}"
            )

            self.episode_cnt += 1

            # Game variables
            # 对局变量
            self.episode_cnt += 1
            frame_no = 0
            reward_sum_list = [0] * self.agent_num

            # Reward segment statistics for time-decay fitting
            # 用于拟合时间衰减的分段 reward 统计
            reward_log_interval = 500
            next_reward_log_frame = reward_log_interval

            # 只统计训练侧 / monitor_side 智能体
            train_side = monitor_side

            reward_interval_sum = 0.0
            reward_interval_count = 0
            reward_interval_start_frame = 0

            # 一局结束后统一打印
            reward_interval_records = []


            is_train_test = os.environ.get("is_train_test", "False").lower() == "true"
            self.logger.info(f"Episode {self.episode_cnt} start, usr_conf is {usr_conf}")

            # Reward initialization
            # 回报初始化
            for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                if do_sample:
                    reward = agent.reward_manager.result(observation[str(i)]["frame_state"])
                    observation[str(i)]["reward"] = reward
                    reward_sum_list[i] += reward["reward_sum"]

            while True:
                # Initialize the default actions. If the agent does not make a decision, env.step uses the default action.
                # 初始化默认的actions，如果智能体不进行决策，则env.step使用默认action
                actions = [NONE_ACTION] * self.agent_num

                for index, (do_predict, do_sample, agent) in enumerate(
                    zip(self.do_predicts, self.do_samples, self.agents)
                ):
                    if do_predict:
                        if not is_eval:
                            actions[index] = agent.predict(observation[str(index)])
                        else:
                            actions[index] = agent.exploit(observation[str(index)])

                        # Only sample when do_sample=True and is_eval=False
                        # 评估对局数据不采样，不是训练中最新模型产生的数据不采样
                        if not is_eval and do_sample:
                            frame = build_frame(agent, observation[str(index)])
                            frame_collector.save_frame(frame, agent_id=index)

                # Step forward
                # 推进环境到下一帧，得到新的状态
                env_reward, env_obs = self.env.step(actions)
                # Disaster recovery
                # 容灾
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                frame_no = env_obs["frame_no"]
                observation = env_obs["observation"]
                extra_info = env_obs["extra_info"]
                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]

                

                # Reward generation
                # 计算回报，作为当前环境状态observation的一部分
                for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                    if do_sample:
                        reward = agent.reward_manager.result(observation[str(i)]["frame_state"])
                        observation[str(i)]["reward"] = reward
                        reward_sum_list[i] += reward["reward_sum"]
                        # 只统计训练侧智能体，不统计两个智能体之和
                        if i == train_side:
                            reward_interval_sum += reward['reward_sum']
                            reward_interval_count += 1
                if frame_no >= next_reward_log_frame:
                    reward_interval_records.append(
                        (
                            reward_interval_start_frame,
                            frame_no,
                            reward_interval_sum,
                        )
                    )

                    reward_interval_sum = 0.0
                    reward_interval_count = 0
                    reward_interval_start_frame = frame_no

                    while next_reward_log_frame <= frame_no:
                        next_reward_log_frame += reward_log_interval

                # Normal end or timeout exit, run train_test will exit early
                # 正常结束或超时退出，运行train_test时会提前退出
                is_gameover = terminated or truncated or (is_train_test and frame_no >= 1000)


                
                
                if is_gameover:

                    if reward_interval_records:
                        reward_interval_text = ", ".join(
                            [
                                f"[{start},{end}]:{reward_sum:.6f}"
                                for start, end, reward_sum in reward_interval_records
                            ]
                        )
                    else:
                        reward_interval_text = ""

                    self.logger.info(
                        f"[REWARD_INTERVAL_EPISODE] "
                        f"episode={self.episode_cnt}, "
                        f"train_side={train_side}, "
                        f"camp={self.agents[train_side].hero_camp}, "
                        f"is_eval={is_eval}, "
                        f"frame_no={frame_no}, "
                        f"segments={reward_interval_text}"
                    )


                    terminal_rewards = self._calc_terminal_win_loss_rewards(
                        observation=observation,
                        terminated=terminated,
                        truncated=truncated,
                    )


                    keep_train_samples = True
                    lineup_tuple = tuple(self.current_lineup)

                    if lineup_tuple in ((112, 133), (133, 112)):
                        winner_hero_id = self._get_winner_hero_id_by_tower_hp(observation)
                        length_bucket = self._get_episode_length_bucket(frame_no)

                        if self.current_use_pool_opponent:
                            # 模型池弱鲁班局：
                            # 1. 狄仁杰赢但帧数 < 10000：丢弃，避免学习打弱鲁班的短局莽赢。
                            # 2. 狄仁杰输且帧数 < 8000：50% 概率丢弃，保留少量明显错误负样本。
                            # 3. 其他情况保留。
                            random_drop_roll = None

                            if winner_hero_id is None:
                                direnjie_win = None
                                keep_train_samples = False
                                sample_reason = "pool_unknown_winner_drop"
                            else:
                                direnjie_win = winner_hero_id == 133

                                if direnjie_win and frame_no < self.pool_win_min_frame:
                                    keep_train_samples = False
                                    sample_reason = "pool_direnjie_win_under_10000_drop"
                                elif (not direnjie_win) and frame_no < self.pool_short_loss_drop_frame:
                                    random_drop_roll = random.random()
                                    keep_train_samples = random_drop_roll >= self.pool_short_loss_drop_prob
                                    sample_reason = (
                                        "pool_direnjie_loss_under_8000_keep"
                                        if keep_train_samples
                                        else "pool_direnjie_loss_under_8000_drop"
                                    )
                                else:
                                    keep_train_samples = True
                                    sample_reason = "pool_keep"

                            self.logger.info(
                                f"[POOL_OPPONENT_SAMPLE] episode={self.episode_cnt}, "
                                f"lineup={lineup_tuple}, "
                                f"frame_no={frame_no}, "
                                f"length_bucket={length_bucket}, "
                                f"winner_hero_id={winner_hero_id}, "
                                f"direnjie_win={direnjie_win}, "
                                f"pool_opponent_agent={self.current_pool_opponent_agent}, "
                                f"direnjie_side={self.current_pool_direnjie_side}, "
                                f"luban_side={self.current_pool_luban_side}, "
                                f"random_drop_roll={random_drop_roll}, "
                                f"sample_reason={sample_reason}, "
                                f"keep_train_samples={keep_train_samples}"
                            )
                        else:
                            (
                                mixed_sample_bucket,
                                length_bucket,
                                direnjie_win,
                                sample_reason,
                            ) = self._get_mixed_sample_bucket(
                                winner_hero_id=winner_hero_id,
                                frame_no=frame_no,
                            )

                            if mixed_sample_bucket is None:
                                keep_train_samples = False
                            elif self.last_mixed_sample_bucket is None:
                                keep_train_samples = True
                                self.last_mixed_sample_bucket = mixed_sample_bucket
                            elif mixed_sample_bucket != self.last_mixed_sample_bucket:
                                # 只接受和上一次采样类别不同的混合局，近似形成：
                                # good : mid_loss = 1 : 1
                                # 其中 good = 狄仁杰赢 + 狄仁杰长局输。
                                keep_train_samples = True
                                self.last_mixed_sample_bucket = mixed_sample_bucket
                            else:
                                keep_train_samples = False

                            self.logger.info(
                                f"[MIXED_BALANCE] episode={self.episode_cnt}, "
                                f"lineup={lineup_tuple}, "
                                f"frame_no={frame_no}, "
                                f"length_bucket={length_bucket}, "
                                f"winner_hero_id={winner_hero_id}, "
                                f"direnjie_win={direnjie_win}, "
                                f"mixed_sample_bucket={mixed_sample_bucket}, "
                                f"sample_reason={sample_reason}, "
                                f"keep_train_samples={keep_train_samples}, "
                                f"last_mixed_sample_bucket={self.last_mixed_sample_bucket}"
                            )

                    for i, terminal_reward in enumerate(terminal_rewards):
                        if terminal_reward != 0.0 and "reward" in observation[str(i)]:
                            observation[str(i)]["reward"]["win_loss"] = terminal_reward
                            observation[str(i)]["reward"]["reward_sum"] += terminal_reward
                            reward_sum_list[i] += terminal_reward
                            
                    self.logger.info(
                        f"episode_{self.episode_cnt} terminated in fno_{frame_no}, "
                        f"truncated:{truncated}, eval:{is_eval}, "
                        f"terminal_rewards:{terminal_rewards}, "
                        f"reward_sum:{reward_sum_list[monitor_side]}"
                    )
                    
                    # Reward for saving the last state of the environment
                    # 保存环境最后状态的reward
                    for i, (do_sample, agent) in enumerate(zip(self.do_samples, self.agents)):
                        if keep_train_samples and not is_eval and do_sample:
                            frame_collector.save_last_frame(
                                agent_id=i,
                                reward=observation[str(i)]["reward"]["reward_sum"],
                            )
                            

                    now = time.time()
                    if now - self.last_report_monitor_time >= 60:
                        monitor_data = {"episode_cnt": self.episode_cnt}
                        if self.monitor:
                            if is_eval:
                                monitor_data["reward"] = round(reward_sum_list[monitor_side], 2)
                            self.monitor.put_data({os.getpid(): monitor_data})
                            self.last_report_monitor_time = now

                    if keep_train_samples and len(frame_collector) > 0 and not is_eval:
                        list_agents_samples = sample_process(frame_collector)

                        lineup_tuple = tuple(self.current_lineup)
                        if lineup_tuple in ((112, 133), (133, 112)):
                            if self.current_use_pool_opponent:
                                self.train_direnjie_luban_pool_cnt += 1
                                train_source = "pool"
                            else:
                                self.train_direnjie_luban_selfplay_cnt += 1
                                train_source = "selfplay"

                            pool_cnt = self.train_direnjie_luban_pool_cnt
                            selfplay_cnt = self.train_direnjie_luban_selfplay_cnt
                            total_cnt = pool_cnt + selfplay_cnt

                            pool_rate = pool_cnt / total_cnt if total_cnt > 0 else 0.0
                            selfplay_rate = selfplay_cnt / total_cnt if total_cnt > 0 else 0.0
                            pool_to_selfplay_ratio = (
                                pool_cnt / selfplay_cnt if selfplay_cnt > 0 else float("inf")
                            )

                        self.logger.info(
                                f"[DIRENJIE_LUBAN_TRAIN_SOURCE_COUNT] "
                                f"pool_cnt={self.train_direnjie_luban_pool_cnt}, "
                                f"selfplay_cnt={self.train_direnjie_luban_selfplay_cnt}, "
                        )

                        yield list_agents_samples
                    break

    def reset_agents(self, observation, episode_cnt):
        opponent_agent = self.env_conf_manager.get_opponent_agent()

        monitor_side = self.env_conf_manager.get_monitor_side()
        is_train_test = os.environ.get("is_train_test", "False").lower() == "true"

        # The 'do_predicts' specifies which agents are to perform model predictions.
        # do_predicts 指定哪些智能体进行模型预测
        # The 'do_samples' specifies which agents are to perform training sampling.
        # do_samples 指定哪些智能体进行训练采样；只有 do_samples=True 的样本会进入 PPO。
        self.do_predicts = [True, True]
        self.do_samples = [True, True]

        lineup_tuple = tuple(self.current_lineup)

        # Load model according to the configuration
        # 根据对局配置加载模型
        for i, agent in enumerate(self.agents):
            # 模型池局特殊处理：
            # 狄仁杰侧加载 latest 并采样训练；鲁班侧加载模型池 opponent，不采样。
            # 这样即使 lineup=(112,133)，也能保证进入训练的是狄仁杰样本。
            if self.current_use_pool_opponent:
                if i == self.current_pool_direnjie_side:
                    agent.load_model(id="latest")
                    self.do_predicts[i] = True
                    self.do_samples[i] = True

                    self.logger.info(
                        f"[POOL_OPPONENT_LOAD] side={i}, hero=133, "
                        f"model=latest, do_sample=True"
                    )
                elif i == self.current_pool_luban_side:
                    if is_train_test:
                        self.logger.info(
                            "Run train_test with pool opponent selected, "
                            "replace pool opponent with latest model"
                        )
                        agent.load_model(id="latest")
                    else:
                        agent.load_opponent_agent(id=str(self.current_pool_opponent_agent))

                    self.do_predicts[i] = True
                    self.do_samples[i] = False

                    self.logger.info(
                        f"[POOL_OPPONENT_LOAD] side={i}, hero=112, "
                        f"model_id={self.current_pool_opponent_agent}, do_sample=False"
                    )
                else:
                    # 理论上 1v1 不会进入这里，防御性处理。
                    agent.load_model(id="latest")
                    self.do_predicts[i] = True
                    self.do_samples[i] = False

                    self.logger.info(
                        f"[POOL_OPPONENT_LOAD] side={i}, hero=unknown, "
                        f"fallback=latest, do_sample=False"
                    )

                obs = observation[str(i)]
                obs["episode_cnt"] = episode_cnt
                agent.reset(obs)
                continue

            # Report the latest model in the training camp to the monitor
            # 训练中最新模型所在阵营上报监控
            if i == monitor_side:
                # monitor_side uses the latest model
                # monitor_side 使用最新模型
                agent.load_model(id="latest")
            else:
                if opponent_agent == "common_ai":
                    # common_ai does not need to load a model, no need to predict
                    # 如果对手是 common_ai 则不需要加载模型，也不需要进行预测
                    self.do_predicts[i] = False
                    self.do_samples[i] = False
                elif opponent_agent == "selfplay":
                    # Training model, "latest" - latest model, "random" - random model from the model pool
                    # 加载训练过的模型，可以选择最新模型，也可以选择随机模型 "latest" - 最新模型, "random" - 模型池
                    agent.load_model(id="latest")
                else:
                    # Opponent model, model_id is checked from kaiwu.json
                    # 选择 kaiwu.json 中设置的对手模型，model_id 即 opponent_agent
                    eval_candidate_model = get_valid_model_pool(self.logger)
                    if int(opponent_agent) not in eval_candidate_model:
                        raise Exception(f"opponent_agent model_id {opponent_agent} not in {eval_candidate_model}")
                    else:
                        if is_train_test:
                            # Run train_test, cannot get opponent agent, so replace with latest model
                            # 运行 train_test 时，无法获取对手模型，因此替换为 latest
                            self.logger.info(f"Run train_test, cannot get opponent agent, so replace with latest model")
                            agent.load_model(id="latest")
                        else:
                            agent.load_opponent_agent(id=opponent_agent)
                        self.do_samples[i] = False

            # Reset agent
            # 重置 agent
            obs = observation[str(i)]
            obs["episode_cnt"] = episode_cnt
            agent.reset(obs)
