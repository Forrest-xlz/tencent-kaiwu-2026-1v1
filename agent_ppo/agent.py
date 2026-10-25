#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import os
import random
from agent_ppo.model.model import Model
from agent_ppo.feature.definition import *
import numpy as np
from kaiwudrl.interface.agent import BaseAgent

from agent_ppo.conf.conf import Config
from agent_ppo.feature.reward_process import GameRewardManager
from torch.optim.lr_scheduler import LambdaLR
from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.feature.feature_process import FeatureProcess


# Available summoner skills / 可选召唤师技能
"""
SUMMONER_SKILL_MAP = {
    80102: "治疗",
    80109: "疾跑",
    80104: "惩击",
    80108: "终结",
    80110: "狂暴",
    80105: "干扰",
    80103: "晕眩",
    80107: "净化",
    80121: "弱化",
    80115: "闪现",
}
"""
SUMMONER_SKILL_MAP = {
    80110: "狂暴",
}
SUMMONER_SKILL_IDS = list(SUMMONER_SKILL_MAP.keys())


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        self.cur_model_name = ""
        self.device = device
        # Create Model and convert the model to achannel-last memory format to achieve better performance.
        # 创建模型, 将模型转换为通道后内存格式，以获得更好的性能。
        self.model = Model().to(self.device)
        self.model = self.model.to(memory_format=torch.channels_last)

        # config info
        # 配置信息
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.lstm_hidden = np.zeros([self.lstm_unit_size])
        self.lstm_cell = np.zeros([self.lstm_unit_size])
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE

        # env info
        # 环境信息
        self.hero_camp = 0
        self.hero_id = None
        self.player_id = 0
        self.env_id = None

        # learning info
        # 学习信息
        self.train_step = 0
        self.lr = Config.INIT_LEARNING_RATE_START
        parameters = self.model.parameters()
        self.optimizer = torch.optim.Adam(params=parameters, lr=self.lr, betas=(0.9, 0.999), eps=1e-8)
        self.parameters = [p for param_group in self.optimizer.param_groups for p in param_group["params"]]
        self.target_lr = Config.TARGET_LR
        self.target_step = Config.TARGET_STEP
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=self.lr_lambda)

        # tools
        # 工具
        self.reward_manager = None
        self.logger = logger
        self.monitor = monitor

        self.algorithm = Algorithm(self.model, self.optimizer, self.scheduler, self.device, self.logger, self.monitor)

        # 当前帧英雄状态信息
        self.luban_enemy_dist_lt_10000 = False

        super().__init__(agent_type, device, logger, monitor)

    def lr_lambda(self, step):
        # Define learning rate decay function
        # 定义学习率衰减函数
        if step <= 200:
            return 0.0
            
        if step > self.target_step:
            return self.target_lr / self.lr
        else:
            return 1.0 - ((1.0 - self.target_lr / self.lr) * step / self.target_step)
    
    def lr_lambda_pretrain(self, step):
        # 前 target_step 个 scheduler step，lr = 0
        # 之后 lr = target_lr
        if step <= self.target_step:
            return 0.0
        else:
            return self.target_lr / self.lr

    def init_config(self, config_data):
        # Select summoner skill for each hero based on hero lineup of both camps
        # 根据双方阵营英雄阵容，为己方每个英雄选择召唤师技能
        my_heroes = config_data.get("my_heroes", [])
        select_skills = {}
        for hero_id in my_heroes:
            skill_id = random.choice(SUMMONER_SKILL_IDS)
            select_skills[hero_id] = skill_id
        return select_skills

    def reset(self, observation):
        # Reset function, called at the beginning of each episode
        # 重置函数，每局开始时调用

        episode_cnt = observation.get("episode_cnt", 0)
        
        self.hero_camp = observation["camp"]
        frame_state = observation["frame_state"]

        for hero in frame_state["hero_states"]:
            if hero["camp"] == self.hero_camp:
                self.hero_id = hero["config_id"]

        self.player_id = observation["player_id"]

        self.lstm_hidden = np.zeros([self.lstm_unit_size])
        self.lstm_cell = np.zeros([self.lstm_unit_size])

        self.reward_manager = GameRewardManager(self.player_id, episode_cnt)
        self.feature_processes = FeatureProcess(self.hero_camp)

        reward_weights = self.reward_manager.current_reward_weight_dict
        time_scale_arg = self.reward_manager.current_time_scale_arg

        if self.logger:
            self.logger.info(
                f"[REWARD_SCHEDULE] episode={episode_cnt}, "
                f"hero_camp={self.hero_camp}, "
                f"hero_id={self.hero_id}, "
                f"time_scale_arg={time_scale_arg}, "
                f"weights={reward_weights}"
            )

        if self.monitor:
            monitor_data = {
                "episode_cnt": episode_cnt,
                "time_scale_arg": time_scale_arg,
                "reward_weight_forward": reward_weights.get("forward", 0.0),
                "reward_weight_money": reward_weights.get("money", 0.0),
                "reward_weight_exp": reward_weights.get("exp", 0.0),
                "reward_weight_hp_point": reward_weights.get("hp_point", 0.0),
                "reward_weight_ep_rate": reward_weights.get("ep_rate", 0.0),
                "reward_weight_kill": reward_weights.get("kill", 0.0),
                "reward_weight_death": reward_weights.get("death", 0.0),
                "reward_weight_tower_hp_point": reward_weights.get("tower_hp_point", 0.0),
                "reward_weight_last_hit": reward_weights.get("last_hit", 0.0),
                "reward_weight_monster_kill": reward_weights.get("monster_kill", 0.0),
                "reward_weight_monster_first_hit": reward_weights.get("monster_first_hit", 0.0),
                "reward_weight_wave_hit_enemy_hero": reward_weights.get("wave_hit_enemy_hero", 0.0),
                "rw_dj_ult_kill_hit": reward_weights.get("direnjie_ult_kill_after_hit", 0.0),
                "rw_dj_ult_miss": reward_weights.get("direnjie_ult_miss", 0.0),

                # 狄仁杰大招塔下反击权重
                "rw_dj_ult_tower_hit": reward_weights.get("direnjie_ult_hit_under_tower", 0.0),

                "rw_rage_kill": reward_weights.get("rage_kill_after_use", 0.0),
                "rw_rage_no_kill": reward_weights.get("rage_no_kill_after_use", 0.0),
                "rw_rage_use_pen": reward_weights.get("rage_use_penalty", 0.0),

                # 恢复使用惩罚权重
                "rw_recover_use_pen": reward_weights.get("recover_use_penalty", 0.0),

                "rw_grass_ambush": reward_weights.get("grass_ambush_kill", 0.0),

                # 狄仁杰塔外消耗敌方塔内英雄权重
                "rw_dj_tower_poke": reward_weights.get("direnjie_ult_tower_poke", 0.0),

                # 狄仁杰 4 级大招命中后击杀 3 级敌方权重
                "rw_dj_lv4_kill": reward_weights.get("direnjie_ult_level4_kill", 0.0),
            }

            self.monitor.put_data({os.getpid(): monitor_data})

    def _model_inference(self, list_obs_data):
        # Using the network for inference
        # 使用网络进行推理
        feature = [obs_data.feature for obs_data in list_obs_data]
        legal_action = [obs_data.legal_action for obs_data in list_obs_data]
        lstm_cell = [obs_data.lstm_cell for obs_data in list_obs_data]
        lstm_hidden = [obs_data.lstm_hidden for obs_data in list_obs_data]

        input_list = [np.array(feature), np.array(lstm_cell), np.array(lstm_hidden)]
        torch_inputs = [torch.from_numpy(nparr).to(torch.float32) for nparr in input_list]
        for i, data in enumerate(torch_inputs):
            data = data.reshape(-1)
            torch_inputs[i] = data.float()

        feature, lstm_cell, lstm_hidden = torch_inputs
        feature_vec = feature.reshape(-1, self.seri_vec_split_shape[0][0])
        lstm_hidden_state = lstm_hidden.reshape(-1, self.lstm_unit_size)
        lstm_cell_state = lstm_cell.reshape(-1, self.lstm_unit_size)

        format_inputs = [feature_vec, lstm_hidden_state, lstm_cell_state]

        self.model.set_eval_mode()
        with torch.no_grad():
            output_list = self.model(format_inputs, inference=True)

        np_output = []
        for output in output_list:
            np_output.append(output.numpy())

        logits, value, _lstm_cell, _lstm_hidden = np_output[:4]

        _lstm_cell = _lstm_cell.squeeze(axis=0)
        _lstm_hidden = _lstm_hidden.squeeze(axis=0)

        list_act_data = list()
        for i in range(len(legal_action)):
            prob, d_prob, action, d_action = self._sample_masked_action(logits[i], legal_action[i])
            list_act_data.append(
                ActData(
                    action=action,
                    d_action=d_action,
                    prob=prob,
                    d_prob=d_prob,
                    value=value,
                    lstm_cell=_lstm_cell[i],
                    lstm_hidden=_lstm_hidden[i],
                )
            )
        return list_act_data

    def predict(self, observation):
        # Prediction function, usually called during training
        # Returns a random sampling action
        # 预测函数，通常在训练时调用，返回随机采样动作
        obs_data = self.observation_process(observation)
        act_data = self._model_inference([obs_data])[0]
        self.update_status(obs_data, act_data)
        action = self.action_process(observation, act_data, True)
        return action

    def exploit(self, observation):
        # Exploitation function, usually called during evaluation
        # Returns the action with the highest probability
        # 利用函数，在评估时调用，返回最大概率动作
        obs_data = self.observation_process(observation)
        act_data = self._model_inference([obs_data])[0]
        self.update_status(obs_data, act_data)
        d_action = self.action_process(observation, act_data, False)
        return d_action

    
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
        if hero is None:
            return False

        camp_visible = hero.get("camp_visible", [])
        idx = camp - 1

        if idx < 0 or idx >= len(camp_visible):
            return False

        return bool(camp_visible[idx])

    def _get_skill_by_slot_type(self, hero, slot_type):
        """
        从 hero["skill_state"]["slot_states"] 中取指定 slot_type 的技能槽。
        slot_type=1/2/3 一般对应英雄 1/2/3 技能。
        """
        if hero is None:
            return None

        skill_states = hero.get("skill_state", {}).get("slot_states", [])

        for skill in skill_states:
            if skill.get("slot_type") == slot_type:
                return skill

        return None

    def _get_skill_cd_debug_value(self, skill):
        """
        尝试从技能字典中取剩余 CD。
        不同版本字段名可能不一样，所以这里做兼容。
        如果都取不到，返回 None，同时后面会打印完整 skill 字典。
        """
        if skill is None:
            return None

        cd_keys = [
            "cd",
            "cooldown",
            "coolDown",
            "left_cd",
            "leftCd",
            "remain_cd",
            "remainCd",
            "remaining_cd",
            "remainingCd",
            "cooldown_time",
            "cooldownTime",
            "nextCanUseFrame",
            "next_can_use_frame",
        ]

        for key in cd_keys:
            if key in skill:
                return skill.get(key)

        return None

    def _debug_enemy_skill3_cd_and_visibility(self, frame_state, main_hero, enemy_hero):
        """
        调试：打印当前帧、我方是否能看到敌方，以及敌方英雄完整 hero 信息。
        """
        if main_hero is None or enemy_hero is None:
            return

        frame_no = frame_state.get("frame_no", -1)
        main_camp = main_hero.get("camp", self.hero_camp)

        main_can_see_enemy = self._get_camp_visible(enemy_hero, main_camp)

        msg = (
            f"[ENEMY_HERO_DEBUG] "
            f"frame={frame_no}, "
            f"main_can_see_enemy={main_can_see_enemy}, "
            f"enemy_hero={enemy_hero}"
        )

        print(msg, flush=True)

        if self.logger:
            
            self.logger.info(msg)
        

    def observation_process(self, observation):
        
        feature = self.feature_processes.process_feature(observation)
        feature_vec, legal_action = (
            feature,
            observation["legal_action"],
        )
        # logger 打印一些信息
        frame_state = observation["frame_state"]
        npc_states = frame_state.get("npc_states", [])
        hero_states = frame_state['hero_states']

        main_hero = None
        enemy_hero = None

        for hero in hero_states:
            if hero['camp'] == self.hero_camp:
                main_hero = hero
                main_camp = main_hero['camp']
            else:
                enemy_hero = hero

        #self._debug_enemy_skill3_cd_and_visibility(frame_state,main_hero,enemy_hero)

        # 默认关闭，避免上一帧状态残留
        self.luban_enemy_dist_lt_10000 = False

        # 鲁班：敌我距离小于 10000 时，允许强制 1 技能朝敌方英雄中心释放
        if self.hero_id == 112 and main_hero is not None and enemy_hero is not None:
            main_loc = main_hero.get("location", {})
            enemy_loc = enemy_hero.get("location", {})

            main_x = main_loc.get("x", 0)
            main_z = main_loc.get("z", 0)
            enemy_x = enemy_loc.get("x", 0)
            enemy_z = enemy_loc.get("z", 0)

            dx = main_x - enemy_x
            dz = main_z - enemy_z
            dist = (dx * dx + dz * dz) ** 0.5

            if dist < 10000:
                self.luban_enemy_dist_lt_10000 = True

        

        monster_candidates = []
        soldier_infos = []

        for npc in npc_states:
            config_id = npc.get("config_id")
            actor_type = npc.get("actor_type")
            sub_type = npc.get("sub_type")
            camp = npc.get("camp")

            # 小兵：actor_type=1, sub_type=11
            if actor_type == 1 and sub_type == 11:
                soldier_infos.append({
                    "config_id": config_id,
                    "runtime_id": npc.get("runtime_id"),
                    "camp": camp,
                    "max_hp": npc.get("max_hp"),
                    "attack_range": npc.get("attack_range"),
                    "kill_income": npc.get("kill_income"),
                    "camp_visible": npc.get("camp_visible"),
                    
                })
                continue

            # 已知建筑/防御塔/水晶/泉水：排除
            if actor_type == 2:
                continue

            # 剩下的优先认为是疑似野怪/中立单位
            if camp == 0 or actor_type == 1:
                monster_candidates.append(npc)
        """
        hero_states = frame_state['hero_states']
        for hero in hero_states:
            if hero['camp'] == self.hero_camp:
                self.logger.info(f"主英雄。 frame:{frame_state['frame_no']}, buff:{hero['buff_state']}, 伤害承受：{hero['take_hurt_infos']}, 命中目标：{hero['hit_target_info']}, 对英雄的伤害:{hero['total_hurt_to_hero']}")
            else:
                self.logger.info(f"敌方英雄。 frame:{frame_state['frame_no']}, buff:{hero['buff_state']}, 伤害承受：{hero['take_hurt_infos']}, 命中目标：{hero['hit_target_info']}, 对英雄的伤害:{hero['total_hurt_to_hero']}")
        
        bullets = frame_state.get('bullets', None)
        if bullets is not None:
            self.logger.info(f"bullets:{bullets}")

        if len(monster_candidates)!=0:
            self.logger.info(f"frame:{frame_state['frame_no']}.{monster_candidates}")

        
        config_ids = [0,0]
        for i in range(2):
            hero = hero_states[i]
            config_ids[i] = hero['config_id']
            
        self.logger.info(f"英雄ids{config_ids}")
        """
        
                
        """
        if main_hero['camp_visible'][main_hero['camp']-1]:
            print("yes")
        else:
            print("no")
        """

        #print(f"player_id:{self.player_id}, main_hero_player_id:{main_hero['player_id']}, main_hero_runtime_id:{main_hero['runtime_id']}")

        frame_action = frame_state["frame_action"]
                # 如果发生了死亡事件
        if "dead_action" in frame_action:
                    # 取出当前的死亡事件信息
                    dead_actions = frame_action["dead_action"]
                    
                    for dead_action in dead_actions:
                        """
                        # 这个部分鼓励我方英雄去杀小兵
                        if (
                            # 杀人者-我方英雄
                            dead_action["killer"]["runtime_id"] == main_hero["runtime_id"]
                            # 死亡者-敌方小兵
                            and dead_action["death"]["actor_type"] == "1" and dead_action['death']['sub_type'] == 0:
                        ):
                            self.logger.info(f"我方英雄击杀了野怪")

                        # 这个部分鼓励我方英雄去杀小兵
                        if (
                            # 杀人者-我方英雄
                            dead_action["killer"]["runtime_id"] == main_hero["runtime_id"]
                            # 死亡者-敌方小兵
                            and dead_action["death"]["actor_type"] == "ACTOR_TYPE_SHENFU"
                        ):
                            self.logger.info(f"我方英雄吃到了buff")
                        """
                        if (dead_action['death']['actor_type'] == 1 and dead_action['death']['sub_type'] == 11) or (dead_action['death']['actor_type'] == 0 and dead_action['death']['sub_type'] == 0) or (dead_action['death']['actor_type'] == 1 and dead_action['death']['sub_type'] == 0):
                            continue
                        else:
                            #self.logger.info(f"死亡事件：{dead_actions}")
                            pass
            



        return ObsData(
            feature=feature_vec, legal_action=legal_action, lstm_cell=self.lstm_cell, lstm_hidden=self.lstm_hidden
        )

    def action_process(self, observation, act_data, is_stochastic):
        if is_stochastic:
            # Use stochastic sampling action
            # 采用随机采样动作 action
            return act_data.action
        else:
            # Use the action with the highest probability
            # 采用最大概率动作 d_action
            return act_data.d_action

    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        # To save the model, it can consist of multiple files, and it is important to ensure that
        #  each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        torch.save(self.model.state_dict(), model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        # When loading the model, you can load multiple files, and it is important to ensure that
        # each filename matches the one used during the save_model process.
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        if self.cur_model_name == model_file_path:
            self.logger.info(f"current model is {model_file_path}, so skip load model")
        else:
            self.model.load_state_dict(
                torch.load(
                    model_file_path,
                    map_location=self.device,
                )
            )
            self.cur_model_name = model_file_path
            self.logger.info(f"load model {model_file_path} successfully")

    def load_opponent_agent(self, id="1"):
        # Framework provides loading opponent agent function, no need to implement function content
        # 框架提供的加载对手模型功能，无需实现函数内容
        pass

    def update_status(self, obs_data, act_data):
        self.obs_data = obs_data
        self.act_data = act_data
        self.lstm_cell = act_data.lstm_cell
        self.lstm_hidden = act_data.lstm_hidden

    def _sample_masked_action(self, logits, legal_action):
        """
        Sample actions from predicted logits and legal actions
        return: probability, stochastic and deterministic actions with additional list
        """
        """
        从预测的logits和合法动作中采样动作
        返回：以列表形式概率、随机和确定性动作
        """

        prob_list = []
        d_prob_list = []
        action_list = []
        d_action_list = []
        label_split_size = [sum(self.label_size_list[: index + 1]) for index in range(len(self.label_size_list))]
        legal_actions = np.split(legal_action, label_split_size[:-1])
        logits_split = np.split(logits, label_split_size[:-1])
        for index in range(0, len(self.label_size_list) - 1):
            probs = self._legal_soft_max(logits_split[index], legal_actions[index])
            prob_list += list(probs)
            d_prob_list += list(probs)
            sample_action = self._legal_sample(probs, use_max=False)
            action_list.append(sample_action)
            d_action = self._legal_sample(probs, use_max=True)
            d_action_list.append(d_action)

            

        # deals with the last prediction, target
        # 处理最后的预测，目标
        index = len(self.label_size_list) - 1
        target_legal_action_o = np.reshape(
            legal_actions[index],
            [
                self.legal_action_size[0],
                self.legal_action_size[-1] // self.legal_action_size[0],
            ],
        )
        one_hot_actions = np.eye(self.label_size_list[0])[action_list[0]]
        one_hot_actions = np.reshape(one_hot_actions, [self.label_size_list[0], 1])
        target_legal_action = np.sum(target_legal_action_o * one_hot_actions, axis=0)

        
        if action_list[0] == 6 and self.hero_id == 133:
            # 强制 skill_x / skill_z 选择中心附近的中心点
            action_list[3] = 8
            action_list[4] = 8

            # 强制 target = EnemyHero
            for i in range(9):
                target_legal_action[i] = False
            target_legal_action[1] = True

        # 鲁班 skill2：以敌方英雄为目标，中心偏移
        if action_list[0] == 5 and self.hero_id == 112:
            action_list[3] = 8
            action_list[4] = 8

            for i in range(9):
                target_legal_action[i] = False
            target_legal_action[1] = True

        # 鲁班 skill1：敌方距离小于 10000 时，以敌方英雄为目标，中心偏移
        if action_list[0] == 4 and self.hero_id == 112 and self.luban_enemy_dist_lt_10000:
            action_list[3] = 8
            action_list[4] = 8

            for i in range(9):
                target_legal_action[i] = False
            target_legal_action[1] = True
                
        
        legal_actions[index] = target_legal_action
        probs = self._legal_soft_max(logits_split[-1], target_legal_action)
        prob_list += list(probs)
        sample_action = self._legal_sample(probs, use_max=False)
        action_list.append(sample_action)

        """
        # ====== 训练采样动作打印：放在 target 采样完成之后 ======
        target_name_map = {
            0: "None",
            1: "EnemyHero",
            2: "SelfHero",
            3: "Soldier0",
            4: "Soldier1",
            5: "Soldier2",
            6: "Soldier3",
            7: "Tower",
            8: "Monster",
        }

        if action_list[0] == 6 and self.hero_id == 133:
            button = action_list[0]
            move_x = action_list[1]
            move_z = action_list[2]
            skill_x = action_list[3]
            skill_z = action_list[4]
            target = action_list[5]

            msg = (
                f"[SKILL3_SAMPLE] hero_id={self.hero_id}, "
                f"button={button}, "
                f"move_x={move_x}, move_z={move_z}, "
                f"skill_x={skill_x}, skill_z={skill_z}, "
                f"target={target}({target_name_map.get(target, 'Unknown')}), "
                f"target_legal_action={target_legal_action.astype(int).tolist()}"
            )

            if self.logger:
                self.logger.info(msg)
            else:
                print(msg)
        """

        one_hot_actions = np.eye(self.label_size_list[0])[d_action_list[0]]
        one_hot_actions = np.reshape(one_hot_actions, [self.label_size_list[0], 1])
        target_legal_action_d = np.sum(target_legal_action_o * one_hot_actions, axis=0)

        
        if d_action_list[0] == 6 and self.hero_id == 133:
            # 强制 skill_x / skill_z 选择中心附近的中心点
            d_action_list[3] = 8
            d_action_list[4] = 8

            # 强制 target = EnemyHero
            for i in range(9):
                target_legal_action_d[i] = False
            target_legal_action_d[1] = True


        # 鲁班 skill2：以敌方英雄为目标，中心偏移
        if d_action_list[0] == 5 and self.hero_id == 112:
            d_action_list[3] = 8
            d_action_list[4] = 8

            for i in range(9):
                target_legal_action_d[i] = False
            target_legal_action_d[1] = True

        # 鲁班 skill1：敌方距离小于 10000 时，以敌方英雄为目标，中心偏移
        if d_action_list[0] == 4 and self.hero_id == 112 and self.luban_enemy_dist_lt_10000:
            d_action_list[3] = 8
            d_action_list[4] = 8

            for i in range(9):
                target_legal_action_d[i] = False
            target_legal_action_d[1] = True
                

        probs = self._legal_soft_max(logits_split[-1], target_legal_action_d)
        d_prob_list += list(probs)

        d_action = self._legal_sample(probs, use_max=True)
        d_action_list.append(d_action)

        return [prob_list], [d_prob_list], action_list, d_action_list

    def _legal_soft_max(self, input_hidden, legal_action):
        _lsm_const_w, _lsm_const_e = 1e20, 1e-5
        _lsm_const_e = 0.00001

        tmp = input_hidden - _lsm_const_w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        tmp = np.clip(tmp - tmp_max, -_lsm_const_w, 1)
        tmp = (np.exp(tmp) + _lsm_const_e) * legal_action
        probs = tmp / np.sum(tmp, keepdims=True)
        return probs

    def _legal_sample(self, probs, legal_action=None, use_max=False):
        # Sample with probability, input probs should be 1D array
        # 根据概率采样，输入的probs应该是一维数组
        if use_max:
            return np.argmax(probs)

        return np.argmax(np.random.multinomial(1, probs, size=1))
