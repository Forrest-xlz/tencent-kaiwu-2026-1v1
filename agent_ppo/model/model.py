#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import torch
import torch.nn as nn
from torch.nn import ModuleDict

import numpy as np
from typing import List

from agent_ppo.conf.conf import DimConfig, Config


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # feature configure parameter
        # 特征配置参数
        self.model_name = Config.NETWORK_NAME
        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE
        self.m_learning_rate = Config.INIT_LEARNING_RATE_START
        self.m_var_beta = Config.BETA_START
        self.log_epsilon = Config.LOG_EPSILON
        self.label_size_list = Config.LABEL_SIZE_LIST
        self.is_reinforce_task_list = Config.IS_REINFORCE_TASK_LIST
        self.min_policy = Config.MIN_POLICY
        self.clip_param = Config.CLIP_PARAM
        self.restore_list = []
        self.var_beta = self.m_var_beta
        self.learning_rate = self.m_learning_rate
        self.target_embed_dim = Config.TARGET_EMBED_DIM
        self.cut_points = [value[0] for value in Config.data_shapes]
        self.legal_action_size = Config.LEGAL_ACTION_SIZE_LIST

        self.feature_dim = Config.SERI_VEC_SPLIT_SHAPE[0][0]
        self.legal_action_dim = np.sum(Config.LEGAL_ACTION_SIZE_LIST)
        self.lstm_hidden_dim = Config.LSTM_UNIT_SIZE
        self.feature_dim = int(DimConfig.DIM_OF_FEATURE[0])

        # ------------------------------------------------------------------
        # Feature split order must match FeatureProcess.process_feature:
        # main hero + enemy hero + main organ + enemy organ + monster
        # + main soldiers + enemy soldiers + game
        # + main luban + enemy luban + main direnjie + enemy direnjie
        # + main_hero extra vector.
        #
        # Private hero features are NOT appended to the public trunk directly.
        # They are first concatenated to the corresponding public hero vector:
        #   main hero  = public_main_hero  + main_luban  + main_direnjie
        #   enemy hero = public_enemy_hero + enemy_luban + enemy_direnjie
        # Then the rest of the model architecture stays unchanged.
        # ------------------------------------------------------------------
        self.main_hero_dim = DimConfig.PUBLIC_MAIN_HERO_DIM
        self.enemy_hero_dim = DimConfig.PUBLIC_ENEMY_HERO_DIM
        self.main_organ_dim = DimConfig.MAIN_ORGAN_DIM
        self.enemy_organ_dim = DimConfig.ENEMY_ORGAN_DIM
        self.monster_dim = DimConfig.MONSTER_DIM
        self.main_soldier_total_dim = DimConfig.MAIN_SOLDIER_DIM
        self.enemy_soldier_total_dim = DimConfig.ENEMY_SOLDIER_DIM
        self.game_dim = DimConfig.GAME_DIM

        self.main_luban_dim = DimConfig.MAIN_LUBAN_DIM
        self.enemy_luban_dim = DimConfig.ENEMY_LUBAN_DIM
        self.main_direnjie_dim = DimConfig.MAIN_DIRENJIE_DIM
        self.enemy_direnjie_dim = DimConfig.ENEMY_DIRENJIE_DIM
        # Extra main-hero vector, following the official 1v1 style:
        # process it through an independent small MLP before concatenating
        # to the public trunk.
        self.main_hero_extra_dim = DimConfig.MAIN_HERO_DIM

        self.main_hero_input_dim = self.main_hero_dim + self.main_luban_dim + self.main_direnjie_dim
        self.enemy_hero_input_dim = self.enemy_hero_dim + self.enemy_luban_dim + self.enemy_direnjie_dim

        # Keep the hero encoder shared between main/enemy heroes.
        # With the current config, both sides must have the same input dim.
        # If this assertion fails, fix DimConfig or make main/enemy hero encoders explicitly separate.
        assert self.main_hero_input_dim == self.enemy_hero_input_dim, (
            "shared hero_mlp requires equal main/enemy hero input dims, "
            "got main={} enemy={}".format(self.main_hero_input_dim, self.enemy_hero_input_dim)
        )
        self.hero_input_dim = self.main_hero_input_dim

        self.soldier_num = 4
        self.soldier_dim = self.main_soldier_total_dim // self.soldier_num
        assert self.main_soldier_total_dim == self.soldier_num * self.soldier_dim
        assert self.enemy_soldier_total_dim == self.soldier_num * self.soldier_dim

        # ------------------------------------------------------------------
        # Object encoders, following the official 1v1 baseline style:
        # shared object MLP -> side-specific FC -> pool for multi-object groups.
        # ------------------------------------------------------------------
        self.hero_mlp = MLP([self.hero_input_dim, 512, 256], "hero_mlp", non_linearity_last=True)
        self.main_hero_fc = MLP([256, 128], "main_hero_fc", non_linearity_last=True)
        self.enemy_hero_fc = MLP([256, 128], "enemy_hero_fc", non_linearity_last=True)

        # Independent main_hero vector branch: 20 -> 64 -> 32 -> 16.
        # This mirrors the official 1v1 main_vector treatment and avoids
        # mixing main_hero identity/context features directly into object encoders.
        self.hero_main_mlp = MLP([self.main_hero_extra_dim, 64, 32, 16], "hero_main_mlp", non_linearity_last=True)

        self.soldier_mlp = MLP([self.soldier_dim, 64, 64], "soldier_mlp", non_linearity_last=True)
        self.main_soldier_fc = MLP([64, 32], "main_soldier_fc", non_linearity_last=True)
        self.enemy_soldier_fc = MLP([64, 32], "enemy_soldier_fc", non_linearity_last=True)

        # Tower / organ only has one slot in your feature, so no max-pooling is used.
        self.organ_mlp = MLP([self.main_organ_dim, 64, 64], "organ_mlp", non_linearity_last=True)
        self.main_organ_fc = MLP([64, 32], "main_organ_fc", non_linearity_last=True)
        self.enemy_organ_fc = MLP([64, 32], "enemy_organ_fc", non_linearity_last=True)

        # Extra object in your current feature set. Encode it like a single object.
        self.monster_mlp = MLP([self.monster_dim, 64, 64], "monster_mlp", non_linearity_last=True)
        self.monster_fc = MLP([64, 32], "monster_fc", non_linearity_last=True)

        # public trunk concat dim:
        # main hero 128 + enemy hero 128
        # + main soldier pool 32 + enemy soldier pool 32
        # + main organ 32 + enemy organ 32
        # + monster 32 + game
        # + independent main_hero vector embedding 16
        self.public_concat_dim = 128 + 128 + 32 + 32 + 32 + 32 + 32 + self.game_dim + 16
        self.concat_mlp = MLP(
            [self.public_concat_dim, self.lstm_unit_size],
            "concat_mlp",
            non_linearity_last=True,
        )

        self.lstm = torch.nn.LSTM(
            input_size=self.lstm_unit_size,
            hidden_size=self.lstm_unit_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0,
            bidirectional=False,
        )

        # First 5 policy heads are normal categorical heads.
        # Last target head is produced by dot-product with target embeddings.
        self.label_mlp = ModuleDict(
            {
                "hero_label{0}_mlp".format(label_index): MLP(
                    [self.lstm_unit_size, self.label_size_list[label_index]],
                    "hero_label{0}_mlp".format(label_index),
                )
                for label_index in range(len(self.label_size_list) - 1)
            }
        )

        self.lstm_tar_embed_mlp = make_fc_layer(self.lstm_unit_size, self.target_embed_dim)
        self.target_embed_mlp = make_fc_layer(self.target_embed_dim, self.target_embed_dim, use_bias=False)

        self.value_mlp = MLP([self.lstm_unit_size, 64, 1], "hero_value_mlp")

    def _split_feature(self, feature_vec):
        split_size = [
            self.main_hero_dim,
            self.enemy_hero_dim,
            self.main_organ_dim,
            self.enemy_organ_dim,
            self.monster_dim,
            self.main_soldier_total_dim,
            self.enemy_soldier_total_dim,
            self.game_dim,
            self.main_luban_dim,
            self.enemy_luban_dim,
            self.main_direnjie_dim,
            self.enemy_direnjie_dim,
            self.main_hero_extra_dim,
        ]

        expected_dim = sum(split_size)
        if feature_vec.shape[1] != expected_dim:
            raise RuntimeError(
                "feature dim mismatch: got {}, expected {}. "
                "Check FeatureProcess concat order and DimConfig.".format(feature_vec.shape[1], expected_dim)
            )

        return torch.split(feature_vec, split_size, dim=1)

    def _reshape_lstm_state(self, state):
        # state is expected to be [batch, 512]. Keep this helper tolerant to
        # extra dimensions from framework wrappers.
        return state.reshape(-1, self.lstm_unit_size).unsqueeze(0).contiguous()

    def forward(self, data_list, inference=False):
        feature_vec, lstm_hidden_init, lstm_cell_init = data_list

        (
            main_hero_vec,
            enemy_hero_vec,
            main_organ_vec,
            enemy_organ_vec,
            monster_vec,
            main_soldier_vec,
            enemy_soldier_vec,
            game_vec,
            main_luban_vec,
            enemy_luban_vec,
            main_direnjie_vec,
            enemy_direnjie_vec,
            main_hero_extra_vec,
        ) = self._split_feature(feature_vec)

        batch_time = feature_vec.shape[0]

        # hero encoders
        # Fold hero-private features into the corresponding public hero vector.
        main_hero_input = torch.cat([main_hero_vec, main_luban_vec, main_direnjie_vec], dim=1)
        enemy_hero_input = torch.cat([enemy_hero_vec, enemy_luban_vec, enemy_direnjie_vec], dim=1)

        # The shared hero_mlp consumes main/enemy public + private hero features.

        main_hero_shared = self.hero_mlp(main_hero_input)
        enemy_hero_shared = self.hero_mlp(enemy_hero_input)
        main_hero_out = self.main_hero_fc(main_hero_shared)      # [B*T, 128]
        enemy_hero_out = self.enemy_hero_fc(enemy_hero_shared)   # [B*T, 128]

        # independent main_hero vector branch
        hero_main_result = self.hero_main_mlp(main_hero_extra_vec)  # [B*T, 16]

        # soldier encoders
        main_soldier = main_soldier_vec.reshape(batch_time, self.soldier_num, self.soldier_dim)
        enemy_soldier = enemy_soldier_vec.reshape(batch_time, self.soldier_num, self.soldier_dim)

        main_soldier_shared = self.soldier_mlp(main_soldier.reshape(-1, self.soldier_dim))
        enemy_soldier_shared = self.soldier_mlp(enemy_soldier.reshape(-1, self.soldier_dim))

        main_soldier_out = self.main_soldier_fc(main_soldier_shared).reshape(
            batch_time, self.soldier_num, self.target_embed_dim
        )
        enemy_soldier_out = self.enemy_soldier_fc(enemy_soldier_shared).reshape(
            batch_time, self.soldier_num, self.target_embed_dim
        )

        main_soldier_pool = torch.max(main_soldier_out, dim=1).values    # [B*T, 32]
        enemy_soldier_pool = torch.max(enemy_soldier_out, dim=1).values  # [B*T, 32]

        # organ encoders; only one organ slot, so no max-pooling.
        main_organ_out = self.main_organ_fc(self.organ_mlp(main_organ_vec))    # [B*T, 32]
        enemy_organ_out = self.enemy_organ_fc(self.organ_mlp(enemy_organ_vec)) # [B*T, 32]

        # monster encoder
        monster_out = self.monster_fc(self.monster_mlp(monster_vec))          # [B*T, 32]

        # public trunk
        public_concat = torch.cat(
            [
                main_hero_out,
                enemy_hero_out,
                main_soldier_pool,
                enemy_soldier_pool,
                main_organ_out,
                enemy_organ_out,
                monster_out,
                game_vec,
                hero_main_result,
            ],
            dim=1,
        )
        fc_public_result = self.concat_mlp(public_concat)  # [B*T, 512]

        # LSTM
        lstm_input = fc_public_result.reshape(-1, self.lstm_time_steps, self.lstm_unit_size)
        h0 = self._reshape_lstm_state(lstm_hidden_init)
        c0 = self._reshape_lstm_state(lstm_cell_init)

        lstm_output, (hn, cn) = self.lstm(lstm_input, (h0, c0))
        lstm_output = lstm_output.reshape(-1, self.lstm_unit_size)

        self.lstm_hidden_output = hn
        self.lstm_cell_output = cn

        result_list = []

        # normal policy heads: label0 ~ label4
        for label_index in range(len(self.label_size_list) - 1):
            label_mlp_out = self.label_mlp["hero_label{0}_mlp".format(label_index)](lstm_output)
            result_list.append(label_mlp_out)

        # target head: label5
        # 9 target slots for current config:
        # 0: empty/default target
        # 1: enemy hero
        # 2: main hero
        # 3-6: enemy soldiers 1-4
        # 7: enemy organ/tower
        # 8: monster
        tar_embed_empty = 0.1 * torch.ones_like(enemy_hero_out[:, -self.target_embed_dim :])
        tar_embed_list = [
            tar_embed_empty,
            enemy_hero_out[:, -self.target_embed_dim :],
            main_hero_out[:, -self.target_embed_dim :],
            enemy_soldier_out[:, 0, :],
            enemy_soldier_out[:, 1, :],
            enemy_soldier_out[:, 2, :],
            enemy_soldier_out[:, 3, :],
            enemy_organ_out,
            monster_out,
        ]

        target_label_size = self.label_size_list[-1]
        if len(tar_embed_list) != target_label_size:
            raise RuntimeError(
                "target embedding count ({}) != target label size ({}). "
                "Please update tar_embed_list or Config.LABEL_SIZE_LIST[-1].".format(
                    len(tar_embed_list), target_label_size
                )
            )

        tar_embed = torch.stack(tar_embed_list, dim=1)  # [B*T, 9, 32]
        tar_embed = self.target_embed_mlp(tar_embed.reshape(-1, self.target_embed_dim)).reshape(
            -1, target_label_size, self.target_embed_dim
        )
        tar_query = self.lstm_tar_embed_mlp(lstm_output).unsqueeze(1)  # [B*T, 1, 32]
        target_logits = torch.sum(tar_embed * tar_query, dim=2)        # [B*T, 9]
        result_list.append(target_logits)

        # value head
        value_result = self.value_mlp(lstm_output)
        result_list.append(value_result)

        # prepare for infer graph
        # 准备推理图
        logits = torch.flatten(torch.cat(result_list[:-1], 1), start_dim=1)
        value = result_list[-1]

        if inference:
            # Preserve the original return order: logits, value, cell, hidden.
            return [logits, value, self.lstm_cell_output, self.lstm_hidden_output]
        else:
            return result_list

    def compute_loss(self, data_list, rst_list):
        seri_vec = data_list[0].reshape(-1, self.data_split_shape[0])
        usq_reward = data_list[1].reshape(-1, self.data_split_shape[1])
        usq_advantage = data_list[2].reshape(-1, self.data_split_shape[2])
        usq_is_train = data_list[-3].reshape(-1, self.data_split_shape[-3])

        usq_label_list = data_list[3 : 3 + len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            usq_label_list[shape_index] = (
                usq_label_list[shape_index].reshape(-1, self.data_split_shape[3 + shape_index]).long()
            )

        old_label_probability_list = data_list[3 + len(self.label_size_list) : 3 + 2 * len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            old_label_probability_list[shape_index] = old_label_probability_list[shape_index].reshape(
                -1, self.data_split_shape[3 + len(self.label_size_list) + shape_index]
            )

        usq_weight_list = data_list[3 + 2 * len(self.label_size_list) : 3 + 3 * len(self.label_size_list)]
        for shape_index in range(len(self.label_size_list)):
            usq_weight_list[shape_index] = usq_weight_list[shape_index].reshape(
                -1,
                self.data_split_shape[3 + 2 * len(self.label_size_list) + shape_index],
            )

        # squeeze tensor
        # 压缩张量
        reward = usq_reward.squeeze(dim=1)
        advantage = usq_advantage.squeeze(dim=1)

        # 保留原始 advantage，用于 monitor。
        # 如果以后你想做 advantage normalization，也应该先记录 raw_advantage。
        raw_advantage = advantage.detach()

        label_list = []
        for ele in usq_label_list:
            label_list.append(ele.squeeze(dim=1))
        weight_list = []
        for weight in usq_weight_list:
            weight_list.append(weight.squeeze(dim=1))
        frame_is_train = usq_is_train.squeeze(dim=1)

        label_result = rst_list[:-1]
        value_result = rst_list[-1]

        _, split_feature_legal_action = torch.split(
            seri_vec,
            [
                np.prod(self.seri_vec_split_shape[0]),
                np.prod(self.seri_vec_split_shape[1]),
            ],
            dim=1,
        )
        feature_legal_action_shape = list(self.seri_vec_split_shape[1])
        feature_legal_action_shape.insert(0, -1)
        feature_legal_action = split_feature_legal_action.reshape(feature_legal_action_shape)

        legal_action_flag_list = torch.split(feature_legal_action, self.label_size_list, dim=1)

        # loss of value net
        # 值网络的损失
        fc2_value_result_squeezed = value_result.squeeze(dim=1)
        self.value_cost = 0.5 * torch.mean(torch.square(reward - fc2_value_result_squeezed), dim=0)
        new_advantage = reward - fc2_value_result_squeezed
        self.value_cost = 0.5 * torch.mean(torch.square(new_advantage), dim=0)

        with torch.no_grad():
            train_mask = frame_is_train > 0

            if torch.any(train_mask):
                adv_valid = raw_advantage[train_mask]
                adv_mean = adv_valid.mean()
                adv_std = adv_valid.std(unbiased=False)

                value_target = reward[train_mask]
                value_pred = fc2_value_result_squeezed.detach()[train_mask]

                var_y = torch.var(value_target, unbiased=False)
                var_err = torch.var(value_target - value_pred, unbiased=False)

                if var_y > 1e-8:
                    explained_variance = 1.0 - var_err / (var_y + 1e-8)
                else:
                    explained_variance = torch.tensor(0.0, device=value_target.device)
            else:
                adv_mean = torch.tensor(0.0, device=reward.device)
                adv_std = torch.tensor(0.0, device=reward.device)
                explained_variance = torch.tensor(0.0, device=reward.device)

        clipfrac_list = []
        approx_kl_list = []
        valid_count_list = []

        # for entropy loss calculate
        # 用于熵损失计算
        label_probability_list = []

        epsilon = 1e-5

        # policy loss: ppo clip loss
        # 策略损失：PPO剪辑损失
        self.policy_cost = torch.tensor(0.0, device=reward.device)
        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                final_log_p = torch.tensor(0.0, device=reward.device)
                boundary = torch.pow(
                    torch.tensor(10.0, device=reward.device),
                    torch.tensor(20.0, device=reward.device),
                )
                one_hot_actions = nn.functional.one_hot(label_list[task_index].long(), self.label_size_list[task_index])

                legal_action_flag_list_max_mask = (1 - legal_action_flag_list[task_index]) * boundary

                label_logits_subtract_max = torch.clamp(
                    label_result[task_index]
                    - torch.max(
                        label_result[task_index] - legal_action_flag_list_max_mask,
                        dim=1,
                        keepdim=True,
                    ).values,
                    -boundary,
                    1,
                )

                label_exp_logits = (
                    legal_action_flag_list[task_index] * torch.exp(label_logits_subtract_max) + self.min_policy
                )

                label_sum_exp_logits = label_exp_logits.sum(1, keepdim=True)

                label_probability = 1.0 * label_exp_logits / label_sum_exp_logits
                label_probability_list.append(label_probability)

                policy_p = (one_hot_actions * label_probability).sum(1)
                policy_log_p = torch.log(policy_p + epsilon)
                old_policy_p = (one_hot_actions * old_label_probability_list[task_index] + epsilon).sum(1)
                old_policy_log_p = torch.log(old_policy_p)
                final_log_p = final_log_p + policy_log_p - old_policy_log_p
                ratio = torch.exp(final_log_p)

                with torch.no_grad():
                    valid_weight = weight_list[task_index].float() * frame_is_train
                    valid_count = valid_weight.sum()
                    denom = torch.clamp(valid_count, min=1.0)

                    log_ratio = final_log_p.detach()
                    ratio_detached = ratio.detach()

                    # PPO clipfrac: 有多少样本的 ratio 超过 [1-eps, 1+eps]
                    clipfrac = (
                        (((ratio_detached < 1.0 - self.clip_param) | (ratio_detached > 1.0 + self.clip_param)).float()
                        * valid_weight).sum()
                        / denom
                    )

                    # stable approx KL: E[(ratio - 1) - log_ratio]
                    approx_kl = (((ratio_detached - 1.0) - log_ratio) * valid_weight).sum() / denom

                    clipfrac_list.append(clipfrac)
                    approx_kl_list.append(approx_kl)
                    valid_count_list.append(valid_count)

                clip_ratio = ratio.clamp(0.0, 3.0)

                surr1 = clip_ratio * advantage
                surr2 = ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param) * advantage
                denom = torch.maximum(
                    torch.sum((weight_list[task_index].float()) * frame_is_train),
                    torch.tensor(1.0, device=reward.device),
                )
                temp_policy_loss = -torch.sum(
                    torch.minimum(surr1, surr2) * (weight_list[task_index].float()) * frame_is_train
                ) / denom

                self.policy_cost = self.policy_cost + temp_policy_loss

        # cross entropy loss
        # 交叉熵损失
        current_entropy_loss_index = 0
        entropy_loss_list = []
        for task_index in range(len(self.is_reinforce_task_list)):
            if self.is_reinforce_task_list[task_index]:
                temp_entropy_loss = -torch.sum(
                    label_probability_list[current_entropy_loss_index]
                    * legal_action_flag_list[task_index]
                    * torch.log(label_probability_list[current_entropy_loss_index] + epsilon),
                    dim=1,
                )

                denom = torch.maximum(
                    torch.sum(weight_list[task_index].float() * frame_is_train),
                    torch.tensor(1.0, device=reward.device),
                )
                temp_entropy_loss = -torch.sum(
                    (temp_entropy_loss * weight_list[task_index].float() * frame_is_train)
                ) / denom

                entropy_loss_list.append(temp_entropy_loss)
                current_entropy_loss_index = current_entropy_loss_index + 1
            else:
                temp_entropy_loss = torch.tensor(0.0, device=reward.device)
                entropy_loss_list.append(temp_entropy_loss)

        self.entropy_cost = torch.tensor(0.0, device=reward.device)
        for entropy_element in entropy_loss_list:
            self.entropy_cost = self.entropy_cost + entropy_element

        self.entropy_cost_list = entropy_loss_list

        self.loss = self.value_cost + self.policy_cost + self.var_beta * self.entropy_cost

        with torch.no_grad():
            if len(valid_count_list) > 0:
                total_valid_count = torch.stack(valid_count_list).sum()
                if total_valid_count > 0:
                    clipfrac_mean = (
                        torch.stack([c * w for c, w in zip(clipfrac_list, valid_count_list)]).sum()
                        / total_valid_count
                    )
                    approx_kl_mean = (
                        torch.stack([k * w for k, w in zip(approx_kl_list, valid_count_list)]).sum()
                        / total_valid_count
                    )
                else:
                    clipfrac_mean = torch.tensor(0.0, device=reward.device)
                    approx_kl_mean = torch.tensor(0.0, device=reward.device)
            else:
                clipfrac_mean = torch.tensor(0.0, device=reward.device)
                approx_kl_mean = torch.tensor(0.0, device=reward.device)

        self.extra_metrics = {
            "explained_variance": explained_variance.detach(),
            "clipfrac": clipfrac_mean.detach(),
            "approx_kl": approx_kl_mean.detach(),
            "adv_mean": adv_mean.detach(),
            "adv_std": adv_std.detach(),
        }
        return self.loss, [
            self.loss,
            [self.value_cost, self.policy_cost, self.entropy_cost],
        ]

    def set_train_mode(self):
        self.lstm_time_steps = Config.LSTM_TIME_STEPS
        self.train()

    def set_eval_mode(self):
        self.lstm_time_steps = 1
        self.eval()


def make_fc_layer(in_features: int, out_features: int, use_bias=True):
    fc_layer = nn.Linear(in_features, out_features, bias=use_bias)

    nn.init.orthogonal(fc_layer.weight)
    if use_bias:
        nn.init.zeros_(fc_layer.bias)

    return fc_layer


class MLP(nn.Module):
    def __init__(
        self,
        fc_feat_dim_list: List[int],
        name: str,
        non_linearity: nn.Module = nn.ReLU,
        non_linearity_last: bool = False,
    ):
        super(MLP, self).__init__()
        self.fc_layers = nn.Sequential()
        for i in range(len(fc_feat_dim_list) - 1):
            fc_layer = make_fc_layer(fc_feat_dim_list[i], fc_feat_dim_list[i + 1])
            self.fc_layers.add_module("{0}_fc{1}".format(name, i + 1), fc_layer)
            if i + 1 < len(fc_feat_dim_list) - 1 or non_linearity_last:
                self.fc_layers.add_module("{0}_non_linear{1}".format(name, i + 1), non_linearity())

    def forward(self, data):
        return self.fc_layers(data)
