#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import torch
import numpy as np
import os
import time
from agent_ppo.conf.conf import Config


class Algorithm:
    def __init__(self, model, optimizer, scheduler, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.parameters = [p for param_group in self.optimizer.param_groups for p in param_group["params"]]
        self.train_step = 0

        self.logger = logger
        self.monitor = monitor

        self.cut_points = [value[0] for value in Config.data_shapes]
        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.seri_vec_split_shape = Config.SERI_VEC_SPLIT_SHAPE
        self.lstm_unit_size = Config.LSTM_UNIT_SIZE

        self.last_report_monitor_time = 0
        

    def _to_float(self, x):
        if isinstance(x, torch.Tensor):
            return float(x.detach().cpu().item())
        return float(x)

    def _get_entropy_beta(self):
        if not getattr(Config, "ENTROPY_BETA_DECAY_ENABLE", False):
            return float(Config.BETA_START)

        decay_steps = max(int(getattr(Config, "BETA_DECAY_STEPS", Config.TARGET_STEP)), 1)
        beta_start = float(Config.BETA_START)
        beta_end = float(getattr(Config, "BETA_END", 0.0))

        progress = min(float(self.train_step) / float(decay_steps), 1.0)
        beta = beta_start + progress * (beta_end - beta_start)
        return beta

    def learn(self, list_sample_data):
        """
        list_sample_data: list[SampleData]
        SampleData对象列表
        """
        # Extract sample field from SampleData objects and stack into tensor
        # 从 SampleData 对象中提取 sample 字段并 stack 成 tensor
        _input_datas = torch.stack([sample.sample for sample in list_sample_data]).to(self.device)
        results = {}

        data_list = list(_input_datas.split(self.cut_points, dim=1))
        for i, data in enumerate(data_list):
            data = data.reshape(-1)
            data_list[i] = data.float()

        seri_vec = data_list[0].reshape(-1, self.data_split_shape[0])
        feature, legal_action = seri_vec.split(
            [
                np.prod(self.seri_vec_split_shape[0]),
                np.prod(self.seri_vec_split_shape[1]),
            ],
            dim=1,
        )
        init_lstm_cell = data_list[-2]
        init_lstm_hidden = data_list[-1]

        feature_vec = feature.reshape(-1, self.seri_vec_split_shape[0][0])
        lstm_hidden_state = init_lstm_hidden.reshape(-1, self.lstm_unit_size)
        lstm_cell_state = init_lstm_cell.reshape(-1, self.lstm_unit_size)

        format_inputs = [feature_vec, lstm_hidden_state, lstm_cell_state]

        self.model.set_train_mode()
        self.optimizer.zero_grad()
        self.model.var_beta = self._get_entropy_beta()

        rst_list = self.model(format_inputs)
        total_loss, info_list = self.model.compute_loss(data_list, rst_list)
        results["total_loss"] = total_loss.item()

        total_loss.backward()

        # grad norm / grad clip
        # clip_grad_norm_ 返回的是裁剪前的 global grad norm
        if Config.USE_GRAD_CLIP:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)
            grad_norm_value = self._to_float(grad_norm)
            gradclip_frac = 1.0 if grad_norm_value > Config.GRAD_CLIP_RANGE else 0.0
        else:
            total_norm_sq = 0.0
            for p in self.parameters:
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2).item()
                    total_norm_sq += param_norm ** 2
            grad_norm_value = total_norm_sq ** 0.5
            gradclip_frac = 0.0

        lr = self.optimizer.param_groups[0]["lr"]

        self.optimizer.step()
        self.train_step += 1

        # update the learning rate
        # 更新学习率
        self.scheduler.step(self.train_step)

        _info_list = []
        for info in info_list:
            if isinstance(info, list):
                _info = [i.item() for i in info]
            else:
                _info = info.item()
            _info_list.append(_info)

        now = time.time()
        if now - self.last_report_monitor_time >= 60:
            _, (value_loss, policy_loss, entropy_loss) = _info_list

            results["total_loss"] = round(total_loss.item(), 6)
            results["value_loss"] = round(value_loss, 6)
            results["policy_loss"] = round(policy_loss, 6)
            results["entropy_loss"] = round(entropy_loss, 6)
            results["grad_norm"] = round(grad_norm_value, 6)
            results["gradclip_frac"] = round(gradclip_frac, 6)
            results["lr"] = round(lr*1e4, 6)
            results["train_steps"] = round(self.train_step, 6)
            results["entropy_beta"] = round(float(self.model.var_beta), 6)
            

            extra_metrics = getattr(self.model, "extra_metrics", {})
            for metric_name, metric_value in extra_metrics.items():
                results[metric_name] = round(self._to_float(metric_value), 6)

            if self.monitor:
                self.monitor.put_data({os.getpid(): results})

            self.last_report_monitor_time = now
