# ============================================================
# NoMaD模型定义
# ============================================================
# NoMaD (Navigation with Masked Diffusion) 是一个基于扩散模型的视觉导航系统
# 主要特点：
# 1. 使用扩散模型生成机器人动作，支持多模态预测
# 2. 支持目标mask机制，实现探索和导航两种模式
# 3. 包含视觉编码器、噪声预测网络和距离预测网络三个核心组件

import os
import argparse
import time
import pdb

import torch
import torch.nn as nn


class NoMaD(nn.Module):
    """
    NoMaD (Navigation with Masked Diffusion) 模型
    
    架构组成:
        1. vision_encoder: 视觉编码器
           - 输入：观测图像、目标图像、目标mask
           - 输出：视觉特征嵌入
           - 功能：编码观测和目标的视觉信息，支持目标mask
        
        2. noise_pred_net: 噪声预测网络（扩散模型核心）
           - 输入：带噪声的动作样本、时间步、全局条件（视觉特征）
           - 输出：预测的噪声
           - 功能：在扩散去噪过程中预测并去除噪声，生成干净的动作
        
        3. dist_pred_net: 距离预测网络
           - 输入：观测-目标条件特征
           - 输出：预测的距离
           - 功能：预测当前位置到目标的距离
    
    工作模式:
        - 探索模式（Unconditional）：目标被mask，模型生成探索性动作
        - 导航模式（Goal-Conditioned）：使用目标图像，模型生成导向目标的动作
    
    设计理念:
        - 使用扩散模型的多模态特性处理导航中的不确定性
        - 目标mask机制使单一模型同时支持探索和导航
        - 模块化设计便于替换不同的编码器和预测网络
    """

    def __init__(self, vision_encoder, 
                       noise_pred_net,
                       dist_pred_net):
        """
        初始化NoMaD模型
        
        参数:
            vision_encoder: 视觉编码器，将图像编码为特征向量
            noise_pred_net: 噪声预测网络，扩散模型的核心组件
            dist_pred_net: 距离预测网络，预测到目标的距离
        """
        super(NoMaD, self).__init__()

        # 三个核心组件
        self.vision_encoder = vision_encoder      # 视觉编码器
        self.noise_pred_net = noise_pred_net      # 噪声预测网络（扩散模型）
        self.dist_pred_net = dist_pred_net        # 距离预测网络
    
    def forward(self, func_name, **kwargs):
        """
        前向传播函数，使用函数名调度不同的子模块
        
        参数:
            func_name: 要调用的子模块名称
                - "vision_encoder": 编码视觉特征
                - "noise_pred_net": 预测噪声（扩散去噪）
                - "dist_pred_net": 预测距离
            **kwargs: 传递给子模块的参数
        
        返回:
            子模块的输出
        
        设计说明:
            使用函数名调度而非直接调用子模块，是为了：
            1. 在训练和推理时灵活调用不同组件
            2. 支持部分模块的独立使用（如只用视觉编码器）
            3. 便于在训练循环中分别处理不同的损失
        """
        if func_name == "vision_encoder":
            # 编码视觉特征
            # 输入: obs_img（观测图像）, goal_img（目标图像）, input_goal_mask（目标mask）
            # 输出: 视觉特征嵌入
            output = self.vision_encoder(kwargs["obs_img"], kwargs["goal_img"], input_goal_mask=kwargs["input_goal_mask"])
        elif func_name == "noise_pred_net":
            # 预测噪声（扩散模型的去噪步骤）
            # 输入: sample（带噪声的动作）, timestep（扩散时间步）, global_cond（全局条件/视觉特征）
            # 输出: 预测的噪声
            output = self.noise_pred_net(sample=kwargs["sample"], timestep=kwargs["timestep"], global_cond=kwargs["global_cond"])
        elif func_name == "dist_pred_net":
            # 预测距离
            # 输入: obsgoal_cond（观测-目标条件特征）
            # 输出: 预测的距离值
            output = self.dist_pred_net(kwargs["obsgoal_cond"])
        else:
            raise NotImplementedError
        return output


class DenseNetwork(nn.Module):
    """
    密集连接网络（全连接网络）
    
    用途:
        作为NoMaD的距离预测网络（dist_pred_net）
        将视觉特征嵌入映射为标量距离值
    
    架构:
        输入维度 -> 1/4维度 -> 1/16维度 -> 1（标量输出）
        使用ReLU激活函数
    
    设计说明:
        - 逐步降维的设计有助于提取关键特征
        - 简单的MLP结构足以完成距离回归任务
        - 输出单个标量表示到目标的距离
    """
    
    def __init__(self, embedding_dim):
        """
        初始化密集网络
        
        参数:
            embedding_dim: 输入特征嵌入的维度（来自视觉编码器）
        """
        super(DenseNetwork, self).__init__()
        
        self.embedding_dim = embedding_dim 
        # 构建三层全连接网络，逐步降维
        self.network = nn.Sequential(
            # 第一层：embedding_dim -> embedding_dim//4
            nn.Linear(self.embedding_dim, self.embedding_dim//4),
            nn.ReLU(),
            # 第二层：embedding_dim//4 -> embedding_dim//16
            nn.Linear(self.embedding_dim//4, self.embedding_dim//16),
            nn.ReLU(),
            # 第三层：embedding_dim//16 -> 1（标量输出）
            nn.Linear(self.embedding_dim//16, 1)
        )
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x: 输入特征嵌入 [batch_size, embedding_dim] 或 [batch_size, seq_len, embedding_dim]
        
        返回:
            距离预测值 [batch_size, 1]
        
        说明:
            如果输入是序列特征，会先展平为二维张量
        """
        # 重塑输入为 [batch_size, embedding_dim]
        x = x.reshape((-1, self.embedding_dim))
        # 通过网络得到距离预测
        output = self.network(x)
        return output



