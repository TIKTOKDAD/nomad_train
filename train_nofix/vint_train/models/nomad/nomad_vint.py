# ============================================================
# NoMaD_ViNT 视觉编码器
# ============================================================
# NoMaD_ViNT是NoMaD模型的视觉编码器实现
# 主要特点：
# 1. 使用EfficientNet作为特征提取器
# 2. 支持时序上下文（多帧观测）
# 3. 使用Transformer自注意力机制融合时序信息
# 4. 支持目标mask机制，实现探索和导航模式切换
# 5. 使用GroupNorm替代BatchNorm提高训练稳定性

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from typing import List, Dict, Optional, Tuple, Callable
from efficientnet_pytorch import EfficientNet
from vint_train.models.vint.self_attention import PositionalEncoding

class NoMaD_ViNT(nn.Module):
    """
    NoMaD_ViNT 视觉编码器
    
    架构组成:
        1. 观测编码器（obs_encoder）：
           - 使用EfficientNet编码每一帧观测图像
           - 处理时序上下文（context_size帧）
        
        2. 目标编码器（goal_encoder）：
           - 使用EfficientNet编码观测-目标图像对
           - 输入为最新观测帧和目标图像的拼接
        
        3. 特征压缩层：
           - 将编码器输出压缩到统一维度
        
        4. 位置编码（positional_encoding）：
           - 为时序特征添加位置信息
        
        5. Transformer自注意力层（sa_encoder）：
           - 融合时序上下文信息
           - 支持目标mask机制
    
    工作流程:
        观测图像 -> 观测编码器 -> 观测特征序列
        (观测+目标) -> 目标编码器 -> 目标特征
        拼接 -> 位置编码 -> Transformer -> 平均池化 -> 输出特征
    
    目标Mask机制:
        - no_mask (0): 使用目标图像（导航模式）
        - goal_mask (1): 屏蔽目标图像（探索模式）
        - 通过src_key_padding_mask控制Transformer的注意力
    """
    
    def __init__(
        self,
        context_size: int = 5,
        obs_encoder: Optional[str] = "efficientnet-b0",
        obs_encoding_size: Optional[int] = 512,
        mha_num_attention_heads: Optional[int] = 2,
        mha_num_attention_layers: Optional[int] = 2,
        mha_ff_dim_factor: Optional[int] = 4,
    ) -> None:
        """
        初始化NoMaD_ViNT编码器
        
        参数:
            context_size: 时序上下文大小（历史观测帧数，默认5）
            obs_encoder: 观测编码器类型（默认"efficientnet-b0"）
            obs_encoding_size: 编码特征维度（默认512）
            mha_num_attention_heads: 多头注意力的头数（默认2）
            mha_num_attention_layers: Transformer层数（默认2）
            mha_ff_dim_factor: 前馈网络维度因子（默认4，即FFN维度=4*obs_encoding_size）
        
        设计说明:
            - context_size控制使用多少历史帧，更多帧提供更丰富的时序信息
            - 使用两个独立的EfficientNet分别编码观测和目标
            - Transformer用于融合时序信息，捕捉动态变化
            - 目标mask通过注意力掩码实现，无需修改网络结构
        """
        super().__init__()
        self.obs_encoding_size = obs_encoding_size
        self.goal_encoding_size = obs_encoding_size  # 目标编码维度与观测相同
        self.context_size = context_size

        # ========== 初始化观测编码器 ==========
        if obs_encoder.split("-")[0] == "efficientnet":
            # 使用EfficientNet作为观测编码器
            self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=3)  # 输入3通道RGB图像
            # 将BatchNorm替换为GroupNorm，提高小batch训练的稳定性
            self.obs_encoder = replace_bn_with_gn(self.obs_encoder)
            # 获取EfficientNet最后全连接层的输入特征数
            self.num_obs_features = self.obs_encoder._fc.in_features
            self.obs_encoder_type = "efficientnet"
        else:
            raise NotImplementedError
        
        # ========== 初始化目标编码器 ==========
        # 目标编码器输入6通道（3通道观测 + 3通道目标）
        self.goal_encoder = EfficientNet.from_name("efficientnet-b0", in_channels=6)
        self.goal_encoder = replace_bn_with_gn(self.goal_encoder)
        self.num_goal_features = self.goal_encoder._fc.in_features

        # ========== 初始化特征压缩层 ==========
        # 如果编码器输出维度与目标维度不同，使用线性层压缩
        if self.num_obs_features != self.obs_encoding_size:
            self.compress_obs_enc = nn.Linear(self.num_obs_features, self.obs_encoding_size)
        else:
            self.compress_obs_enc = nn.Identity()  # 维度相同则不压缩
        
        if self.num_goal_features != self.goal_encoding_size:
            self.compress_goal_enc = nn.Linear(self.num_goal_features, self.goal_encoding_size)
        else:
            self.compress_goal_enc = nn.Identity()

        # ========== 初始化位置编码和自注意力层 ==========
        # 位置编码：为序列中的每个位置添加位置信息
        # 序列长度 = context_size + 1（当前帧）+ 1（目标）
        self.positional_encoding = PositionalEncoding(self.obs_encoding_size, max_seq_len=self.context_size + 2)
        
        # Transformer编码器层
        self.sa_layer = nn.TransformerEncoderLayer(
            d_model=self.obs_encoding_size,  # 特征维度
            nhead=mha_num_attention_heads,   # 注意力头数
            dim_feedforward=mha_ff_dim_factor*self.obs_encoding_size,  # 前馈网络维度
            activation="gelu",  # 使用GELU激活函数
            batch_first=True,   # batch维度在第一维
            norm_first=True     # 在注意力和FFN之前进行归一化（Pre-LN）
        )
        # 堆叠多层Transformer
        self.sa_encoder = nn.TransformerEncoder(self.sa_layer, num_layers=mha_num_attention_layers)

        # ========== 定义目标mask（约定：0=不mask，1=mask）==========
        # 注册为 buffer 后会跟随 model.to(device)，避免每次 forward 里重复 .to(device)。
        goal_mask = torch.zeros((1, self.context_size + 2), dtype=torch.bool)
        goal_mask[:, -1] = True
        no_mask = torch.zeros((1, self.context_size + 2), dtype=torch.bool)
        all_masks = torch.cat([no_mask, goal_mask], dim=0)
        avg_pool_mask = torch.cat([
            1 - no_mask.float(),
            (1 - goal_mask.float()) * ((self.context_size + 2)/(self.context_size + 1))
        ], dim=0)
        self.register_buffer("goal_mask", goal_mask, persistent=False)
        self.register_buffer("no_mask", no_mask, persistent=False)
        self.register_buffer("all_masks", all_masks, persistent=False)
        self.register_buffer("avg_pool_mask", avg_pool_mask, persistent=False)


    def forward(self, obs_img: torch.tensor, goal_img: torch.tensor, input_goal_mask: torch.tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：编码观测和目标图像为特征向量
        
        参数:
            obs_img: 观测图像 [batch_size, 3*(context_size+1), H, W]
                    包含context_size个历史帧和1个当前帧，每帧3通道
            goal_img: 目标图像 [batch_size, 3, H, W]
            input_goal_mask: 目标mask [batch_size]
                           0 = 使用目标（导航模式）
                           1 = 屏蔽目标（探索模式）
        
        返回:
            obs_encoding_tokens: 融合后的特征向量 [batch_size, obs_encoding_size]
        
        工作流程:
            1. 编码目标：拼接最新观测帧和目标图像，通过目标编码器
            2. 编码观测：分别编码每一帧观测图像
            3. 拼接特征：将观测特征序列和目标特征拼接
            4. 应用mask：根据input_goal_mask决定是否屏蔽目标特征
            5. 位置编码：添加位置信息
            6. Transformer：通过自注意力融合时序信息
            7. 平均池化：得到最终的特征向量
        """
        device = obs_img.device

        # ========== 初始化目标编码 ==========
        goal_encoding = torch.zeros((obs_img.size()[0], 1, self.goal_encoding_size), device=device)
        
        # 获取输入的目标mask
        if input_goal_mask is not None:
            goal_mask = input_goal_mask.to(device)

        # ========== 编码目标 ==========
        # 拼接最新观测帧和目标图像：[batch, 6, H, W]
        # obs_img[:, 3*self.context_size:, :, :] 提取最新的观测帧（3通道）
        obsgoal_img = torch.cat([obs_img[:, 3*self.context_size:, :, :], goal_img], dim=1)
        
        # 通过目标编码器提取特征
        obsgoal_encoding = self.goal_encoder.extract_features(obsgoal_img)
        # 全局平均池化
        obsgoal_encoding = self.goal_encoder._avg_pooling(obsgoal_encoding)
        
        # 如果编码器包含顶层（分类层），则展平并应用dropout
        if self.goal_encoder._global_params.include_top:
            obsgoal_encoding = obsgoal_encoding.flatten(start_dim=1)
            obsgoal_encoding = self.goal_encoder._dropout(obsgoal_encoding)
        
        # 压缩到目标维度
        obsgoal_encoding = self.compress_goal_enc(obsgoal_encoding)

        # 确保维度正确：[batch_size, 1, goal_encoding_size]
        if len(obsgoal_encoding.shape) == 2:
            obsgoal_encoding = obsgoal_encoding.unsqueeze(1)
        assert obsgoal_encoding.shape[2] == self.goal_encoding_size
        goal_encoding = obsgoal_encoding
        
        # ========== 编码观测序列 ==========
        # 数学等价于 split(obs, 3, dim=1) 后按帧 concat，但避免 Python list 和额外 cat 开销。
        batch_size, _, height, width = obs_img.shape
        num_obs_frames = self.context_size + 1
        obs_img = (
            obs_img.reshape(batch_size, num_obs_frames, 3, height, width)
            .permute(1, 0, 2, 3, 4)
            .reshape(batch_size * num_obs_frames, 3, height, width)
        )

        # 通过观测编码器提取特征
        obs_encoding = self.obs_encoder.extract_features(obs_img)
        # 全局平均池化
        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)
        
        # 如果编码器包含顶层，则展平并应用dropout
        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding)
        
        # 压缩到目标维度
        obs_encoding = self.compress_obs_enc(obs_encoding)
        obs_encoding = obs_encoding.unsqueeze(1)
        
        # 重塑为序列形式：[context_size+1, batch_size, obs_encoding_size]
        obs_encoding = obs_encoding.reshape((self.context_size+1, -1, self.obs_encoding_size))
        # 转置为batch-first：[batch_size, context_size+1, obs_encoding_size]
        obs_encoding = torch.transpose(obs_encoding, 0, 1)
        
        # 拼接观测序列和目标特征：[batch_size, context_size+2, obs_encoding_size]
        obs_encoding = torch.cat((obs_encoding, goal_encoding), dim=1)
        
        # ========== 应用目标mask ==========
        # 如果提供了goal_mask，根据mask值选择对应的padding mask
        if goal_mask is not None:
            no_goal_mask = goal_mask.long()
            # 从预定义的mask中选择：0->no_mask（导航），1->goal_mask（探索）
            src_key_padding_mask = torch.index_select(self.all_masks, 0, no_goal_mask)
        else:
            src_key_padding_mask = None
        
        # ========== 应用位置编码 ==========
        if self.positional_encoding:
            obs_encoding = self.positional_encoding(obs_encoding)

        # ========== Transformer自注意力 ==========
        # 通过Transformer融合时序信息，src_key_padding_mask控制哪些token被屏蔽
        obs_encoding_tokens = self.sa_encoder(obs_encoding, src_key_padding_mask=src_key_padding_mask)
        
        # ========== 平均池化 ==========
        # 如果使用了mask，需要调整平均池化的权重
        if src_key_padding_mask is not None:
            # 选择对应的池化mask
            avg_mask = torch.index_select(self.avg_pool_mask, 0, no_goal_mask).unsqueeze(-1)
            # 应用mask权重
            obs_encoding_tokens = obs_encoding_tokens * avg_mask
        
        # 对所有token取平均，得到最终的特征向量
        obs_encoding_tokens = torch.mean(obs_encoding_tokens, dim=1)

        return obs_encoding_tokens



# ========== GroupNorm工具函数 ==========

def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    将所有BatchNorm层替换为GroupNorm层
    
    参数:
        root_module: 要处理的根模块
        features_per_group: 每组的特征数（默认16）
    
    返回:
        替换后的模块
    
    为什么使用GroupNorm:
        - BatchNorm在小batch size时性能不稳定
        - GroupNorm不依赖batch统计，更适合小batch训练
        - 在视觉导航任务中，batch size通常较小
        - GroupNorm在迁移学习中表现更好
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),  # 查找所有BatchNorm2d层
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,  # 组数 = 特征数 / 每组特征数
            num_channels=x.num_features)  # 通道数
    )
    return root_module


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    递归替换所有满足条件的子模块
    
    参数:
        root_module: 要处理的根模块
        predicate: 判断函数，返回True表示该模块需要被替换
        func: 替换函数，接收旧模块返回新模块
    
    返回:
        替换后的根模块
    
    工作流程:
        1. 如果根模块本身满足条件，直接替换并返回
        2. 否则，遍历所有子模块
        3. 对满足条件的子模块应用替换函数
        4. 更新父模块中的引用
        5. 验证所有目标模块都已被替换
    
    设计说明:
        - 支持嵌套模块的递归替换
        - 处理nn.Sequential和普通模块两种情况
        - 使用断言确保替换完整性
    """
    # 如果根模块本身满足条件，直接替换
    if predicate(root_module):
        return func(root_module)

    # 查找所有需要替换的子模块
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    
    # 遍历并替换每个子模块
    for *parent, k in bn_list:
        # 获取父模块
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        
        # 获取源模块
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        
        # 应用替换函数得到目标模块
        tgt_module = func(src_module)
        
        # 更新父模块中的引用
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    
    # 验证所有模块都已被替换
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0, "并非所有目标模块都被成功替换"
    
    return root_module



    
