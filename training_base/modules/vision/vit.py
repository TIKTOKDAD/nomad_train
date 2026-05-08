# ============================================================
# ViT encoder - patch based goal-masked visual transformer
# ============================================================
# 本文件实现一个可选的 ViT 视觉编码器：
# 1. 将 context 多帧观测和 goal 图像沿宽度方向拼成一张长图
# 2. 用 patch embedding + Transformer 提取全局特征
# 3. MaskedGoalViT 可通过 input_goal_mask 屏蔽目标区域的注意力

from typing import Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange


# 视觉 Transformer 编码器（支持多帧观测 + 目标拼接）
class ViT(nn.Module):
    # 初始化 ViT 主干网络
    def __init__(
        self,
        obs_encoding_size: Optional[int] = 512,
        context_size: int = 5,
        image_size: int = 128,
        patch_size: int = 16,
        mha_num_attention_heads: Optional[int] = 4,
        mha_num_attention_layers: Optional[int] = 4,
    ) -> None:
        """
        ViT class
        """
        super(ViT, self).__init__()
        self.context_size = context_size
        self.patch_size = patch_size
        # image_size 支持 int 或 (width, height)，内部统一保存 height/width
        if type(image_size) == int:
            self.image_height = image_size
            self.image_width = image_size
        else:
            self.image_width = image_size[0]
            self.image_height = image_size[1]
        self.ViT = MaskedGoalViT(
            context_size=context_size,
            # 宽度乘以 context_size+2，因为多帧观测和目标图会横向拼接
            image_size=(self.image_height, self.image_width*(self.context_size + 2)),
            patch_size=self.patch_size,
            dim=obs_encoding_size,
            depth = mha_num_attention_layers,
            heads = mha_num_attention_heads,
            mlp_dim = obs_encoding_size
        )

    # 前向：拼接观测与目标后送入 ViT
    def forward(
        self, obs_img: torch.tensor, goal_img: torch.tensor, input_goal_mask: torch.tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        # obs_img: [B, 3*(context+1), H, W]，每 3 通道拆成一帧 RGB
        obs_img_list = list(torch.split(obs_img, 3, dim=1))
        # 横向拼接观测帧和目标图，得到 [B,3,H,W*(context+2)]
        obsgoal_img_list = obs_img_list + [goal_img]
        x = torch.cat(obsgoal_img_list, dim=-1)
        assert len(x.shape) == 4, "输入图像必须是 4D 张量"
        assert x.shape[1] == 3, "输入图像通道数必须为 3"
        assert x.shape[2] == self.image_height, f"输入图像高度必须为 {self.image_height}"
        assert x.shape[3] == self.image_width*(self.context_size + 2), f"输入图像宽度必须为 {self.image_width}*(context_size + 2)"
       
        final_repr = self.ViT(x)
        
        return final_repr

# Helper Functions for ViT

# 将单值转换为二维元组
def pair(t):
    # patch_size/image_size 可写 int，也可写 (h,w)
    return t if isinstance(t, tuple) else (t, t)

# 2D sin/cos 位置编码
def posemb_sincos_2d(patches, temperature = 10000, dtype = torch.float32):
    # patches 形状为 [B, H_patch, W_patch, dim]，位置编码只依赖网格尺寸
    _, h, w, dim, device, dtype = *patches.shape, patches.device, patches.dtype

    y, x = torch.meshgrid(torch.arange(h, device = device), torch.arange(w, device = device), indexing = 'ij')
    assert (dim % 4) == 0, '特征维度必须是 4 的倍数，才能生成 sincos 位置编码'
    omega = torch.arange(dim // 4, device = device) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim = 1)
    return pe.type(dtype)

# Classes

# 前馈网络块
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            # PreNorm FFN：先 LayerNorm，再 MLP 投影回原维度
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

# 多头注意力块（支持 mask）
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        # inner_dim 是所有 attention head 拼接后的维度
        inner_dim = dim_head *  heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, mask):
        x = self.norm(x)

        # q/k/v: [B, heads, num_tokens, dim_head]
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if len(mask.shape) == 3:
            mask = mask.unsqueeze(1)
        # mask 中 0 表示可见，-1e9 表示屏蔽，直接加到 attention logits
        attn = self.attend(dots + mask) 
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# Transformer 编码器堆叠
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            # 每层由一个自注意力块和一个 FFN 块组成
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
    def forward(self, x, mask):
        for attn, ff in self.layers:
            # 标准残差结构：Attention + residual，FFN + residual
            x = attn(x, mask) + x
            x = ff(x) + x
        return x


# Implementation of ViT with goal masking
class MaskedGoalViT(nn.Module):
    def __init__(self, *, context_size, image_size, patch_size, dim, depth, heads, mlp_dim, channels = 3, dim_head = 64):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, '图像尺寸必须能被 patch_size 整除。'

        # 计算 patch 网格大小和每个 patch 展平后的像素维度
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.h = image_height // patch_height
        self.w = image_width // patch_width
        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.to_latent = nn.Identity()

        self.goal_mask = torch.ones((self.h, self.w))
        assert self.w % (context_size + 2) == 0, "宽度方向 patch 数必须能被 context_size + 2 整除"
        # 最后一段宽度对应 goal 图像；goal_mask=0 表示这些 token 可被屏蔽
        self.goal_mask[:, -self.w//(context_size + 2):] = 0
        self.goal_mask = rearrange(self.goal_mask, 'h w -> (h w)')
        self.no_mask = torch.ones(self.h*self.w)
        self.all_masks = torch.stack([self.no_mask, self.goal_mask,], dim=0)
        self.no_cross_mask = torch.ones((self.h*self.w, self.h*self.w))
        self.goal_cross_mask = torch.ones((self.h*self.w, self.h*self.w))
        for i in range(self.h*self.w):
            for j in range(self.h*self.w):
                # 只要 attention 两端任意一个 token 属于 goal 区域，就在 goal_mask 模式下屏蔽
                if self.goal_mask[i] + self.goal_mask[j] < 2:
                    self.goal_cross_mask[i, j] = 0
        self.all_cross_masks = torch.stack([self.no_cross_mask, self.goal_cross_mask], dim=0)
        self.mean_mask = self.all_masks / self.all_masks.mean(dim=1, keepdim=True)

        self.all_cross_masks = torch.where(self.all_cross_masks == 0, -1e9, 0.0)
        self.all_masks = torch.where(self.all_masks == 0, -1e9, 0.0)


    # 前向：根据目标 mask 控制注意力范围
    def forward(self, img, input_goal_mask=None):
        b, c, h, w, dtype = *img.shape, img.dtype
        device = img.device

        if input_goal_mask is None:
            # 默认不屏蔽目标区域
            input_goal_mask = torch.zeros(b, dtype=torch.int64)

        # 为 batch 中每个样本选择 no_cross_mask 或 goal_cross_mask
        final_mask = torch.index_select(self.all_cross_masks.to(device), 0, input_goal_mask.to(device))

        x = self.to_patch_embedding(img)
        pe = posemb_sincos_2d(x)
        # 展平 patch 网格，并加上固定 sin/cos 位置编码
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = self.transformer(x, mask=final_mask)
        # mean_mask 重新归一化平均池化，防止屏蔽目标后特征幅值变小
        final_mask = torch.index_select(self.mean_mask.to(device), 0, input_goal_mask.to(device)).unsqueeze(-1)
        x = x * final_mask 
        x = x.mean(dim = 1)

        x = self.to_latent(x)
        return x
