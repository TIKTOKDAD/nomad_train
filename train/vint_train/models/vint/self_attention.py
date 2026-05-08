import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """
    位置编码模块（Sinusoidal Positional Encoding）

    用固定的正弦/余弦函数为序列中每个位置生成一个位置向量，
    再与输入特征逐元素相加，让Transformer感知“顺序信息”。
    """

    def __init__(self, d_model, max_seq_len=6):
        super().__init__()

        # 一次性预计算 [max_seq_len, d_model] 的位置编码矩阵
        # 后续前向传播直接切片使用，避免每次重复计算。
        pos_enc = torch.zeros(max_seq_len, d_model)

        # pos: [max_seq_len, 1]，表示每个时间步/位置索引
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # div_term 控制不同通道的频率尺度：
        # 通道索引越大，频率越低（波长越长）
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 偶数维使用 sin，奇数维使用 cos
        # 这是Transformer经典做法，保证不同位置在嵌入空间中可区分且平滑。
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)

        # 扩展 batch 维，得到 [1, max_seq_len, d_model]
        # 前向时可与 [B, L, D] 自动广播相加。
        pos_enc = pos_enc.unsqueeze(0)

        # 注册为 buffer（非可学习参数）：
        # 1) 会随 model.to(device) 自动迁移设备
        # 2) 会随 state_dict 保存/加载
        # 3) 不参与梯度更新
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        # 只截取当前实际序列长度 x.size(1) 对应的位置编码。
        x = x + self.pos_enc[:, :x.size(1), :]
        return x


class MultiLayerDecoder(nn.Module):
    """
    基于自注意力编码器的多层解码头。

    流程：
    1. 输入序列加位置编码
    2. 经过多层 TransformerEncoder 提取全局时序特征
    3. 展平为向量后，经过多层全连接映射到目标维度
    """

    def __init__(self, embed_dim=512, seq_len=6, output_layers=[256, 128, 64], nhead=8, num_layers=8, ff_dim_factor=4):
        super(MultiLayerDecoder, self).__init__()

        # 位置编码：让注意力层知道 token 的时序位置
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len=seq_len)

        # 单层 TransformerEncoderLayer
        # - d_model: 每个 token 的特征维度
        # - nhead: 多头注意力头数
        # - dim_feedforward: 前馈网络隐藏层维度
        # - batch_first=True: 输入形状使用 [B, L, D]
        # - norm_first=True: Pre-LN 结构，训练更稳定
        self.sa_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=ff_dim_factor * embed_dim,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        # 堆叠 num_layers 层 encoder layer，形成自注意力特征提取器
        self.sa_decoder = nn.TransformerEncoder(self.sa_layer, num_layers=num_layers)

        # 输出 MLP：
        # 先把 [B, L, D] 展平为 [B, L*D]，再映射回 embed_dim，
        # 再逐层映射到 output_layers 指定的维度序列。
        self.output_layers = nn.ModuleList([nn.Linear(seq_len*embed_dim, embed_dim)])
        self.output_layers.append(nn.Linear(embed_dim, output_layers[0]))
        for i in range(len(output_layers) - 1):
            self.output_layers.append(nn.Linear(output_layers[i], output_layers[i + 1]))

    def forward(self, x):
        # x: [batch_size, seq_len, embed_dim]
        if self.positional_encoding:
            x = self.positional_encoding(x)

        # 自注意力编码后仍是 [B, L, D]
        x = self.sa_decoder(x)

        # 展平时序维和通道维，供后续全连接层使用
        x = x.reshape(x.shape[0], -1)

        # 逐层线性 + ReLU
        # 注意：当前实现最后一层后也会接 ReLU，这是原始行为，保持不变。
        for i in range(len(self.output_layers)):
            x = self.output_layers[i](x)
            x = F.relu(x)
        return x
