# ============================================================
# Image grid visualizer helpers - observation/goal quick plots
# ============================================================
# 本文件提供最基础的图像可视化工具，适合调试单对 obs/goal 图像。

import os

import matplotlib.pyplot as plt
import torch


# 将张量图像转换为 numpy 格式，便于绘图
def tensor_image_to_numpy(image: torch.Tensor):
    # 输入通常是 [C,H,W]；Matplotlib 需要 [H,W,C]
    image = image.detach().cpu()
    if image.dim() == 3:
        image = image.permute(1, 2, 0)
    image = image.numpy()
    image = image.clip(0.0, 1.0)
    return image


# 保存观测/目标图像对的并排可视化
def save_obs_goal_pair(path: str, obs_image: torch.Tensor, goal_image: torch.Tensor, title: str = "") -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # 左观测、右目标，便于快速检查图像预处理是否正确
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(tensor_image_to_numpy(obs_image))
    axes[0].set_title("obs")
    axes[0].axis("off")
    axes[1].imshow(tensor_image_to_numpy(goal_image))
    axes[1].set_title("goal")
    axes[1].axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
