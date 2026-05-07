import os

import matplotlib.pyplot as plt
import torch


def tensor_image_to_numpy(image: torch.Tensor):
    image = image.detach().cpu()
    if image.dim() == 3:
        image = image.permute(1, 2, 0)
    image = image.numpy()
    image = image.clip(0.0, 1.0)
    return image


def save_obs_goal_pair(path: str, obs_image: torch.Tensor, goal_image: torch.Tensor, title: str = "") -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
