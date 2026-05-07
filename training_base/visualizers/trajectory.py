import os

import matplotlib.pyplot as plt
import numpy as np
import torch


def _to_numpy(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def _xy(action):
    action = _to_numpy(action)
    return action[:, :2]


def _image_array(image):
    image = _to_numpy(image)
    if image is None:
        return None
    if image.ndim == 4:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in {1, 3, 4}:
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    return np.clip(image, 0.0, 1.0)


def _metric_scale(metric_scale):
    value = _to_numpy(metric_scale)
    if value is None:
        return 1.0
    return float(np.asarray(value).reshape(-1)[0])


def _maybe_scale(action, normalized: bool, metric_scale):
    action = _to_numpy(action).copy()
    if normalized:
        action[:, :2] *= _metric_scale(metric_scale)
    return action


def _plot_trajectory(ax, action, *, label, color, alpha=1.0, marker="o"):
    xy = action[:, :2]
    ax.plot(xy[:, 0], xy[:, 1], marker=marker, color=color, alpha=alpha, label=label)
    if action.shape[1] > 2:
        if action.shape[1] >= 4:
            direction = action[:, 2:4]
            norm = np.linalg.norm(direction, axis=1, keepdims=True)
            norm[norm == 0] = 1.0
            direction = direction / norm
        else:
            direction = np.stack([np.cos(action[:, 2]), np.sin(action[:, 2])], axis=-1)
        ax.quiver(xy[:, 0], xy[:, 1], direction[:, 0], direction[:, 1], color=color, alpha=alpha, scale=12.0)


def save_navigation_plot(
    path: str,
    *,
    label,
    pred=None,
    pred_samples=None,
    obs_image=None,
    goal_image=None,
    goal_pos=None,
    dist_pred=None,
    dist_label=None,
    normalized: bool = False,
    dataset_index=None,
    metric_scale=None,
    title: str = "",
) -> str:
    """Save an observation/goal plus trajectory comparison image."""
    del dataset_index
    os.makedirs(os.path.dirname(path), exist_ok=True)
    label = _maybe_scale(label, normalized, metric_scale)
    pred = None if pred is None else _maybe_scale(pred, normalized, metric_scale)
    if pred_samples is not None:
        pred_samples = [_maybe_scale(sample, normalized, metric_scale) for sample in _to_numpy(pred_samples)]
    goal_pos = _to_numpy(goal_pos)
    if goal_pos is not None:
        goal_pos = goal_pos.reshape(-1)[:2].copy()
        if normalized:
            goal_pos *= _metric_scale(metric_scale)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    if title:
        fig.suptitle(title)

    ax = axes[0]
    if pred_samples is not None:
        for idx, sample in enumerate(pred_samples):
            _plot_trajectory(
                ax,
                sample,
                label="sample" if idx == 0 else None,
                color="tab:cyan",
                alpha=0.25,
                marker=".",
            )
    if pred is not None:
        _plot_trajectory(ax, pred, label="prediction", color="tab:blue", marker="x")
    _plot_trajectory(ax, label, label="label", color="magenta", marker="o")
    ax.scatter([0], [0], c="black", s=25, label="robot")
    if goal_pos is not None:
        ax.scatter([goal_pos[0]], [goal_pos[1]], c="tab:red", s=35, label="goal")
    ax.set_title("Trajectory")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    obs = _image_array(obs_image)
    if obs is not None:
        axes[1].imshow(obs)
    axes[1].set_title("Observation")
    axes[1].axis("off")

    goal = _image_array(goal_image)
    if goal is not None:
        axes[2].imshow(goal)
    distance_text = []
    if dist_pred is not None:
        distance_text.append(f"pred={float(np.asarray(_to_numpy(dist_pred)).reshape(-1)[0]):.3f}")
    if dist_label is not None:
        distance_text.append(f"label={float(np.asarray(_to_numpy(dist_label)).reshape(-1)[0]):.3f}")
    axes[2].set_title("Goal" + (f" ({', '.join(distance_text)})" if distance_text else ""))
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def save_trajectory_plot(path: str, *, label: torch.Tensor, pred: torch.Tensor = None, title: str = "") -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4))
    label_xy = _xy(label)
    ax.plot(label_xy[:, 0], label_xy[:, 1], "o-", label="label")
    if pred is not None:
        pred_xy = _xy(pred)
        ax.plot(pred_xy[:, 0], pred_xy[:, 1], "x-", label="pred")
    ax.scatter([0], [0], c="black", s=20)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
