# ============================================================
# Trajectory visualizers - plot trajectories and image projections
# ============================================================
# 本文件负责把模型输出保存成可读图片：
# 1. 绘制局部坐标系下的 label/pred/sample 轨迹
# 2. 若数据集提供 camera_metrics，则把轨迹投影回观测图像
# 3. 同时展示观测图、目标图、距离预测和标签

import os

import matplotlib.pyplot as plt
import numpy as np
import torch


# 将输入统一转换为 numpy
def _to_numpy(value):
    # Matplotlib 只消费 CPU numpy；Tensor 先 detach，避免保留计算图
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


# 提取 (x, y) 轨迹
def _xy(action):
    action = _to_numpy(action)
    return action[:, :2]


# 将图像张量转换为可绘制数组
def _image_array(image):
    image = _to_numpy(image)
    if image is None:
        return None
    # 支持 [B,C,H,W] 或 [C,H,W] 输入；batch 维存在时取第一张
    if image.ndim == 4:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in {1, 3, 4}:
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]
    return np.clip(image, 0.0, 1.0)


# 读取 metric_scale 缩放系数
def _metric_scale(metric_scale):
    value = _to_numpy(metric_scale)
    if value is None:
        return 1.0
    return float(np.asarray(value).reshape(-1)[0])


# 根据是否归一化对轨迹进行缩放
def _maybe_scale(action, normalized: bool, metric_scale):
    action = _to_numpy(action).copy()
    if normalized:
        # 训练标签若已按 metric_waypoint_spacing 归一化，绘图前乘回米制尺度
        action[:, :2] *= _metric_scale(metric_scale)
    return action


# 将样本轨迹整理为列表形式
def _as_sample_list(samples, normalized: bool, metric_scale):
    samples = _to_numpy(samples)
    if samples is None:
        return []
    if samples.ndim == 2:
        samples = samples[None]
    return [_maybe_scale(sample, normalized, metric_scale) for sample in samples]


# 根据 dataset_index 获取对应元信息
def _dataset_metadata_for_index(dataset_metadata, dataset_index):
    if not isinstance(dataset_metadata, dict) or not dataset_metadata:
        return {}
    if "camera_metrics" in dataset_metadata:
        # 允许调用方直接传单个数据集 metadata
        return dataset_metadata
    value = _to_numpy(dataset_index)
    if value is None:
        return {}
    index = int(np.asarray(value).reshape(-1)[0])
    return dataset_metadata.get(str(index), dataset_metadata.get(index, {})) or {}


# 提取相机参数用于投影
def _camera_metrics(dataset_metadata, dataset_index):
    metadata = _dataset_metadata_for_index(dataset_metadata, dataset_index)
    metrics = metadata.get("camera_metrics", {})
    return metrics if isinstance(metrics, dict) else {}


# 读取图像尺寸
def _image_size(image):
    if image is None:
        return None
    if image.ndim == 2:
        return image.shape[1], image.shape[0]
    return image.shape[1], image.shape[0]


# 在坐标轴上绘制轨迹（可选朝向箭头）
def _plot_trajectory(ax, action, *, label, color, alpha=1.0, marker="o"):
    xy = action[:, :2]
    ax.plot(xy[:, 0], xy[:, 1], marker=marker, color=color, alpha=alpha, label=label)
    if action.shape[1] > 2:
        # 角度可能是 cos/sin 向量，也可能是 yaw 标量，按维度自动识别
        if action.shape[1] >= 4:
            direction = action[:, 2:4]
            norm = np.linalg.norm(direction, axis=1, keepdims=True)
            norm[norm == 0] = 1.0
            direction = direction / norm
        else:
            direction = np.stack([np.cos(action[:, 2]), np.sin(action[:, 2])], axis=-1)
        ax.quiver(xy[:, 0], xy[:, 1], direction[:, 0], direction[:, 1], color=color, alpha=alpha, scale=12.0)


# 构建相机内参矩阵
def gen_camera_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


# 将平面轨迹投影到图像坐标
def project_points(
    xy: np.ndarray,
    camera_height: float,
    camera_x_offset: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
):
    try:
        # OpenCV 只是可选依赖；缺失时跳过投影，不影响主训练
        import cv2
    except ImportError:
        return None

    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim == 2:
        xy = xy[None]
    batch_size, horizon, _ = xy.shape
    xyz = np.concatenate([xy, -camera_height * np.ones(list(xy.shape[:-1]) + [1])], axis=-1)
    # camera_x_offset 把机器人局部坐标中的点平移到相机坐标参考点
    xyz[..., 0] += camera_x_offset
    # OpenCV 相机坐标系与机器人局部坐标系轴定义不同，这里做轴重排/符号转换
    xyz_cv = np.stack([xyz[..., 1], -xyz[..., 2], xyz[..., 0]], axis=-1)
    uv, _ = cv2.projectPoints(
        xyz_cv.reshape(batch_size * horizon, 3),
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        camera_matrix,
        dist_coeffs,
    )
    return uv.reshape(batch_size, horizon, 2)


# 解析相机参数为投影所需数组
def _camera_arrays(camera_metrics):
    required = ("camera_height", "camera_matrix", "dist_coeffs")
    if not all(key in camera_metrics for key in required):
        return None
    # YAML 里的 camera_matrix/dist_coeffs 是展开字段，这里转成 OpenCV 所需数组
    matrix_config = camera_metrics["camera_matrix"]
    dist_config = camera_metrics["dist_coeffs"]
    camera_matrix = gen_camera_matrix(
        matrix_config["fx"],
        matrix_config["fy"],
        matrix_config["cx"],
        matrix_config["cy"],
    )
    dist_coeffs = np.array(
        [
            dist_config.get("k1", 0.0),
            dist_config.get("k2", 0.0),
            dist_config.get("p1", 0.0),
            dist_config.get("p2", 0.0),
            dist_config.get("k3", 0.0),
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )
    return (
        float(camera_metrics["camera_height"]),
        float(camera_metrics.get("camera_x_offset", 0.0)),
        camera_matrix,
        dist_coeffs,
    )


# 计算轨迹点在图像中的像素坐标
def get_pos_pixels(points: np.ndarray, camera_metrics: dict, image_size, clip: bool = False):
    arrays = _camera_arrays(camera_metrics)
    if arrays is None or image_size is None:
        return None
    width, height = image_size
    projected = project_points(np.asarray(points, dtype=np.float64), *arrays)
    if projected is None:
        return None
    pixels = projected[0]
    # 图像坐标 x 方向和投影坐标约定相反，按当前历史可视化逻辑做镜像
    pixels[:, 0] = width - pixels[:, 0]
    if clip:
        pixels = np.array(
            [
                [np.clip(p[0], 0, width), np.clip(p[1], 0, height)]
                for p in pixels
            ],
            dtype=np.float64,
        )
    else:
        pixels = np.array(
            [
                p
                for p in pixels
                if np.all(p > 0) and np.all(p < [width, height])
            ],
            dtype=np.float64,
        )
    return pixels


# 将轨迹/点叠加到观测图像上
def plot_trajs_and_points_on_image(
    ax,
    image,
    camera_metrics,
    trajectories,
    points,
):
    if image is not None:
        ax.imshow(image)
    ax.axis("off")
    if not camera_metrics:
        return False
    image_size = _image_size(image)
    if image_size is None:
        return False

    drew_projection = False
    for item in trajectories:
        # clip=False 会过滤画面外轨迹点，避免线条跨越整张图
        pixels = get_pos_pixels(item["trajectory"][:, :2], camera_metrics, image_size, clip=False)
        if pixels is not None and len(pixels) > 0:
            ax.plot(
                pixels[:250, 0],
                pixels[:250, 1],
                color=item["color"],
                alpha=item.get("alpha", 1.0),
                lw=item.get("linewidth", 2.0),
            )
            drew_projection = True

    for item in points:
        point = np.asarray(item["point"], dtype=np.float64).reshape(-1, 2)
        # 点使用 clip=True，让目标/机器人位置即使略越界也显示在图边缘
        pixels = get_pos_pixels(point[:, :2], camera_metrics, image_size, clip=True)
        if pixels is not None and len(pixels) > 0:
            ax.plot(
                pixels[:250, 0],
                pixels[:250, 1],
                color=item["color"],
                marker="o",
                markersize=item.get("markersize", 7.0),
                alpha=item.get("alpha", 1.0),
            )
            drew_projection = True

    return drew_projection


# 格式化距离预测与标签文本
def _distance_text(dist_pred, dist_label):
    entries = []
    if dist_pred is not None:
        values = np.asarray(_to_numpy(dist_pred), dtype=np.float64).reshape(-1)
        if values.size > 1:
            entries.append(f"pred={values.mean():.3f}+/-{values.std():.3f}")
        elif values.size == 1:
            entries.append(f"pred={values[0]:.3f}")
    if dist_label is not None:
        values = np.asarray(_to_numpy(dist_label), dtype=np.float64).reshape(-1)
        if values.size:
            entries.append(f"label={values[0]:.3f}")
    return ", ".join(entries)


# 保存导航可视化：轨迹 + 观测/目标图像
def save_navigation_plot(
    path: str,
    *,
    label,
    pred=None,
    pred_samples=None,
    sample_groups=None,
    obs_image=None,
    goal_image=None,
    goal_pos=None,
    dist_pred=None,
    dist_label=None,
    normalized: bool = False,
    dataset_index=None,
    metric_scale=None,
    dataset_metadata=None,
    title: str = "",
) -> str:
    """Save an observation/goal plus trajectory comparison image."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # label/pred/sample 都统一在绘图入口做 normalized -> metric 尺度转换
    label = _maybe_scale(label, normalized, metric_scale)
    pred = None if pred is None else _maybe_scale(pred, normalized, metric_scale)

    prepared_groups = []
    if pred_samples is not None:
        # 兼容旧接口 pred_samples，同时支持新接口 sample_groups
        prepared_groups.append(
            {
                "label": "sample",
                "samples": _as_sample_list(pred_samples, normalized, metric_scale),
                "color": "tab:cyan",
                "alpha": 0.25,
                "marker": ".",
            }
        )
    for group in sample_groups or []:
        prepared_groups.append(
            {
                "label": group.get("label", "sample"),
                "samples": _as_sample_list(group.get("samples"), normalized, metric_scale),
                "color": group.get("color", "tab:cyan"),
                "alpha": float(group.get("alpha", 0.25)),
                "marker": group.get("marker", "."),
            }
        )

    goal_pos = _to_numpy(goal_pos)
    if goal_pos is not None:
        goal_pos = goal_pos.reshape(-1)[:2].copy()
        if normalized:
            # goal_pos 和动作轨迹使用同一个 metric_scale
            goal_pos *= _metric_scale(metric_scale)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    if title:
        fig.suptitle(title)

    ax = axes[0]
    projection_trajs = []
    for group in prepared_groups:
        # 采样轨迹通常很多，使用低 alpha 叠加展示分布
        for idx, sample in enumerate(group["samples"]):
            label_name = group["label"] if idx == 0 else None
            _plot_trajectory(
                ax,
                sample,
                label=label_name,
                color=group["color"],
                alpha=group["alpha"],
                marker=group["marker"],
            )
            projection_trajs.append(
                {
                    "trajectory": sample,
                    "color": group["color"],
                    "alpha": group["alpha"],
                    "linewidth": 1.4,
                }
            )
    if pred is not None:
        # 主预测轨迹用更明显的蓝色 x 标记
        _plot_trajectory(ax, pred, label="prediction", color="tab:blue", marker="x")
        projection_trajs.append({"trajectory": pred, "color": "tab:blue", "alpha": 1.0, "linewidth": 2.0})
    _plot_trajectory(ax, label, label="label", color="magenta", marker="o")
    projection_trajs.append({"trajectory": label, "color": "magenta", "alpha": 1.0, "linewidth": 2.4})
    ax.scatter([0], [0], c="black", s=25, label="robot")
    projection_points = [{"point": np.array([0.0, 0.0]), "color": "black", "markersize": 6.0}]
    if goal_pos is not None:
        ax.scatter([goal_pos[0]], [goal_pos[1]], c="tab:red", s=35, label="goal")
        projection_points.append({"point": goal_pos, "color": "tab:red", "markersize": 7.0})
    ax.set_title("Trajectory")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    obs = _image_array(obs_image)
    camera_metrics = _camera_metrics(dataset_metadata, dataset_index)
    projection = plot_trajs_and_points_on_image(
        axes[1],
        obs,
        camera_metrics,
        projection_trajs,
        projection_points,
    )
    axes[1].set_title("Observation Projection" if projection else "Observation")

    goal = _image_array(goal_image)
    if goal is not None:
        axes[2].imshow(goal)
    distance_text = _distance_text(dist_pred, dist_label)
    axes[2].set_title("Goal" + (f" ({distance_text})" if distance_text else ""))
    axes[2].axis("off")

    fig.tight_layout()
    # bbox_inches="tight" 减少 W&B 图片周围空白
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# 保存轨迹对比图
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
