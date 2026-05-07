# ============================================================
# 动作可视化工具模块
# ============================================================
# 本文件提供轨迹和路径点的可视化功能
# 主要功能：
# 1. 可视化预测轨迹与真实轨迹的对比
# 2. 在图像上叠加轨迹（使用相机内参投影）
# 3. 绘制带方向的轨迹（使用箭头表示朝向）
# 4. 支持多种可视化模式（纯轨迹图、图像叠加图）
# 5. 集成wandb日志记录

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import Optional, List
import wandb
import yaml
import torch
import torch.nn as nn
from vint_train.visualizing.visualize_utils import (
    to_numpy,
    numpy_to_img,
    VIZ_IMAGE_SIZE,
    RED,
    GREEN,
    BLUE,
    CYAN,
    YELLOW,
    MAGENTA,
)

# 加载数据配置文件（包含相机参数和数据集信息）
with open(os.path.join(os.path.dirname(__file__), "../data/data_config.yaml"), "r",encoding="utf-8") as f:
    data_config = yaml.safe_load(f)


def visualize_traj_pred(
    batch_obs_images: np.ndarray,
    batch_goal_images: np.ndarray,
    dataset_indices: np.ndarray,
    batch_goals: np.ndarray,
    batch_pred_waypoints: np.ndarray,
    batch_label_waypoints: np.ndarray,
    eval_type: str,
    normalized: bool,
    save_folder: str,
    epoch: int,
    num_images_preds: int = 8,
    use_wandb: bool = True,
    display: bool = False,
):
    """
    使用自我中心视角可视化预测轨迹与真实轨迹的对比
    
    参数:
        batch_obs_images: 观测图像批次 [batch_size, height, width, channels]
        batch_goal_images: 目标图像批次 [batch_size, height, width, channels]
        dataset_indices: 数据集索引，对应data_config中的数据集名称
        batch_goals: 目标位置批次 [batch_size, 2]
        batch_pred_waypoints: 预测路径点批次 [batch_size, horizon, 4] 或 [batch_size, horizon, 2]
                             或 [batch_size, num_trajs_sampled, horizon, {2 or 4}]
        batch_label_waypoints: 真实路径点批次 [batch_size, T, 4] 或 [batch_size, horizon, 2]
        eval_type: 评估类型，格式为"{data_type}_{eval_type}"（如"recon_train"、"gs_test"等）
        normalized: 路径点是否已归一化
        save_folder: 保存图像的文件夹路径，如果为None则不保存
        epoch: 当前epoch编号
        num_images_preds: 要可视化的图像数量（默认8）
        use_wandb: 是否使用wandb记录图像（默认True）
        display: 是否显示图像（默认False）
    
    功能:
        1. 为batch中的每个样本生成可视化
        2. 显示三个子图：轨迹对比图、观测图像叠加图、目标图像
        3. 如果路径点已归一化，则反归一化到真实尺度
        4. 保存图像到本地并上传到wandb
    
    设计说明:
        - 使用自我中心坐标系（机器人位于原点）
        - 支持多模态预测（多条预测轨迹）
        - 颜色编码：青色=预测，洋红色=真实，绿色=起点，红色=终点
    """
    visualize_path = None
    if save_folder is not None:
        # 创建保存路径：save_folder/visualize/eval_type/epoch{N}/action_prediction
        visualize_path = os.path.join(
            save_folder, "visualize", eval_type, f"epoch{epoch}", "action_prediction"
        )

    # 确保保存路径存在
    if not os.path.exists(visualize_path):
        os.makedirs(visualize_path)

    # 验证所有输入的batch大小一致
    assert (
        len(batch_obs_images)
        == len(batch_goal_images)
        == len(batch_goals)
        == len(batch_pred_waypoints)
        == len(batch_label_waypoints)
    )

    # 获取数据集名称列表（按字母顺序排序）
    dataset_names = list(data_config.keys())
    dataset_names.sort()

    batch_size = batch_obs_images.shape[0]
    wandb_list = []
    
    # 为每个样本生成可视化
    for i in range(min(batch_size, num_images_preds)):
        # 提取当前样本的数据
        obs_img = numpy_to_img(batch_obs_images[i])
        goal_img = numpy_to_img(batch_goal_images[i])
        dataset_name = dataset_names[int(dataset_indices[i])]
        goal_pos = batch_goals[i]
        pred_waypoints = batch_pred_waypoints[i]
        label_waypoints = batch_label_waypoints[i]

        # 如果路径点已归一化，反归一化到真实尺度（米）
        if normalized:
            # 使用数据集特定的路径点间距进行缩放
            pred_waypoints *= data_config[dataset_name]["metric_waypoint_spacing"]
            label_waypoints *= data_config[dataset_name]["metric_waypoint_spacing"]
            goal_pos *= data_config[dataset_name]["metric_waypoint_spacing"]

        save_path = None
        if visualize_path is not None:
            # 生成保存路径，使用零填充的索引（如0001.png）
            save_path = os.path.join(visualize_path, f"{str(i).zfill(4)}.png")

        # 生成对比可视化
        compare_waypoints_pred_to_label(
            obs_img,
            goal_img,
            dataset_name,
            goal_pos,
            pred_waypoints,
            label_waypoints,
            save_path,
            display,
        )
        
        # 添加到wandb列表
        if use_wandb:
            wandb_list.append(wandb.Image(save_path))
    
    # 上传到wandb
    if use_wandb:
        wandb.log({f"{eval_type}_action_prediction": wandb_list}, commit=False)


def compare_waypoints_pred_to_label(
    obs_img,
    goal_img,
    dataset_name: str,
    goal_pos: np.ndarray,
    pred_waypoints: np.ndarray,
    label_waypoints: np.ndarray,
    save_path: Optional[str] = None,
    display: Optional[bool] = False,
):
    """
    使用自我中心视角对比预测路径和真实路径
    
    参数:
        obs_img: 观测图像
        goal_img: 目标图像
        dataset_name: 数据集名称，在data_config.yaml中定义（如"recon"）
        goal_pos: 目标位置 [2]
        pred_waypoints: 预测路径点 [horizon, 2或4] 或 [num_samples, horizon, 2或4]
        label_waypoints: 真实路径点 [horizon, 2或4]
        save_path: 保存图像的路径，如果为None则不保存
        display: 是否显示图像
    
    功能:
        生成三个子图的可视化：
        1. 左图：纯轨迹对比图（俯视图）
        2. 中图：观测图像上叠加轨迹（使用相机投影）
        3. 右图：目标图像
    
    设计说明:
        - 使用自我中心坐标系，机器人位于原点(0,0)
        - 青色表示预测轨迹，洋红色表示真实轨迹
        - 绿色点表示起点（机器人位置），红色点表示终点（目标位置）
        - 支持多模态预测（多条预测轨迹）
    """
    # 创建1行3列的子图
    fig, ax = plt.subplots(1, 3)
    start_pos = np.array([0, 0])  # 机器人起始位置（自我中心坐标系原点）
    
    # 组织轨迹列表：如果有多条预测轨迹，展开它们；最后添加真实轨迹
    if len(pred_waypoints.shape) > 2:
        # 多模态预测：[num_samples, horizon, 2或4]
        trajs = [*pred_waypoints, label_waypoints]
    else:
        # 单一预测：[horizon, 2或4]
        trajs = [pred_waypoints, label_waypoints]
    
    # 左图：纯轨迹对比图（俯视图）
    plot_trajs_and_points(
        ax[0],
        trajs,
        [start_pos, goal_pos],
        traj_colors=[CYAN, MAGENTA],  # 青色=预测，洋红色=真实
        point_colors=[GREEN, RED],     # 绿色=起点，红色=终点
    )
    
    # 中图：在观测图像上叠加轨迹（使用相机投影）
    plot_trajs_and_points_on_image(
        ax[1],
        obs_img,
        dataset_name,
        trajs,
        [start_pos, goal_pos],
        traj_colors=[CYAN, MAGENTA],
        point_colors=[GREEN, RED],
    )
    
    # 右图：目标图像
    ax[2].imshow(goal_img)

    # 设置图像大小和标题
    fig.set_size_inches(18.5, 10.5)
    ax[0].set_title(f"Action Prediction")
    ax[1].set_title(f"Observation")
    ax[2].set_title(f"Goal")

    # 保存图像
    if save_path is not None:
        fig.savefig(
            save_path,
            bbox_inches="tight",
        )

    # 关闭图像（如果不显示）
    if not display:
        plt.close(fig)


def plot_trajs_and_points_on_image(
    ax: plt.Axes,
    img: np.ndarray,
    dataset_name: str,
    list_trajs: list,
    list_points: list,
    traj_colors: list = [CYAN, MAGENTA],
    point_colors: list = [RED, GREEN],
):
    """
    在图像上绘制轨迹和点（使用相机投影）
    
    参数:
        ax: matplotlib坐标轴
        img: 要绘制的图像
        dataset_name: 数据集名称，在data_config.yaml中定义（如"recon"）
        list_trajs: 轨迹列表，每条轨迹是形状为(horizon, 2)或(horizon, 4)的numpy数组
        list_points: 点列表，每个点是形状为(2,)的numpy数组
        traj_colors: 轨迹颜色列表
        point_colors: 点颜色列表
    
    功能:
        1. 在图像上显示观测图像
        2. 如果数据集配置了相机内参，将3D轨迹投影到2D图像平面
        3. 在图像上绘制投影后的轨迹和点
    
    设计说明:
        - 使用相机内参矩阵和畸变系数进行准确投影
        - 如果没有相机配置，图像将按原样显示（不绘制轨迹）
        - 投影考虑了相机高度和x轴偏移
        - 限制绘制前250个点以避免图像过于拥挤
    """
    assert len(list_trajs) <= len(traj_colors), "轨迹颜色数量不足"
    assert len(list_points) <= len(point_colors), "点颜色数量不足"
    assert (
        dataset_name in data_config
    ), f"数据集 {dataset_name} 未在 data/data_config.yaml 中找到"

    # 显示图像
    ax.imshow(img)
    
    # 检查是否配置了相机参数
    if (
        "camera_metrics" in data_config[dataset_name]
        and "camera_height" in data_config[dataset_name]["camera_metrics"]
        and "camera_matrix" in data_config[dataset_name]["camera_metrics"]
        and "dist_coeffs" in data_config[dataset_name]["camera_metrics"]
    ):
        # 提取相机参数
        camera_height = data_config[dataset_name]["camera_metrics"]["camera_height"]
        camera_x_offset = data_config[dataset_name]["camera_metrics"]["camera_x_offset"]

        # 提取相机内参矩阵参数
        fx = data_config[dataset_name]["camera_metrics"]["camera_matrix"]["fx"]
        fy = data_config[dataset_name]["camera_metrics"]["camera_matrix"]["fy"]
        cx = data_config[dataset_name]["camera_metrics"]["camera_matrix"]["cx"]
        cy = data_config[dataset_name]["camera_metrics"]["camera_matrix"]["cy"]
        camera_matrix = gen_camera_matrix(fx, fy, cx, cy)

        # 提取畸变系数
        k1 = data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["k1"]
        k2 = data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["k2"]
        p1 = data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["p1"]
        p2 = data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["p2"]
        k3 = data_config[dataset_name]["camera_metrics"]["dist_coeffs"]["k3"]
        dist_coeffs = np.array([k1, k2, p1, p2, k3, 0.0, 0.0, 0.0])

        # 绘制轨迹
        for i, traj in enumerate(list_trajs):
            xy_coords = traj[:, :2]  # 提取x,y坐标 (horizon, 2)
            # 将3D坐标投影到2D图像平面
            traj_pixels = get_pos_pixels(
                xy_coords, camera_height, camera_x_offset, camera_matrix, dist_coeffs, clip=False
            )
            # 如果投影成功（返回2D数组），绘制轨迹
            if len(traj_pixels.shape) == 2:
                ax.plot(
                    traj_pixels[:250, 0],  # 限制前250个点
                    traj_pixels[:250, 1],
                    color=traj_colors[i],
                    lw=2.5,
                )

        # 绘制点
        for i, point in enumerate(list_points):
            # 确保点的形状正确
            if len(point.shape) == 1:
                # 添加batch维度
                point = point[None, :2]
            else:
                point = point[:, :2]
            # 投影点到图像平面
            pt_pixels = get_pos_pixels(
                point, camera_height, camera_x_offset, camera_matrix, dist_coeffs, clip=True
            )
            # 绘制点
            ax.plot(
                pt_pixels[:250, 0],
                pt_pixels[:250, 1],
                color=point_colors[i],
                marker="o",
                markersize=10.0,
            )
        
        # 隐藏坐标轴
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        # 设置图像范围
        ax.set_xlim((0.5, VIZ_IMAGE_SIZE[0] - 0.5))
        ax.set_ylim((VIZ_IMAGE_SIZE[1] - 0.5, 0.5))


def plot_trajs_and_points(
    ax: plt.Axes,
    list_trajs: list,
    list_points: list,
    traj_colors: list = [CYAN, MAGENTA],
    point_colors: list = [RED, GREEN],
    traj_labels: Optional[list] = ["prediction", "ground truth"],
    point_labels: Optional[list] = ["robot", "goal"],
    traj_alphas: Optional[list] = None,
    point_alphas: Optional[list] = None,
    quiver_freq: int = 1,
    default_coloring: bool = True,
):
    """
    绘制可能包含朝向信息的轨迹和点
    
    参数:
        ax: matplotlib坐标轴
        list_trajs: 轨迹列表，每条轨迹是形状为(horizon, 2)或(horizon, 4)的numpy数组
                   如果是4维，后两维表示朝向（sin(theta), cos(theta)）
        list_points: 点列表，每个点是形状为(2,)的numpy数组
        traj_colors: 轨迹颜色列表
        point_colors: 点颜色列表
        traj_labels: 轨迹标签列表（用于图例）
        point_labels: 点标签列表（用于图例）
        traj_alphas: 轨迹透明度列表
        point_alphas: 点透明度列表
        quiver_freq: 箭头绘制频率（如果轨迹包含朝向信息）
        default_coloring: 是否使用默认颜色
    
    功能:
        1. 绘制轨迹路径（使用线条和点）
        2. 如果轨迹包含朝向信息，使用箭头表示方向
        3. 绘制关键点（起点、终点等）
        4. 添加图例
    
    设计说明:
        - 支持2D轨迹（仅位置）和4D轨迹（位置+朝向）
        - 使用quiver绘制方向箭头
        - 透明度用于区分多条轨迹（如多模态预测）
        - 保持纵横比相等以避免失真
    """
    assert (
        len(list_trajs) <= len(traj_colors) or default_coloring
    ), "轨迹颜色数量不足"
    assert len(list_points) <= len(point_colors), "点颜色数量不足"
    assert (
        traj_labels is None or len(list_trajs) == len(traj_labels) or default_coloring
    ), "轨迹标签数量不足"
    assert point_labels is None or len(list_points) == len(point_labels), "点标签数量不足"

    # 绘制轨迹
    for i, traj in enumerate(list_trajs):
        if traj_labels is None:
            # 不使用标签
            ax.plot(
                traj[:, 0], 
                traj[:, 1], 
                color=traj_colors[i],
                alpha=traj_alphas[i] if traj_alphas is not None else 1.0,
                marker="o",
            )
        else:
            # 使用标签（用于图例）
            ax.plot(
                traj[:, 0],
                traj[:, 1],
                color=traj_colors[i],
                label=traj_labels[i],
                alpha=traj_alphas[i] if traj_alphas is not None else 1.0,
                marker="o",
            )
        
        # 如果轨迹包含朝向信息（4维），绘制方向箭头
        if traj.shape[1] > 2 and quiver_freq > 0:
            # 从路径点生成朝向向量
            bearings = gen_bearings_from_waypoints(traj)
            # 使用quiver绘制箭头（每quiver_freq个点绘制一个箭头）
            ax.quiver(
                traj[::quiver_freq, 0],  # x坐标
                traj[::quiver_freq, 1],  # y坐标
                bearings[::quiver_freq, 0],  # x方向分量
                bearings[::quiver_freq, 1],  # y方向分量
                color=traj_colors[i] * 0.5,  # 使用较暗的颜色
                scale=1.0,
            )
    
    # 绘制点
    for i, pt in enumerate(list_points):
        if point_labels is None:
            # 不使用标签
            ax.plot(
                pt[0], 
                pt[1], 
                color=point_colors[i], 
                alpha=point_alphas[i] if point_alphas is not None else 1.0,
                marker="o",
                markersize=7.0
            )
        else:
            # 使用标签（用于图例）
            ax.plot(
                pt[0],
                pt[1],
                color=point_colors[i],
                alpha=point_alphas[i] if point_alphas is not None else 1.0,
                marker="o",
                markersize=7.0,
                label=point_labels[i],
            )

    # 添加图例（放置在图下方）
    if traj_labels is not None or point_labels is not None:
        ax.legend()
        ax.legend(bbox_to_anchor=(0.0, -0.5), loc="upper left", ncol=2)
    
    # 设置纵横比相等，避免轨迹失真
    ax.set_aspect("equal", "box")


def angle_to_unit_vector(theta):
    """
    将角度转换为单位向量
    
    参数:
        theta: 角度（弧度）
    
    返回:
        单位向量 [cos(theta), sin(theta)]
    """
    return np.array([np.cos(theta), np.sin(theta)])


def gen_bearings_from_waypoints(
    waypoints: np.ndarray,
    mag=0.2,
) -> np.ndarray:
    """
    从路径点生成朝向向量
    
    参数:
        waypoints: 路径点数组 [horizon, 2或4]
                  如果是4维：(x, y, sin(theta), cos(theta))
                  如果是3维：(x, y, theta)
        mag: 向量的幅度（用于可视化箭头长度，默认0.2）
    
    返回:
        朝向向量数组 [horizon, 2]，每个向量表示该点的朝向
    
    设计说明:
        - 支持两种朝向表示：sin/cos表示和弧度表示
        - sin/cos表示更稳定，避免角度不连续问题
        - 向量归一化后乘以mag，控制箭头长度
    """
    bearing = []
    for i in range(0, len(waypoints)):
        if waypoints.shape[1] > 3:  # sin/cos表示
            v = waypoints[i, 2:]
            # 归一化向量
            v = v / np.linalg.norm(v)
            v = v * mag
        else:  # 弧度表示
            v = mag * angle_to_unit_vector(waypoints[i, 2])
        bearing.append(v)
    bearing = np.array(bearing)
    return bearing


def project_points(
    xy: np.ndarray,
    camera_height: float,
    camera_x_offset: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
):
    """
    使用相机参数将3D坐标投影到2D图像平面
    
    参数:
        xy: 坐标数组 [batch_size, horizon, 2]，表示(x, y)坐标
        camera_height: 相机离地面的高度（米）
        camera_x_offset: 相机相对于车辆中心的x轴偏移（米）
        camera_matrix: 3x3相机内参矩阵
        dist_coeffs: 畸变系数向量
    
    返回:
        uv: 2D图像坐标数组 [batch_size, horizon, 2]，表示(u, v)像素坐标
    
    工作流程:
        1. 将2D地面坐标扩展为3D坐标（z = -camera_height）
        2. 应用相机x轴偏移
        3. 转换坐标系（OpenCV坐标系）
        4. 使用cv2.projectPoints进行投影
        5. 考虑畸变系数进行校正
    
    设计说明:
        - 假设地面是平面（z = 0）
        - 相机朝向前方，高度固定
        - 使用OpenCV的投影函数处理畸变
    """
    batch_size, horizon, _ = xy.shape

    # 创建3D坐标，相机位于给定高度
    # z坐标为负值，因为相机在地面上方
    xyz = np.concatenate(
        [xy, -camera_height * np.ones(list(xy.shape[:-1]) + [1])], axis=-1
    )

    # 创建虚拟的旋转和平移向量（相机固定不动）
    rvec = tvec = (0, 0, 0)

    # 应用x轴偏移
    xyz[..., 0] += camera_x_offset
    
    # 转换到OpenCV坐标系：(y, -z, x)
    xyz_cv = np.stack([xyz[..., 1], -xyz[..., 2], xyz[..., 0]], axis=-1)
    
    # 使用OpenCV投影函数
    uv, _ = cv2.projectPoints(
        xyz_cv.reshape(batch_size * horizon, 3), rvec, tvec, camera_matrix, dist_coeffs
    )
    # 重塑回原始形状
    uv = uv.reshape(batch_size, horizon, 2)

    return uv


def get_pos_pixels(
    points: np.ndarray,
    camera_height: float,
    camera_x_offset: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    clip: Optional[bool] = False,
):
    """
    将3D坐标投影到2D图像平面并处理边界
    
    参数:
        points: 坐标数组 [batch_size, horizon, 2]，表示(x, y)坐标
        camera_height: 相机离地面的高度（米）
        camera_x_offset: 相机相对于车辆中心的x轴偏移（米）
        camera_matrix: 3x3相机内参矩阵
        dist_coeffs: 畸变系数向量
        clip: 是否裁剪到图像边界（True）或过滤超出边界的点（False）
    
    返回:
        pixels: 2D图像坐标数组 [N, 2]，表示(u, v)像素坐标
               如果clip=False，N可能小于输入点数（过滤了超出边界的点）
    
    功能:
        1. 调用project_points进行投影
        2. 水平翻转（镜像）
        3. 根据clip参数处理边界：
           - clip=True: 裁剪到图像范围内
           - clip=False: 过滤掉超出图像的点
    
    设计说明:
        - 水平翻转是为了匹配图像坐标系
        - clip=True用于关键点（如目标），确保可见
        - clip=False用于轨迹，避免显示不准确的投影
    """
    # 投影到图像平面
    pixels = project_points(
        points[np.newaxis], camera_height, camera_x_offset, camera_matrix, dist_coeffs
    )[0]
    
    # 水平翻转（镜像）
    pixels[:, 0] = VIZ_IMAGE_SIZE[0] - pixels[:, 0]
    
    if clip:
        # 裁剪到图像边界
        pixels = np.array(
            [
                [
                    np.clip(p[0], 0, VIZ_IMAGE_SIZE[0]),
                    np.clip(p[1], 0, VIZ_IMAGE_SIZE[1]),
                ]
                for p in pixels
            ]
        )
    else:
        # 过滤超出边界的点
        pixels = np.array(
            [
                p
                for p in pixels
                if np.all(p > 0) and np.all(p < [VIZ_IMAGE_SIZE[0], VIZ_IMAGE_SIZE[1]])
            ]
        )
    return pixels


def gen_camera_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """
    生成相机内参矩阵
    
    参数:
        fx: x方向焦距（像素）
        fy: y方向焦距（像素）
        cx: 主点x坐标（像素）
        cy: 主点y坐标（像素）
    
    返回:
        3x3相机内参矩阵:
        [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]
    
    说明:
        - fx, fy: 焦距，控制缩放
        - cx, cy: 主点，通常在图像中心
        - 这是标准的针孔相机模型
    """
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
