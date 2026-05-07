# -*- coding: utf-8 -*-
# 该脚本用于将原始 ROS bag 批量转换为训练数据目录。
# 转换后每条（或每个裁剪后的）轨迹目录通常包含：
#   1) 连续图像帧：0.jpg, 1.jpg, ...
#   2) 轨迹数据：traj_data.pkl（位姿/里程计等处理后的序列）
# 具体读取哪些话题、如何处理图像/里程计，由 process_bags_config.yaml 按 dataset-name 配置。

import os
import pickle
from PIL import Image
import io
import argparse
import tqdm
import yaml
import rosbag

# utils
from vint_train.process_data.process_data_utils import *


def main(args: argparse.Namespace):
    """
    主处理入口。

    参数:
        args.dataset_name: 数据集配置名（必须在 process_bags_config.yaml 中存在）
        args.input_dir: 待扫描的 bag 根目录（递归查找 .bag）
        args.output_dir: 处理后数据输出目录
        args.num_trajs: 最多处理多少个 bag，-1 表示全部
        args.sample_rate: bag 消息采样频率（Hz）
    """

    # 读取处理配置文件（按 dataset_name 选择对应的话题与处理函数）
    with open("vint_train/process_data/process_bags_config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # 若输出目录不存在则创建，避免后续保存文件失败
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 递归扫描输入目录，收集全部 .bag 文件路径
    bag_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(".bag"):
                bag_files.append(os.path.join(root, file))

    # 如设置了数量上限，仅处理前 num_trajs 条
    # 默认 -1 表示不截断，处理所有 bag
    if args.num_trajs >= 0:
        bag_files = bag_files[: args.num_trajs]

    # 主处理循环：逐个 bag 读取并落盘
    for bag_path in tqdm.tqdm(bag_files, desc="Bags processed"):
        try:
            # 打开 bag；文件损坏或格式不合法时跳过
            b = rosbag.Bag(bag_path)
        except rosbag.ROSBagException as e:
            print(e)
            print(f"Error loading {bag_path}. Skipping...")
            continue

        # 构造轨迹基名：取路径最后两级（父目录 + 文件名）并以下划线连接，再去掉 .bag
        # 目的：降低不同目录下同名 bag 的命名冲突概率
        traj_name = "_".join(bag_path.split("/")[-2:])[:-4]

        # 从 bag 中提取图像与里程计序列：
        # - imtopics / odomtopics：由配置指定读取哪些 ROS 话题
        # - img_process_func / odom_process_func：由配置指定预处理函数（字符串 -> 函数）
        # - rate：按采样频率下采样，控制时间步密度与数据体量
        # - ang_offset：角度偏移修正参数（由具体数据集配置决定）
        bag_img_data, bag_traj_data = get_images_and_odom(
            b,
            config[args.dataset_name]["imtopics"],
            config[args.dataset_name]["odomtopics"],
            eval(config[args.dataset_name]["img_process_func"]),
            eval(config[args.dataset_name]["odom_process_func"]),
            rate=args.sample_rate,
            ang_offset=config[args.dataset_name]["ang_offset"],
        )

        # 关键话题缺失时无法构建样本，直接跳过
        if bag_img_data is None or bag_traj_data is None:
            print(
                f"{bag_path} did not have the topics we were looking for. Skipping..."
            )
            continue

        # 过滤并切分“后退运动”片段，返回多个可用子轨迹
        # 返回格式通常为 [(img_seq_0, traj_seq_0), (img_seq_1, traj_seq_1), ...]
        cut_trajs = filter_backwards(bag_img_data, bag_traj_data)

        # 逐个子轨迹保存到独立目录
        for i, (img_data_i, traj_data_i) in enumerate(cut_trajs):
            traj_name_i = traj_name + f"_{i}"
            traj_folder_i = os.path.join(args.output_dir, traj_name_i)
            # 为当前子轨迹创建输出目录
            if not os.path.exists(traj_folder_i):
                os.makedirs(traj_folder_i)

            # 保存轨迹序列（pickle），供训练阶段读取
            with open(os.path.join(traj_folder_i, "traj_data.pkl"), "wb") as f:
                pickle.dump(traj_data_i, f)

            # 按时间索引保存图像帧：0.jpg, 1.jpg, ...
            for i, img in enumerate(img_data_i):
                img.save(os.path.join(traj_folder_i, f"{i}.jpg"))


if __name__ == "__main__":
    # 命令行参数定义
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-name",
        "-d",
        type=str,
        help="数据集配置名（必须在 process_bags_config.yaml 中存在）",
        default="tartan_drive",
        required=True,
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        help="待处理 bag 数据集根目录（递归扫描）",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="../datasets/tartan_drive/",
        type=str,
        help="处理后数据输出目录（默认: ../datasets/tartan_drive/）",
    )
    # 最多处理多少个 bag
    parser.add_argument(
        "--num-trajs",
        "-n",
        default=-1,
        type=int,
        help="最多处理多少个 bag（默认: -1，表示全部）",
    )
    # 采样频率（Hz）
    parser.add_argument(
        "--sample-rate",
        "-s",
        default=4.0,
        type=float,
        help="bag 消息采样频率（默认: 4.0 Hz）",
    )

    args = parser.parse_args()
    # 启动处理流程
    print(f"STARTING PROCESSING {args.dataset_name.upper()} DATASET")
    main(args)
    print(f"FINISHED PROCESSING {args.dataset_name.upper()} DATASET")
