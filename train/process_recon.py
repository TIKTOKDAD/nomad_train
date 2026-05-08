# -*- coding: utf-8 -*-
# 该脚本用于将 RECON 数据集（h5 文件）转换为训练可直接读取的目录结构。
# 每条轨迹会输出到独立文件夹，包含：
#   1) traj_data.pkl：位置与航向角数据
#   2) 连续图像帧：0.jpg, 1.jpg, ...

import h5py
import os
import pickle
from PIL import Image
import io
import argparse
import tqdm


def main(args: argparse.Namespace):
    """
    主处理入口。

    参数:
        args.input_dir: RECON 数据集根目录（内部应包含 recon_release 子目录）
        args.output_dir: 处理后数据输出目录
        args.num_trajs: 最多处理多少条轨迹，-1 表示处理全部
    """
    # RECON 原始 h5 文件所在目录
    recon_dir = os.path.join(args.input_dir, "recon_release")
    # 处理后输出目录
    output_dir = args.output_dir

    # 若输出目录不存在则创建
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取 recon_release 下所有文件名（通常为 h5 轨迹文件）
    filenames = os.listdir(recon_dir)
    # 若设置了轨迹数量上限，仅处理前 num_trajs 条
    # 默认 -1 表示不截断
    if args.num_trajs >= 0:
        filenames = filenames[: args.num_trajs]

    # 主处理循环：逐个轨迹文件转换
    for filename in tqdm.tqdm(filenames, desc="Trajectories processed"):
        # 轨迹名：文件名去掉扩展名
        traj_name = filename.split(".")[0]

        # 打开 h5 文件；若文件损坏或不可读则跳过
        try:
            h5_f = h5py.File(os.path.join(recon_dir, filename), "r")
        except OSError:
            print(f"Error loading {filename}. Skipping...")
            continue

        # 提取位姿数据：
        # - position: 取前两维 (x, y)
        # - yaw: 航向角序列
        position_data = h5_f["jackal"]["position"][:, :2]
        yaw_data = h5_f["jackal"]["yaw"][()]

        # 封装轨迹标签数据，供训练阶段读取
        traj_data = {"position": position_data, "yaw": yaw_data}
        traj_folder = os.path.join(output_dir, traj_name)

        # 创建轨迹输出目录，并保存轨迹数据
        os.makedirs(traj_folder, exist_ok=True)
        with open(os.path.join(traj_folder, "traj_data.pkl"), "wb") as f:
            pickle.dump(traj_data, f)

        # 冗余保护：若目录不存在则创建（与上面的 exist_ok=True 作用类似）
        if not os.path.exists(traj_folder):
            os.makedirs(traj_folder)

        # 将图像字节序列解码为 RGB 图片并按时间索引保存：
        # h5 路径：images/rgb_left
        # 输出命名：0.jpg, 1.jpg, ...
        for i in range(h5_f["images"]["rgb_left"].shape[0]):
            img = Image.open(io.BytesIO(h5_f["images"]["rgb_left"][i]))
            img.save(os.path.join(traj_folder, f"{i}.jpg"))


if __name__ == "__main__":
    # 命令行参数定义
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        help="RECON 数据集根目录（内部应包含 recon_release）",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="datasets/recon/",
        type=str,
        help="处理后数据输出目录（默认: datasets/recon/）",
    )
    # 最多处理多少条轨迹
    parser.add_argument(
        "--num-trajs",
        "-n",
        default=-1,
        type=int,
        help="最多处理多少条轨迹（默认: -1，表示全部）",
    )

    args = parser.parse_args()
    # 启动处理流程
    print("STARTING PROCESSING RECON DATASET")
    main(args)
    print("FINISHED PROCESSING RECON DATASET")
