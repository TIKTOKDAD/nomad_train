# -*- coding: utf-8 -*-
# 该脚本用于将包含“观测图像 + 差分子目标图像 + 里程计”的 ROS bag
# 批量转换为训练可直接读取的文件夹结构：
#   每条轨迹一个目录，目录内包含：
#   1) 连续观测图像：0.jpg, 1.jpg, ...
#   2) 对应差分图像：diff_0.jpg, diff_1.jpg, ...
#   3) 轨迹数据：traj_data.pkl（由 /odom 话题序列序列化得到）

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
        args.input_dir: 待扫描的 bag 根目录（递归查找 .bag）
        args.output_dir: 处理后数据输出目录
        args.num_trajs: 最多处理多少条 bag，-1 表示处理全部
        args.sample_rate: 采样频率（Hz），用于下采样 bag 消息
    """

    # 读取处理配置（当前脚本中未直接使用 config，保留用于与同目录处理流程兼容）
    with open("vint_train/process_data/process_bags_config.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # 若输出目录不存在则创建，确保后续写文件不会失败
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 递归扫描输入目录，筛选出文件名中包含 "diff" 的 .bag 文件
    # 这样可以将差分数据与普通 bag 区分开来
    bag_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith(".bag") and "diff" in file:
                bag_files.append(os.path.join(root, file))

    # 如果指定了处理数量上限，则只取前 num_trajs 条
    # num_trajs < 0（默认 -1）时表示不截断，处理所有匹配到的 bag
    if args.num_trajs >= 0:
        bag_files = bag_files[: args.num_trajs]

    # 主处理循环：逐个 bag 转换为轨迹目录
    for bag_path in tqdm.tqdm(bag_files, desc="Bags processed"):
        try:
            # 打开 ROS bag；如果文件损坏或格式异常则跳过
            b = rosbag.Bag(bag_path)
        except rosbag.ROSBagException as e:
            print(e)
            print(f"Error loading {bag_path}. Skipping...")
            continue

        # 轨迹目录名构造规则：
        # 取 bag 路径最后两级（父目录 + 文件名）并用 "_" 拼接，再去掉 ".bag" 后缀
        # 这样可降低不同子目录下同名 bag 的命名冲突概率
        traj_name = "_".join(bag_path.split("/")[-2:])[:-4]

        # 从 bag 中提取图像与里程计：
        # 图像话题：
        #   /usb_cam_front/image_raw  -> 原始观测图
        #   /chosen_subgoal           -> 差分子目标图
        # 轨迹话题：
        #   /odom                     -> 位姿/里程计序列
        # rate 控制采样频率，降低数据体量并统一时间步密度
        bag_img_data, bag_traj_data = get_images_and_odom_2(
            b,
            ['/usb_cam_front/image_raw', '/chosen_subgoal'],
            ['/odom'],
            rate=args.sample_rate,
        )
  
        # 若关键话题缺失，无法构造监督数据，直接跳过该 bag
        if bag_img_data is None:
            print(
                f"{bag_path} did not have the topics we were looking for. Skipping..."
            )
            continue
        # remove backwards movement
        # cut_trajs = filter_backwards(bag_img_data, bag_traj_data)

        # for i, (img_data_i, traj_data_i) in enumerate(cut_trajs):
        #     traj_name_i = traj_name + f"_{i}"
        #     traj_folder_i = os.path.join(args.output_dir, traj_name_i)
        #     # make a folder for the traj
        #     if not os.path.exists(traj_folder_i):
        #         os.makedirs(traj_folder_i)
        #     with open(os.path.join(traj_folder_i, "traj_data.pkl"), "wb") as f:
        #         pickle.dump(traj_data_i, f)
        #     # save the image data to disk
        #     for i, img in enumerate(img_data_i):
        #         img.save(os.path.join(traj_folder_i, f"{i}.jpg"))

        # 为当前轨迹创建输出目录
        traj_folder = os.path.join(args.output_dir, traj_name)
        if not os.path.exists(traj_folder):
                os.makedirs(traj_folder)
        
        # 取出两路图像序列（按时间对齐后由 get_images_and_odom_2 返回）
        obs_images = bag_img_data["/usb_cam_front/image_raw"]
        diff_images = bag_img_data["/chosen_subgoal"]

        # 将观测图与差分图按同一索引成对写盘：
        #   i.jpg      -> 第 i 帧观测图
        #   diff_i.jpg -> 第 i 帧对应差分图
        for i, img_data in enumerate(zip(obs_images, diff_images)):
            obs_image, diff_image = img_data
            # 保存图像到当前轨迹目录
            obs_image.save(os.path.join(traj_folder, f"{i}.jpg"))
            diff_image.save(os.path.join(traj_folder, f"diff_{i}.jpg"))

        # 将 /odom 对应的轨迹序列序列化保存，供后续训练读取
        with open(os.path.join(traj_folder, "traj_data.pkl"), "wb") as f:
                pickle.dump(bag_traj_data['/odom'], f)


if __name__ == "__main__":
    # 命令行参数定义
    parser = argparse.ArgumentParser()
    # get arguments for the recon input dir and the output dir
    # add dataset name
    # parser.add_argument(
    #     "--dataset-name",
    #     "-d",
    #     type=str,
    #     help="name of the dataset (must be in process_config.yaml)",
    #     default="tartan_drive",
    #     required=True,
    # )
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
    # number of trajs to process
    parser.add_argument(
        "--num-trajs",
        "-n",
        default=-1,
        type=int,
        help="最多处理多少个 bag（默认: -1，表示全部）",
    )
    # sampling rate
    parser.add_argument(
        "--sample-rate",
        "-s",
        default=4.0,
        type=float,
        help="bag 消息采样频率（默认: 4.0 Hz）",
    )

    args = parser.parse_args()
    # 启动处理流程
    print(f"STARTING PROCESSING DIFF DATASET")
    main(args)
    print(f"FINISHED PROCESSING DIFF DATASET")
