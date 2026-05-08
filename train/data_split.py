# ============================================================
# 数据集划分工具
# ============================================================
# 本脚本用于将轨迹数据划分为训练集和测试集
# 主要功能：
# 1. 扫描数据目录，查找包含轨迹数据的文件夹
# 2. 随机打乱文件夹顺序
# 3. 按指定比例划分训练集和测试集
# 4. 生成训练集和测试集的轨迹名称列表文件
#
# 使用场景：
# - 准备训练数据前的预处理步骤
# - 确保训练集和测试集的随机性和可重复性
# - 支持多个数据集的独立划分

import argparse
import os
import shutil
import random


def remove_files_in_dir(dir_path: str):
    """
    删除目录中的所有文件和子目录
    
    参数:
        dir_path: 要清空的目录路径
    
    功能:
        递归删除目录中的所有内容，包括：
        - 普通文件
        - 符号链接
        - 子目录及其内容
    
    设计说明:
        - 用于清理旧的数据划分，避免混淆
        - 使用异常处理确保删除失败时不会中断程序
        - 保留目录本身，只删除内容
    """
    for f in os.listdir(dir_path):
        file_path = os.path.join(dir_path, f)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                # 删除文件或符号链接
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                # 递归删除子目录
                shutil.rmtree(file_path)
        except Exception as e:
            print("删除 %s 失败。原因: %s" % (file_path, e))


def main(args: argparse.Namespace):
    """
    主函数：执行数据集划分
    
    参数:
        args: 命令行参数
            - data_dir: 包含轨迹数据的根目录
            - dataset_name: 数据集名称（用于组织输出）
            - split: 训练集比例（默认0.8，即80%训练，20%测试）
            - data_splits_dir: 数据划分输出目录
    
    工作流程:
        1. 扫描数据目录，查找包含'traj_data.pkl'的文件夹
        2. 随机打乱文件夹顺序（确保随机性）
        3. 按比例划分训练集和测试集
        4. 创建输出目录结构
        5. 生成训练集和测试集的轨迹名称列表文件
    
    输出结构:
        data_splits_dir/
        └── dataset_name/
            ├── train/
            │   └── traj_names.txt  # 训练集轨迹名称列表
            └── test/
                └── traj_names.txt  # 测试集轨迹名称列表
    """
    # ========== 步骤1: 扫描数据目录 ==========
    # 查找所有包含'traj_data.pkl'文件的文件夹
    # 这些文件夹代表有效的轨迹数据
    folder_names = [
        f
        for f in os.listdir(args.data_dir)
        if os.path.isdir(os.path.join(args.data_dir, f))  # 必须是目录
        and "traj_data.pkl" in os.listdir(os.path.join(args.data_dir, f))  # 必须包含轨迹数据文件
    ]

    # ========== 步骤2: 随机打乱 ==========
    # 随机打乱文件夹顺序，确保训练集和测试集的随机性
    # 注意：如果需要可重复的划分，可以在这里设置random.seed()
    random.shuffle(folder_names)

    # ========== 步骤3: 划分训练集和测试集 ==========
    # 根据split比例计算划分点
    # 例如：split=0.8，则前80%为训练集，后20%为测试集
    split_index = int(args.split * len(folder_names))
    train_folder_names = folder_names[:split_index]  # 训练集文件夹名称
    test_folder_names = folder_names[split_index:]   # 测试集文件夹名称

    # ========== 步骤4: 创建输出目录 ==========
    # 创建训练集和测试集的目录结构
    train_dir = os.path.join(args.data_splits_dir, args.dataset_name, "train")
    test_dir = os.path.join(args.data_splits_dir, args.dataset_name, "test")
    
    for dir_path in [train_dir, test_dir]:
        if os.path.exists(dir_path):
            # 如果目录已存在，清空其内容（避免旧数据混淆）
            print(f"清空 {dir_path} 中的文件以准备新的数据划分")
            remove_files_in_dir(dir_path)
        else:
            # 如果目录不存在，创建它
            print(f"创建目录 {dir_path}")
            os.makedirs(dir_path)

    # ========== 步骤5: 写入轨迹名称列表 ==========
    # 将训练集文件夹名称写入traj_names.txt
    with open(os.path.join(train_dir, "traj_names.txt"), "w") as f:
        for folder_name in train_folder_names:
            f.write(folder_name + "\n")

    # 将测试集文件夹名称写入traj_names.txt
    with open(os.path.join(test_dir, "traj_names.txt"), "w") as f:
        for folder_name in test_folder_names:
            f.write(folder_name + "\n")

    # 打印划分统计信息
    print(f"数据划分完成:")
    print(f"  总轨迹数: {len(folder_names)}")
    print(f"  训练集: {len(train_folder_names)} ({len(train_folder_names)/len(folder_names)*100:.1f}%)")
    print(f"  测试集: {len(test_folder_names)} ({len(test_folder_names)/len(folder_names)*100:.1f}%)")


if __name__ == "__main__":
    # ========== 命令行参数解析 ==========
    parser = argparse.ArgumentParser(
        description="将轨迹数据随机划分为训练集和测试集"
    )

    parser.add_argument(
        "--data-dir", "-i", 
        help="包含轨迹数据的目录路径（每个子文件夹应包含traj_data.pkl）", 
        required=True
    )
    parser.add_argument(
        "--dataset-name", "-d", 
        help="数据集名称（用于组织输出目录结构）", 
        required=True
    )
    parser.add_argument(
        "--split", "-s", 
        type=float, 
        default=0.8, 
        help="训练集比例（默认: 0.8，即80%%训练，20%%测试）"
    )
    parser.add_argument(
        "--data-splits-dir", "-o", 
        default="vint_train/data/data_splits", 
        help="数据划分输出目录（默认: vint_train/data/data_splits）"
    )
    
    args = parser.parse_args()
    
    # 执行数据划分
    main(args)
    print("完成！")
