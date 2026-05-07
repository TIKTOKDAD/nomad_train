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
#
# 使用方式：
#   单数据集模式（原有方式）：
#     python data_split.py --data-dir /path/to/recon --dataset-name recon
#
#   批量模式（新增）：
#     python data_split.py --data-dir /path/to/vint_dataset --batch
#     自动扫描根路径下的所有子数据集并逐一切分

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


def split_single_dataset(data_dir: str, dataset_name: str, split_ratio: float, data_splits_dir: str):
    """
    对单个数据集执行训练集/测试集划分
    
    参数:
        data_dir: 该数据集的目录路径（包含多个轨迹子文件夹）
        dataset_name: 数据集名称（用于组织输出目录）
        split_ratio: 训练集比例（如 0.8 表示 80% 训练，20% 测试）
        data_splits_dir: 数据划分输出的根目录
    
    返回:
        True 表示成功，False 表示该目录下没有有效轨迹
    
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
        for f in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, f))  # 必须是目录
        and "traj_data.pkl" in os.listdir(os.path.join(data_dir, f))  # 必须包含轨迹数据文件
    ]

    if len(folder_names) == 0:
        print(f"  [跳过] '{dataset_name}': 未找到包含 traj_data.pkl 的轨迹文件夹")
        return False

    # ========== 步骤2: 随机打乱 ==========
    # 随机打乱文件夹顺序，确保训练集和测试集的随机性
    # 注意：如果需要可重复的划分，可以在这里设置random.seed()
    random.shuffle(folder_names)

    # ========== 步骤3: 划分训练集和测试集 ==========
    # 根据split比例计算划分点
    # 例如：split=0.8，则前80%为训练集，后20%为测试集
    split_index = int(split_ratio * len(folder_names))
    train_folder_names = folder_names[:split_index]  # 训练集文件夹名称
    test_folder_names = folder_names[split_index:]   # 测试集文件夹名称

    # ========== 步骤4: 创建输出目录 ==========
    # 创建训练集和测试集的目录结构
    train_dir = os.path.join(data_splits_dir, dataset_name, "train")
    test_dir = os.path.join(data_splits_dir, dataset_name, "test")
    
    for dir_path in [train_dir, test_dir]:
        if os.path.exists(dir_path):
            # 如果目录已存在，清空其内容（避免旧数据混淆）
            print(f"  清空 {dir_path} 中的文件以准备新的数据划分")
            remove_files_in_dir(dir_path)
        else:
            # 如果目录不存在，创建它
            print(f"  创建目录 {dir_path}")
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
    print(f"  [完成] '{dataset_name}': 总轨迹数 {len(folder_names)}, "
          f"训练集 {len(train_folder_names)} ({len(train_folder_names)/len(folder_names)*100:.1f}%), "
          f"测试集 {len(test_folder_names)} ({len(test_folder_names)/len(folder_names)*100:.1f}%)")
    return True


def main(args: argparse.Namespace):
    """
    主函数：根据模式执行数据集划分
    
    参数:
        args: 命令行参数
            - data_dir: 数据目录路径
            - dataset_name: 数据集名称（单数据集模式必填）
            - batch: 是否启用批量模式
            - split: 训练集比例（默认0.8）
            - data_splits_dir: 数据划分输出目录
    """
    if args.batch:
        # ========== 批量模式 ==========
        # data_dir 指向根路径，其下每个子目录视为一个独立数据集
        # 例如: vint_dataset/ 下有 recon/, sacson/, cory_hall/ 等
        print(f"=" * 60)
        print(f"批量模式: 扫描根目录 '{args.data_dir}'")
        print(f"训练集比例: {args.split * 100:.0f}%")
        print(f"输出目录: {args.data_splits_dir}")
        print(f"=" * 60)

        # 获取所有子目录（即各个数据集）
        sub_dirs = sorted([
            d for d in os.listdir(args.data_dir)
            if os.path.isdir(os.path.join(args.data_dir, d))
        ])

        if len(sub_dirs) == 0:
            print(f"错误: 根目录 '{args.data_dir}' 下没有找到任何子目录")
            return

        print(f"发现 {len(sub_dirs)} 个子目录: {sub_dirs}\n")

        success_count = 0
        skip_count = 0

        for dataset_name in sub_dirs:
            dataset_path = os.path.join(args.data_dir, dataset_name)
            print(f"--- 处理数据集: {dataset_name} ---")
            if split_single_dataset(dataset_path, dataset_name, args.split, args.data_splits_dir):
                success_count += 1
            else:
                skip_count += 1

        # 打印汇总
        print(f"\n{'=' * 60}")
        print(f"批量处理完成:")
        print(f"  成功划分: {success_count} 个数据集")
        print(f"  跳过: {skip_count} 个目录（无有效轨迹）")
        print(f"{'=' * 60}")

    else:
        # ========== 单数据集模式（原有逻辑） ==========
        if not args.dataset_name:
            print("错误: 单数据集模式下必须指定 --dataset-name 参数")
            print("提示: 如需批量处理根路径下的所有数据集，请使用 --batch 参数")
            return

        print(f"单数据集模式: {args.dataset_name}")
        split_single_dataset(args.data_dir, args.dataset_name, args.split, args.data_splits_dir)


if __name__ == "__main__":
    # ========== 命令行参数解析 ==========
    parser = argparse.ArgumentParser(
        description="将轨迹数据随机划分为训练集和测试集（支持单数据集和批量模式）"
    )

    parser.add_argument(
        "--data-dir", "-i", 
        help="数据目录路径。单数据集模式: 指向具体数据集; 批量模式: 指向包含多个数据集的根目录", 
        required=True
    )
    parser.add_argument(
        "--dataset-name", "-d", 
        help="数据集名称（单数据集模式必填，批量模式下自动使用子目录名）", 
        default=None
    )
    parser.add_argument(
        "--batch", "-b",
        action="store_true",
        help="启用批量模式: 自动扫描 data-dir 下的所有子目录，将每个子目录视为独立数据集并分别划分"
    )
    parser.add_argument(
        "--split", "-s", 
        type=float, 
        default=0.8, 
        help="训练集比例（默认: 0.8，即80%%训练，20%%测试）"
    )
    parser.add_argument(
        "--data-splits-dir", "-o", 
        default="/root/data1/visualnav-transformer_4_60/data_splits", 
        help="数据划分输出目录（默认: vint_train/data/data_splits）"
    )
    
    args = parser.parse_args()
    
    # 执行数据划分
    main(args)
    print("完成！")
