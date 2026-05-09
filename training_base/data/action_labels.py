# ============================================================
# Navigation action labels - local waypoint label builder
# ============================================================
# 本文件提供一个轻量包装类，把 Dataset 中的配置参数固化下来：
# 1. NavigationDataset 只需要传入当前轨迹片段信息
# 2. 真正的坐标转换和标签计算委托给 labeling.compute_navigation_actions
# 3. 这样可以让采样逻辑与动作标签逻辑保持解耦

from training_base.data.labeling import compute_navigation_actions


# 动作标签构造器：保存固定超参数，并为每个样本生成局部坐标系标签
class NavigationActionLabelBuilder:
    """Builds local-frame waypoint labels for navigation training."""

    # 初始化标签构造所需的固定配置
    def __init__(
        self,
        *,
        len_traj_pred: int,
        waypoint_spacing: int,
        learn_angle: bool,
        normalize: bool,
        metric_waypoint_spacing: float,
        num_action_params: int,
        dataset_name: str,
    ) -> None:
        self.len_traj_pred = int(len_traj_pred)
        self.waypoint_spacing = int(waypoint_spacing)
        self.learn_angle = bool(learn_angle)
        self.normalize = bool(normalize)
        self.metric_waypoint_spacing = float(metric_waypoint_spacing)
        self.num_action_params = int(num_action_params)
        self.dataset_name = dataset_name

    # 为单个观测-目标对构造动作序列和目标位置
    def build(self, *, traj_data, curr_time: int, goal_time: int, trajectory_name: str):
        # compute_navigation_actions 会完成轨迹裁剪、局部坐标变换、角度处理和归一化
        return compute_navigation_actions(
            traj_data=traj_data,
            curr_time=curr_time,
            goal_time=goal_time,
            len_traj_pred=self.len_traj_pred,
            waypoint_spacing=self.waypoint_spacing,
            learn_angle=self.learn_angle,
            normalize=self.normalize,
            metric_waypoint_spacing=self.metric_waypoint_spacing,
            num_action_params=self.num_action_params,
            dataset_name=self.dataset_name,
            trajectory_name=trajectory_name,
        )
