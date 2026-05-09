from training_base.data.labeling import compute_navigation_actions


class NavigationActionLabelBuilder:
    """Builds local-frame waypoint labels for navigation training."""

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

    def build(self, *, traj_data, curr_time: int, goal_time: int, trajectory_name: str):
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
