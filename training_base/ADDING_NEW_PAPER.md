# 新增视觉导航论文训练接入指南

本文档说明：当你要在 `training_base` 中实现一篇新论文的新网络训练时，应该改哪些文件、哪些文件尽量不要改，以及如何按现有 `GNM / ViNT / NoMaD` 的风格完成一次完整接入。

`training_base` 的核心设计是：

```text
YAML config
  -> registry
  -> Algorithm
  -> Model / Objective / Metric / Visualizer
  -> Trainer
```

也就是说，新论文的差异应该优先落在 `algorithm + model + objective + config` 这几个插件点里。`trainer.py` 只负责通用训练循环、DDP、AMP、日志、评估和 callback，不应该因为某一篇论文新增 `if paper_name == ...` 这种分支。

## 1. 先判断你新增的是哪一类变化

新增论文前，先把变化拆清楚。不同变化对应不同文件，不要一上来全目录都改。

| 新论文变化 | 优先修改位置 | 是否通常需要改 `trainer.py` |
| --- | --- | --- |
| 只是换视觉 backbone | `training_base/modules/vision/`、`training_base/modules/vision/__init__.py`、配置里的 `model.encoder.name` 或 `model.vision_encoder.name` | 不需要 |
| 只是换预测头 | `training_base/modules/heads/`、`training_base/modules/heads/__init__.py`、模型 builder | 不需要 |
| 只是换模型组装方式 | `training_base/models/<new_model>.py`、`training_base/models/__init__.py`、配置里的 `model.name` | 不需要 |
| 还是监督式航点预测 | 可以复用 `SupervisedWaypointAlgorithm` 和 `supervised_waypoint` objective | 不需要 |
| 损失组合方式变了 | `training_base/objectives/<new_objective>.py`、`training_base/objectives/__init__.py` | 不需要 |
| 单个 loss primitive 变了 | `training_base/losses/`、`training_base/losses/__init__.py` | 不需要 |
| train/eval step 变了 | `training_base/algorithms/<new_algorithm>.py`、`training_base/algorithms/__init__.py` | 不需要 |
| 需要 EMA、teacher model、diffusion scheduler 等算法状态 | `training_base/algorithms/<new_algorithm>.py` 的 `create_state()` / `state_dict()` / `model_for_eval()` | 不需要 |
| 需要额外低频行为指标 | `training_base/metrics/`、`training_base/metrics/__init__.py`、配置里的 `metrics.heavy` | 不需要 |
| 需要新图像/轨迹可视化 | `training_base/visualizers/`、`training_base/visualizers/__init__.py`、配置里的 `visualization.train/eval` | 不需要 |
| 数据集返回字段真的不够 | `training_base/data/navigation_dataset.py`、`training_base/data/batch.py`、新 algorithm 的 `prepare_batch()` | 一般不需要 |
| 新增通用训练基础设施，例如梯度累积或全局 callback hook | `training_base/trainer.py`、`training_base/callbacks/` | 可能需要 |

判断标准很简单：

- “这篇论文自己的训练配方”放 `algorithms/` 和 `objectives/`。
- “网络长什么样”放 `models/` 和 `modules/`。
- “所有论文都共享的训练机制”才考虑放 `trainer.py`。

## 2. 当前代码里的关键扩展点

### 2.1 `training_base/registry.py`

这里定义全局注册表：

```python
algorithm_registry
model_registry
objective_registry
metric_registry
visualizer_registry
callback_registry
log_sink_registry
module_registry
loss_registry
optimizer_registry
scheduler_registry
noise_scheduler_registry
```

YAML 中的 `name` 字段最终都会映射到这些 registry key。比如：

```yaml
algorithm:
  name: gnm

model:
  name: gnm

objective:
  name: supervised_waypoint
```

CLI 会先调用 `register_builtins()`，它会导入 `training_base.algorithms`、`training_base.models`、`training_base.modules`、`training_base.data` 等包，从而触发装饰器注册。

因此你新增文件后，必须保证它被对应包的 `__init__.py` 导入。只写了新文件但没有导入，registry 就找不到它。

### 2.2 `training_base/algorithms/base.py`

新增算法类至少要实现这些方法：

```python
def build_model(self, config): ...
def build_objective(self, config): ...
def prepare_batch(self, batch, transform, device, mode: str, should_log_images: bool): ...
def train_step(self, model, prepared, state, config) -> StepResult: ...
def eval_step(self, model, prepared, state, config) -> StepResult: ...
```

按需实现这些方法：

```python
def create_state(...): ...
def prepare_resume(...): ...
def model_for_eval(...): ...
def after_optimizer_step(...): ...
def heavy_metrics(...): ...
def light_metrics(...): ...
def visualize(...): ...
def state_dict(...): ...
def step_scheduler(...): ...
```

`Trainer` 只认这些接口。新增论文时，只要这些接口实现正确，通用训练循环就能复用。

### 2.3 `training_base/models/__init__.py`

这里是模型 builder 的集中入口。现有模式是：

```python
@model_registry.register("gnm")
def build_gnm(config) -> ModelBuild:
    ...
    return ModelBuild(model=model, extras={})
```

`ModelBuild.model` 是真正的 `nn.Module`，会被 `Trainer` 移动到 GPU、包 DDP、保存 checkpoint。

`ModelBuild.extras` 放“不属于模型参数但训练时需要”的对象，例如 NoMaD 的 `noise_scheduler`。

### 2.4 `training_base/modules/`

这里放可复用神经网络积木。比如：

- `modules/vision/`: 视觉编码器。
- `modules/heads/`: 距离头、航点头。
- `modules/transformer/`: transformer block。
- `modules/diffusion/`: diffusion 网络。

模块应该尽量只依赖构造参数，不直接读全局 YAML。YAML 到构造参数的转换放在对应 `__init__.py` 的 builder 里。

### 2.5 `training_base/objectives/`

这里放 loss recipe。比如：

- `supervised_waypoint`: 距离损失 + 动作航点损失。
- `nomad_diffusion`: 距离损失 + 扩散噪声预测损失。

如果新论文只是把 MSE 换成 L1，一般扩展 `losses/` 或配置即可。

如果新论文的 loss 组合方式变了，就新增 objective。

### 2.6 `training_base/data/batch.py`

当前标准 batch 字段是：

```python
obs_image
goal_image
actions
distance
goal_pos
dataset_index
action_mask
metric_scale
extras
```

大多数视觉导航论文都应该先尝试复用这个 batch 协议。只有当新论文真的需要额外标签，例如拓扑节点、语言指令、地图 patch 等，才扩展 `NavigationBatch`。如果是数据来源或采样流程不同，优先新增 data-module 并注册到 `data_module_registry`，不要继续给 `NavigationDataset` 加模式分支。

## 3. 标准新增流程

### Step 1: 给新论文确定 registry 名字

先统一命名，比如新论文叫 `paper_nav`：

```text
algorithm.name = paper_nav
model.name = paper_nav
objective.name = supervised_waypoint 或 paper_nav_objective
module encoder name = paper_nav_encoder
visualizer name = paper_nav 或复用 supervised_waypoint
```

这些名字要在 YAML 和注册代码中完全一致。

### Step 2: 新增可复用网络模块

如果论文引入新 encoder/head/block，放到 `modules/` 下，并在对应 `__init__.py` 注册。

常见路径：

```text
training_base/modules/vision/<new_encoder>.py
training_base/modules/heads/<new_head>.py
training_base/modules/transformer/<new_block>.py
training_base/modules/diffusion/<new_diffusion_net>.py
```

### Step 3: 新增 model

如果只是换 encoder，而且输出仍是 `dist_pred, action_pred`，可以继续复用 `SupervisedWaypointAlgorithm`。

如果模型前向接口有变化，新增：

```text
training_base/models/<new_model>.py
```

并在：

```text
training_base/models/__init__.py
```

注册 builder。

### Step 4: 新增 objective

如果新论文仍是监督式航点回归，直接复用：

```yaml
objective:
  name: supervised_waypoint
```

如果 loss recipe 变了，新增：

```text
training_base/objectives/<new_objective>.py
training_base/objectives/__init__.py
```

### Step 5: 新增 algorithm

如果 train/eval step 和 GNM/ViNT 一样，只是换模型，新增 algorithm 可以很薄：

```python
@algorithm_registry.register("paper_nav")
class PaperNavAlgorithm(SupervisedWaypointAlgorithm):
    name = "paper_nav"
```

如果训练逻辑不同，就继承 `Algorithm`，自己实现 `prepare_batch()`、`train_step()`、`eval_step()` 等。

### Step 6: 新增 config

新增：

```text
training_base/configs/<paper_nav>.yaml
```

配置至少要指定：

```yaml
algorithm:
  name: paper_nav

model:
  name: paper_nav

objective:
  name: supervised_waypoint
```

再补数据、优化器、日志、可视化等字段。

### Step 7: 验证

建议至少跑：

```powershell
python -m compileall -q training_base
python -m training_base.cli -c training_base/configs/paper_nav.yaml --build-lmdb-only
python -m training_base.cli -c training_base/configs/paper_nav.yaml
```

多卡训练前，先单进程构建 LMDB，再用 `torchrun`：

```powershell
python -m training_base.cli -c training_base/configs/paper_nav.yaml --build-lmdb-only
torchrun --standalone --nproc_per_node=2 -m training_base.cli -c training_base/configs/paper_nav.yaml
```

## 4. 完整新增示例：新增一个监督式航点论文 `paper_nav`

下面给一个完整示例。假设新论文仍然使用视觉导航数据协议：

- 输入：多帧观测图像 `obs_image` + 目标图像 `goal_image`。
- 输出：目标距离 `dist_pred` + 未来航点 `action_pred`。
- 损失：复用 `supervised_waypoint`。
- 可视化：复用 `supervised_waypoint`。
- 训练循环：复用 `SupervisedWaypointAlgorithm`。
- 新增内容：新的视觉编码器 + 新的模型 wrapper + 新 YAML。

这类新增是最常见、最干净的接入方式。

### 4.1 新增视觉编码器

新增文件：

```text
training_base/modules/vision/paper_nav_encoder.py
```

示例代码：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class PaperNavEncoder(nn.Module):
    def __init__(
        self,
        *,
        context_size: int,
        image_channels: int = 3,
        hidden_dim: int = 256,
        output_dim: int = 512,
    ) -> None:
        super().__init__()
        self.context_size = context_size
        self.output_dim = output_dim

        obs_channels = image_channels * (context_size + 1)
        self.obs_encoder = nn.Sequential(
            nn.Conv2d(obs_channels, hidden_dim // 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.goal_encoder = nn.Sequential(
            nn.Conv2d(image_channels, hidden_dim // 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim),
        )

    def _pool(self, x: torch.Tensor) -> torch.Tensor:
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return torch.flatten(x, 1)

    def forward(self, obs_img: torch.Tensor, goal_img: torch.Tensor) -> torch.Tensor:
        obs_feat = self._pool(self.obs_encoder(obs_img))
        goal_feat = self._pool(self.goal_encoder(goal_img))
        return self.fusion(torch.cat([obs_feat, goal_feat], dim=1))
```

注意点：

- `output_dim` 必须暴露给模型 builder，因为 `waypoint_head` 需要知道输入维度。
- 这里不写 loss、不写 optimizer、不写训练步骤。
- `obs_img` 仍然使用当前数据协议的 `[B, 3*(context_size+1), H, W]`。

### 4.2 注册视觉编码器

修改：

```text
training_base/modules/vision/__init__.py
```

新增 builder：

```python
def build_paper_nav_encoder(config, data_config):
    from training_base.modules.vision.paper_nav_encoder import PaperNavEncoder

    return PaperNavEncoder(
        context_size=data_config["context_size"],
        hidden_dim=int(config.get("hidden_dim", 256)),
        output_dim=int(config.get("output_dim", 512)),
    )
```

新增注册：

```python
module_registry.register("paper_nav_encoder")(build_paper_nav_encoder)
```

如果维护 `__all__`，也把 `build_paper_nav_encoder` 加进去。

### 4.3 新增模型 wrapper

新增文件：

```text
training_base/models/paper_nav.py
```

示例代码：

```python
from typing import Optional, Tuple

import torch

from training_base.models.base import BaseModel


class PaperNav(BaseModel):
    def __init__(
        self,
        *,
        context_size: int,
        len_traj_pred: Optional[int],
        learn_angle: bool,
        encoder,
        head,
    ) -> None:
        super().__init__(
            context_size=context_size,
            len_traj_pred=len_traj_pred,
            learn_angle=learn_angle,
        )
        if encoder is None or head is None:
            raise ValueError("PaperNav requires prebuilt encoder and head modules.")
        self.encoder = encoder
        self.head = head

    def forward(self, obs_img: torch.Tensor, goal_img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(obs_img, goal_img)
        return self.head(features)
```

这里和 `GNM/ViNT` 风格一致：模型只负责把 encoder 和 head 串起来。

### 4.4 注册模型 builder

修改：

```text
training_base/models/__init__.py
```

新增：

```python
@model_registry.register("paper_nav")
def build_paper_nav(config) -> ModelBuild:
    from training_base.models.paper_nav import PaperNav

    data = config["data"]
    model_config = config["model"]
    encoder_config = dict(model_config.get("encoder", {}))
    encoder_name = encoder_config.get("name", "paper_nav_encoder")

    encoder = module_registry.build(encoder_name, encoder_config, data)
    head = module_registry.build(
        model_config.get("head", {}).get("name", "waypoint_head"),
        {
            "input_dim": encoder.output_dim,
            "len_traj_pred": data["len_traj_pred"],
            "num_action_params": 4 if data["learn_angle"] else 2,
            "learn_angle": data["learn_angle"],
        },
    )
    model = PaperNav(
        context_size=data["context_size"],
        len_traj_pred=data["len_traj_pred"],
        learn_angle=data["learn_angle"],
        encoder=encoder,
        head=head,
    )
    return ModelBuild(model=model, extras={})
```

注意点：

- `encoder.output_dim` 是 encoder 和 head 之间的契约。
- `data["learn_angle"]` 决定动作输出维度是 2 还是 4。
- `extras={}` 表示没有额外的非模型状态。像 NoMaD 的 noise scheduler 才需要放 extras。

### 4.5 新增算法

如果新论文仍是监督式航点预测，算法可以直接复用 `SupervisedWaypointAlgorithm`。

新增文件：

```text
training_base/algorithms/paper_nav.py
```

示例代码：

```python
from training_base.algorithms.supervised_waypoint import SupervisedWaypointAlgorithm
from training_base.registry import algorithm_registry


@algorithm_registry.register("paper_nav")
class PaperNavAlgorithm(SupervisedWaypointAlgorithm):
    name = "paper_nav"
```

然后修改：

```text
training_base/algorithms/__init__.py
```

加入导入：

```python
from training_base.algorithms.paper_nav import PaperNavAlgorithm
```

如果维护 `__all__`，也加入：

```python
"PaperNavAlgorithm"
```

### 4.6 新增配置文件

新增：

```text
training_base/configs/paper_nav.yaml
```

示例：

```yaml
# PaperNav experiment config
# 该配置继承 defaults.yaml，只覆盖新论文需要变化的字段。

algorithm:
  # 对应 training_base.algorithms.paper_nav.PaperNavAlgorithm
  name: paper_nav

runtime:
  # 日志目录：logs/<project_name>/<run_name>_<timestamp>
  project_name: visual-nav-paper-nav
  run_name: paper_nav
  # DDP 推荐保持开启；单卡 python 启动时 setup_runtime 会按实际环境处理
  distributed: true
  require_ddp_for_multigpu: true
  # 按显存调 batch。DDP 时建议用 global_batch_size 表示总 batch
  global_batch_size: 256
  eval_batch_size: 256
  epochs: 30
  gpu_ids: [0]
  num_workers: 8
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2
  # 多卡训练前建议先单进程构建 LMDB
  require_lmdb_ready_for_ddp: true
  lmdb_cache_mode: auto
  # 根据显卡选择是否开 AMP
  amp: false
  amp_dtype: fp16

data:
  module_name: navigation
  normalize: true
  # 新数据集元信息默认从 training_base/data/data_config.yaml 读取；
  # 如果新论文有独立数据配置，改成对应 YAML 路径。
  data_config_path: null
  context_type: temporal
  context_size: 5
  image_size: [85, 64]
  image_aspect_ratio: 1.3333333333333333
  goal_sampling:
    negative:
      enabled: true
      policy: offset_zero
      distance_label: max_dist_cat
  goal_type: image
  len_traj_pred: 5
  learn_angle: true
  distance:
    min_dist_cat: 0
    max_dist_cat: 20
  action:
    min_dist_cat: 2
    max_dist_cat: 10
  datasets:
    # 按你的数据集实际路径填写。字段含义和 gnm.yaml / nomad_retrain.yaml 一致。
    recon:
      data_folder: /path/to/recon
      train: /path/to/recon/train
      test: /path/to/recon/test
      waypoint_spacing: 1
      negative_mining: true
      goals_per_obs: 1
      end_slack: 0

model:
  # 对应 training_base.models.__init__.build_paper_nav
  name: paper_nav
  encoder:
    # 对应 module_registry.register("paper_nav_encoder")
    name: paper_nav_encoder
    hidden_dim: 256
    output_dim: 512
  head:
    # 复用 GNM/ViNT 的监督航点预测头
    name: waypoint_head

objective:
  # 复用监督式距离 + 航点损失
  name: supervised_waypoint
  alpha: 0.5
  losses:
    distance: mse
    action: mse

optimizer:
  name: adamw
  lr: 1e-4
  gradient_clip:
    enabled: true
    max_norm: 1.0

scheduler:
  name: cosine
  warmup:
    enabled: true
    epochs: 1

metrics:
  train:
    - name: waypoint_mse
      log_name: waypoint_mse
  eval:
    - name: waypoint_mse
      log_name: waypoint_mse
    - name: waypoint_cosine
      log_name: waypoint_cosine
  heavy: []

logging:
  step:
    by_global_step: true
    first_step: false
  train:
    metrics:
      freq: 100
      unit: step
    behavior:
      freq: 1000
      unit: step
      start_step: 1000
    optim:
      freq: 100
      unit: step
    param_norm:
      freq: 0
      unit: step
  eval:
    schedule:
      freq: 1
      unit: epoch
      fraction: 0.25
    behavior:
      freq: 1
      unit: eval
  media:
    train:
      freq: 1000
      unit: step
      start_step: 1000
    eval:
      trigger: eval
      freq: 1
      unit: eval
      policy: last_batch_per_eval
  runtime:
    perf:
      freq: 50
      unit: step
  system:
    gpu:
      enabled: true
      freq: 50
      unit: step
    ddp:
      log_once: true
  sinks:
    - name: console

visualization:
  num_images_log: 8
  train:
    - name: supervised_waypoint
  eval:
    - name: supervised_waypoint

callbacks:
  - name: checkpoint
    save_latest_every_epoch: true
    checkpoint_freq: 5
  - name: perf_monitor
  - name: optim_monitor
```

### 新数据集元信息和 action stats

`data.datasets.<name>` 只描述数据路径和采样策略。数据集的 `metric_waypoint_spacing`、`camera_metrics` 以及 NoMaD 使用的 `action_stats` 默认放在 `training_base/data/data_config.yaml`。如果你不想改默认文件，可以新增一个 YAML，并在实验配置中写：

`negative_mining` 是旧配置兼容字段；新的负样本语义放在 `data.goal_sampling.negative`。默认策略仍是 `offset_zero`：随机 offset 为 0 时采样跨轨迹目标，负样本距离标签默认写成 `max_dist_cat`。如果新论文完全不需要负样本，把 `enabled` 设为 `false`。

```yaml
data:
  data_config_path: /path/to/my_data_config.yaml

objective:
  # NoMaD 可选：显式覆盖动作归一化统计；优先级高于 data_config_path
  action_stats:
    min: [-2.5, -4.0]
    max: [5.0, 4.0]
```

优先级是：`objective.action_stats` 显式配置最高，其次是 `data.data_config_path`，最后才是默认的 `training_base/data/data_config.yaml`。可视化只消费 `NavigationDataModule` 注入的 `data.dataset_metadata`，不要在 visualizer 里重新读取全局数据配置。

### 4.7 运行命令

单进程构建 LMDB：

```powershell
python -m training_base.cli -c training_base/configs/paper_nav.yaml --build-lmdb-only
```

单卡训练：

```powershell
python -m training_base.cli -c training_base/configs/paper_nav.yaml
```

两卡 DDP 训练：

```powershell
torchrun --standalone --nproc_per_node=2 -m training_base.cli -c training_base/configs/paper_nav.yaml
```

## 5. 如果新论文不是监督式航点预测

如果新论文像 NoMaD 一样训练流程明显不同，例如：

- diffusion denoising。
- contrastive goal representation。
- teacher-student distillation。
- 多个模型交替优化。
- 每一步需要特殊采样策略。
- 评估时必须使用 EMA 或 teacher model。

这时不要继承 `SupervisedWaypointAlgorithm`，而是新建完整 algorithm。

最小骨架如下：

```python
from dataclasses import dataclass
from typing import Optional

from training_base.algorithms.base import Algorithm, StepResult
from training_base.data.batch import split_and_transform_obs, transform_goal
from training_base.models import build_model
from training_base.registry import algorithm_registry, objective_registry


@dataclass
class PaperNavState:
    objective: object
    teacher_model: Optional[object] = None


@algorithm_registry.register("paper_nav_v2")
class PaperNavV2Algorithm(Algorithm):
    name = "paper_nav_v2"

    def build_model(self, config):
        result = build_model(config)
        return result.model, result.extras

    def build_objective(self, config):
        return objective_registry.build(config["objective"]["name"], config["objective"])

    def create_state(self, model, model_extras, objective, config, device, resume_state):
        return PaperNavState(objective=objective)

    def prepare_batch(self, batch, transform, device, mode: str, should_log_images: bool):
        return {
            "obs_image": split_and_transform_obs(batch.obs_image, transform, device),
            "goal_image": transform_goal(batch.goal_image, transform, device),
            "actions": batch.actions.to(device, non_blocking=True),
            "distance": batch.distance.to(device, non_blocking=True),
            "action_mask": batch.action_mask.to(device, non_blocking=True),
        }

    def train_step(self, model, prepared, state: PaperNavState, config):
        outputs = model(prepared["obs_image"], prepared["goal_image"])
        losses = state.objective(outputs=outputs, batch=prepared, mode="train")
        return StepResult(
            loss=losses["total_loss"],
            logs=losses,
            batch_size=int(prepared["obs_image"].shape[0]),
            extras={"outputs": outputs},
        )

    def eval_step(self, model, prepared, state: PaperNavState, config):
        outputs = model(prepared["obs_image"], prepared["goal_image"])
        losses = state.objective(outputs=outputs, batch=prepared, mode="eval")
        return StepResult(
            loss=None,
            logs=losses,
            batch_size=int(prepared["obs_image"].shape[0]),
            extras={"outputs": outputs},
        )
```

这类 algorithm 可以继续复用 `Trainer` 的 DDP、AMP、日志、checkpoint、callback。

## 6. 新增 objective 示例

假设新论文需要 `distance + action + smoothness` 三项损失，可以新增：

```text
training_base/objectives/paper_nav_objective.py
```

示例：

```python
import torch

from training_base.losses import action_reduce, get_configured_loss
from training_base.registry import objective_registry


@objective_registry.register("paper_nav_objective")
class PaperNavObjective:
    def __init__(self, config) -> None:
        self.distance_weight = float(config.get("distance_weight", 0.01))
        self.action_weight = float(config.get("action_weight", 1.0))
        self.smoothness_weight = float(config.get("smoothness_weight", 0.1))
        losses = config.get("losses", {})
        self.distance_loss = get_configured_loss(losses, "distance", "mse")
        self.action_loss = get_configured_loss(losses, "action", "mse")

    def __call__(self, *, dist_pred, action_pred, dist_label, action_label, action_mask):
        dist_loss = self.distance_loss(dist_pred.squeeze(-1), dist_label.float())
        action_loss = action_reduce(
            self.action_loss(action_pred, action_label, reduction="none"),
            action_mask,
        )
        delta = action_pred[:, 1:, :2] - action_pred[:, :-1, :2]
        smoothness_loss = torch.mean(delta.pow(2))
        total_loss = (
            self.distance_weight * dist_loss
            + self.action_weight * action_loss
            + self.smoothness_weight * smoothness_loss
        )
        return {
            "total_loss": total_loss,
            "dist_loss": dist_loss,
            "action_loss": action_loss,
            "smoothness_loss": smoothness_loss,
        }
```

然后修改：

```text
training_base/objectives/__init__.py
```

加入：

```python
from training_base.objectives.paper_nav_objective import PaperNavObjective
```

配置中使用：

```yaml
objective:
  name: paper_nav_objective
  distance_weight: 0.01
  action_weight: 1.0
  smoothness_weight: 0.1
  losses:
    distance: mse
    action: mse
```

## 7. 新增 metric 和 visualizer 的时机

轻量 metric 适合放在 `metrics.train/eval`，它们应该复用当前 step 已经算出的预测，不额外做昂贵采样。

示例：

```python
from training_base.registry import metric_registry


@metric_registry.register("final_waypoint_l2")
def final_waypoint_l2(pred, target, mask):
    ...
```

昂贵 metric 放在 `metrics.heavy`，例如需要 diffusion reverse process、多次采样或完整行为模拟。`Trainer` 会用 `logging.train.behavior.freq` 控制频率。

可视化器适合只做“把 algorithm 已经准备好的张量画出来”。不要在 visualizer 里重新读 YAML、重新构建 Dataset 或重新 forward 模型。

## 8. 最容易漏的注册点

新增文件后，检查这些位置：

```text
training_base/algorithms/__init__.py
training_base/models/__init__.py
training_base/modules/__init__.py
training_base/modules/vision/__init__.py
training_base/modules/heads/__init__.py
training_base/objectives/__init__.py
training_base/metrics/__init__.py
training_base/visualizers/__init__.py
training_base/losses/__init__.py
```

如果 registry 报错：

```text
KeyError: paper_nav is not registered in ...
```

优先检查：

1. YAML 里的 `name` 和注册 key 是否一致。
2. 新文件是否被对应 `__init__.py` 导入。
3. `register_builtins()` 是否在训练入口执行。

## 9. 什么时候才应该改 `trainer.py`

默认不要改。

只有下面这种所有论文共享的训练基础设施，才考虑改 `trainer.py`：

- 通用梯度累积。
- 通用多优化器调度接口。
- 通用 callback hook。
- 通用 AMP 策略。
- 通用分布式指标聚合。
- 通用日志频率语义。

如果只是某一篇论文需要：

- 特殊 goal mask。
- 特殊采样。
- 特殊 EMA。
- 特殊 teacher model。
- 特殊 train/eval loss。
- 特殊 visualization。

这些都应该放进该论文的 `Algorithm`、`Objective`、`Metric` 或 `Visualizer`。

## 10. 新增后的检查清单

代码写完后按这个顺序查：

```powershell
python -m compileall -q training_base
python -m training_base.cli -c training_base/configs/paper_nav.yaml --build-lmdb-only
python -m training_base.cli -c training_base/configs/paper_nav.yaml
torchrun --standalone --nproc_per_node=2 -m training_base.cli -c training_base/configs/paper_nav.yaml
```

同时做人工检查：

- `algorithm.name`、`model.name`、`objective.name` 是否都能在 registry 找到。
- `models/__init__.py` 是否返回 `ModelBuild(model=..., extras=...)`。
- 新模型 forward 输出是否和 objective 期望一致。
- `prepare_batch()` 里 GPU 张量是否 `.to(device, non_blocking=True)`。
- 轻量指标是否没有额外昂贵 forward。
- 重指标是否放到 `metrics.heavy` 并配置低频。
- visualizer 是否只消费 algorithm 传入的数据。
- 多卡训练前是否已经单进程构建 LMDB。
- 没有把论文名分支写进 `trainer.py` 或 `cli.py`。

## 11. 推荐最小改动策略

新增论文时，优先按下面顺序复用：

1. 能复用 `NavigationDataset` 就不要改数据层。
2. 能复用 `NavigationBatch` 就不要扩展 batch 字段。
3. 能复用 `waypoint_head` 就不要新写 head。
4. 能复用 `supervised_waypoint` objective 就不要新写 objective。
5. 能继承 `SupervisedWaypointAlgorithm` 就不要完整重写 algorithm。
6. 能通过 YAML 配置切换就不要写 Python 分支。
7. 除非是通用训练能力，否则不要改 `Trainer`。

这样做的好处是：新论文只像插件一样接进来，GNM、ViNT、NoMaD 的老路径不会被新实验扰动。
