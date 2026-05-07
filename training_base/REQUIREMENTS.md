# Training Base Requirements

本文档记录 `training_base` 的目标、边界、目录职责和后续新增网络的标准流程。它用于让后续接手的人明确：这个目录要做成什么样，以及不应该被改成什么样。

## 1. 总体目标

`training_base` 是视觉导航领域内的训练基座，服务 GNM、ViNT、NoMaD 以及后续同领域论文实现。它不是一个面向所有机器学习任务的通用训练框架。

目标是把视觉导航训练中的公共能力稳定沉淀下来：

- 高效 DDP 训练。
- 安全 LMDB 数据缓存构建和读取。
- 统一 DataLoader、batch、logging、checkpoint、resume、evaluation 流程。
- 可复用的神经网络模块、loss、optimizer、LR scheduler、diffusion/noise scheduler。
- 训练基座层复用 W&B、console logger、checkpoint、callback、metric、visualization 等服务能力。
- 面向论文复现和新网络扩展的 `algorithm + model + objective + config` 插件化结构。

新增同领域网络时，优先新增独立插件和配置，不要继续把分支堆进训练主循环。

## 2. 领域边界

这个基座只需要覆盖视觉导航论文和相关模型族：

- GNM。
- ViNT。
- NoMaD。
- 后续视觉导航领域的新论文、新 backbone、新训练 objective、新评估指标和新可视化。

不要把它扩展成多领域训练框架。比如 NLP、检测、分割、强化学习等完全不同任务不应该塞进这个基座。需要扩展时，也应先判断是否仍属于视觉导航训练流程。

## 3. 必须清理或避免的旧风格

主路径中不应该再出现这些旧式设计：

- `domains/visual_nav` 这类多领域外壳。
- `training_base.visual_nav` 这类过渡兼容层。
- `task_name`、`model_type`、`func_name` 驱动的大分支。
- `TASKS`、`MODELS`、`LOSSES`、`METRICS` 这类旧全局字典。
- 在 `trainer.py` 或 CLI 里写 `if gnm / elif vint / elif nomad`。
- NoMaD 里用 `forward(func_name="vision_encoder")` 这种字符串调度子模块的方式。
- 多卡训练退回 `DataParallel`。
- DDP 训练时多个 rank 同时构建 LMDB。
- 把网络模块、loss、optimizer、scheduler、metric、visualization、logging、callback 全部塞进同一个泛化容器。

允许存在论文算法内部必要的逻辑分支，例如 NoMaD 的 goal mask、EMA、diffusion train/eval loss。判断标准是：这些逻辑是否属于论文训练配方本身。如果属于，应该放在 `algorithms/` 或 `objectives/`，而不是放进通用 `trainer.py`。

## 4. 设计原则

### 4.1 视觉导航领域内插件化

插件化不是为了把项目抽象成所有任务通用，而是为了让视觉导航领域内的新论文更容易接入。

新增论文时，优先新增：

- `models/<new_model>.py`
- `algorithms/<new_algorithm>.py`
- `objectives/<new_objective>.py`
- 必要的 `modules/...`
- 必要的 `losses/...`
- 必要的 `metrics/...`
- 必要的 `visualizers/...`
- `configs/<new_config>.yaml`

不要优先修改 `trainer.py`。

### 4.2 Config + Registry

所有应该可替换的地方尽量通过 YAML 配置选择，再通过 registry 构建。

例如模型不应该在主流程中这样选择：

```python
if model_type == "gnm":
    model = GNM(...)
elif model_type == "vint":
    model = ViNT(...)
elif model_type == "nomad":
    model = NoMaD(...)
```

而应该由配置声明：

```yaml
model:
  name: nomad
```

再由 registry 构建：

```python
model = model_registry.build(config["model"]["name"], config)
```

这种方式适用于：

- algorithm
- model
- objective
- loss primitive
- metric
- visualizer
- callback
- logging sink
- optimizer
- LR scheduler
- diffusion/noise scheduler
- reusable neural-network module

但不要为了形式把论文内部的每一行逻辑都拆成 registry。比如 NoMaD 的 diffusion loss 计算、goal mask 策略、EMA 更新属于算法配方，可以保留在 `algorithms/nomad.py` 和 `objectives/nomad_diffusion.py`。

### 4.3 Trainer 不认识具体论文

`trainer.py` 只负责高性能训练基础设施：

- epoch/batch loop
- DDP
- AMP
- GradScaler
- logging cadence
- evaluation cadence
- callback 调用
- checkpoint callback 调用
- distributed metric reduce

它不应该知道 GNM、ViNT、NoMaD 的具体细节。

### 4.4 DDP 和 LMDB 是一等能力

多卡训练必须以 `torchrun + DistributedDataParallel` 为标准路径。单进程多 GPU 的 `DataParallel` fallback 不应该作为训练路径保留。

LMDB 工作流必须安全：

1. 单进程构建或修复 LMDB。
2. 写入阶段使用完整性标记或等价校验，避免中断后留下半成品。
3. DDP 训练前检查所有 LMDB 缓存是否完整。
4. DDP 训练时只读已完成的 LMDB。
5. 不允许多个 rank 同时构建同一个缓存。

## 5. 目录职责

### `cli.py`

命令行入口。

职责：

- 解析配置路径和命令行参数。
- 加载 defaults 和用户 YAML。
- 注册内置插件。
- 处理 `--build-lmdb-only` 和 `--rebuild-incomplete-lmdb`。
- 初始化 runtime。
- 创建 data module、algorithm、trainer。
- 启动训练或只构建 LMDB 后退出。

不应该放论文训练细节。

### `trainer.py`

通用训练器。

职责：

- 管理 train/eval loop。
- 调用 algorithm 的统一接口。
- 处理 AMP、GradScaler、DDP barrier、distributed metric reduce。
- 控制 logging、heavy metric、image visualization 的频率。
- 调用 callbacks 做 checkpoint、perf monitor 等。

不应该出现具体模型名或论文名分支。

### `registry.py`

全局注册表入口。

职责：

- 定义 `algorithm_registry`、`model_registry`、`objective_registry`、`metric_registry`、`visualizer_registry`、`callback_registry`、`log_sink_registry`、`module_registry`、`loss_registry`、`optimizer_registry`、`scheduler_registry`、`noise_scheduler_registry`。
- 通过 `register_builtins()` 导入内置插件，使装饰器注册生效。

registry 是插件系统的一部分，不是旧代码。

### `core/`

底层基础设施。

职责：

- 配置加载、合并、校验。
- DDP runtime 初始化。
- CUDA device 和 `CUDA_VISIBLE_DEVICES` 处理。
- DDP model wrapping。
- checkpoint 保存和加载。
- AMP、GradScaler、barrier、rank0 helper。
- DataLoader helper，例如 `pin_memory`、`prefetch_factor`、`persistent_workers`、workers per rank。

这里应该保持论文无关。

### `configs/`

训练配置。

职责：

- 选择本次训练使用的 algorithm、model、objective。
- 配置 data、dataset splits、LMDB、DDP、batch size、workers。
- 配置 optimizer、scheduler、AMP、logging、callbacks、metrics、visualization。
- 为 GNM、ViNT、NoMaD 和后续新论文提供独立 YAML。

配置文件应该可读、分组清楚，不要把不同功能的参数混在一起。

### `data/`

视觉导航数据层。

职责：

- `NavigationDataset`。
- LMDB 构建、校验、只读打开。
- DDP 前 LMDB preflight。
- `NavigationDataModule`。
- train/test DataLoader 构建。
- `DistributedSampler`。
- batch/collate 协议。
- 图像 transform 和 batch 到 device 的 non-blocking 传输。
- 数据集元信息和动作归一化配置。

这里可以包含视觉导航数据协议，但不应该包含某篇论文的训练循环，也不应该包含离线统计脚本。离线统计 `metric_waypoint_spacing` 和 `action_stats` 的维护工具统一放到 `training_base/utils/`，例如 `training_base/utils/data_stats.py`。

### `modules/`

神经网络积木库。

职责：

- `modules/vision/`: vision encoder，例如 masked ViNT、ViT、MobileNet encoder。
- `modules/transformer/`: transformer decoder、positional encoding 等。
- `modules/diffusion/`: diffusion neural network，例如 ConditionalUnet1D builder。
- `modules/heads/`: prediction head，例如 distance MLP。

这里只放 `torch.nn.Module` 或直接构建网络模块的 builder。DDPM 这类 noise scheduler 不放这里。

### `losses/`

Loss primitive 和 reduction。

职责：

- MSE、cross entropy、cosine similarity 等 primitive。
- masked/action reduction。
- 从 objective 配置中解析 loss primitive。

复杂 loss recipe 放 `objectives/`，不要放这里。

### `optimizers/`

Optimizer builder。

职责：

- Adam、AdamW、SGD 等 optimizer builder。
- optimizer builder 只负责创建 optimizer；gradient clipping 在 trainer backward 之后、optimizer step 之前执行。

不负责 scheduler step，不负责训练循环。

### `schedulers/`

LR scheduler builder。

职责：

- cosine、cyclic、plateau、warmup wrapper 等学习率调度器。
- 只服务 optimizer 学习率调度。

不放 diffusion/noise scheduler。

### `noise_schedulers/`

Diffusion/noise scheduler builder。

职责：

- DDPM scheduler。
- 后续 diffusion policy 相关的噪声过程 scheduler。

这里服务 NoMaD 这类 diffusion objective，不是 optimizer LR scheduler。

### `models/`

网络结构和模型组装。

职责：

- 定义 GNM、ViNT、NoMaD 或后续新模型的网络结构。
- 组合 reusable modules。
- 暴露清晰方法，例如 NoMaD 的 `encode_vision()`、`predict_noise()`、`predict_distance()`。
- 注册 model builder。

`models/` 回答“网络长什么样”。它不负责 epoch loop、optimizer step、logging、checkpoint。

### `algorithms/`

论文训练配方。

职责：

- 定义某篇论文怎么训练。
- 构建模型和 objective。
- 准备 batch。
- 实现 `train_step()` 和 `eval_step()`。
- 接入 EMA、resume extra state、heavy metrics、visualizer。
- 定义 scheduler step 语义。

`algorithms/` 回答“这篇论文怎么训练”。例如 NoMaD 的 diffusion 训练、goal mask、EMA 更新都属于这里和 `objectives/` 的职责。

### `objectives/`

Loss recipe。

职责：

- 组合 loss primitives。
- 实现 supervised waypoint loss。
- 实现 NoMaD diffusion loss。
- 从 `objective.losses` 配置中选择具体 loss primitive。
- 从 `objective.action_stats` 可选配置或默认 `data/data_config.yaml` 读取 NoMaD action normalization 统计。

`objectives/` 不应该处理 DataLoader、DDP、optimizer step 或完整训练循环。

### `metrics/`

领域级指标。

职责：

- 存放可复用 metric primitives，例如 waypoint MSE、waypoint cosine、flattened waypoint cosine。
- 存放高层导航指标。
- 存放 NoMaD 这类昂贵 behavior metrics。
- 使用 `metric_registry` 注册可配置 metric。

简单的 loss 标量可以在 objective 返回；更昂贵、需要额外采样或推理的指标应该放这里。

### `visualizers/`

可视化 helper 和论文相关可视化适配器。

职责：

- 通用绘图和图像输出 helper。
- 根据 algorithm 准备好的数据和模型输出生成可视化。
- 通过 `visualizer_registry` 注册。
- 从 `visualization.train` 和 `visualization.eval` 配置选择。

### `callbacks/`

训练生命周期 hook。

职责：

- checkpoint callback。
- performance monitor callback。
- 未来所有 algorithm 共享的 trainer hook。

callback 属于 trainer orchestration，不属于模型模块。某篇论文专属的 EMA 更新语义仍应保留在 `algorithms/` 或该 algorithm 的 state 中，例如 NoMaD 的 `algorithm.ema`。

### `loggers/`

训练日志基础设施。

职责：

- W&B sink。
- console sink。
- metric recorder。
- metric store / moving average。
- rank0-only logging 行为。

日志是训练基座服务能力，不是模型模块。

## 6. 后续实现新网络结构的完整流程

### Step 1: 判断变化类型

先判断新论文或新网络改动属于哪一类：

- 只是换视觉 backbone：优先新增 `modules/vision/<encoder>.py`。
- 只是换 prediction head：优先新增 `modules/heads/<head>.py`。
- 新增 diffusion neural network 或 transformer block：放到 `modules/diffusion/` 或 `modules/transformer/`。
- 新增 diffusion/noise scheduler：放到 `noise_schedulers/`。
- 新模型只是新模块组合：新增 `models/<model>.py` 和 model builder。
- loss primitive 变化：新增 `losses/<loss>.py` 或扩展 `losses/primitives.py`。
- loss recipe 变化：新增 `objectives/<objective>.py`。
- batch 处理、EMA、train/eval step、metric 或 visualization 流程变化：新增 `algorithms/<algorithm>.py`。

### Step 2: 新增可复用网络模块

如果新网络引入可复用模块，放在 `modules/` 下，并注册到 `module_registry`。

示例：

```python
from training_base.registry import module_registry


def build_new_encoder(config, data_config):
    return NewEncoder(
        image_size=data_config["image_size"],
        hidden_dim=config["hidden_dim"],
    )


module_registry.register("new_encoder")(build_new_encoder)
```

### Step 3: 新增或更新 model

如果是完整新模型，在 `models/<new_model>.py` 定义网络结构。

要求：

- 只写网络结构和 forward/submodule 方法。
- 不写 optimizer step。
- 不写 epoch loop。
- 不直接处理 W&B。
- 不写 DDP 逻辑。

在 `models/__init__.py` 注册 builder：

```python
@model_registry.register("new_model")
def build_new_model(config):
    model_config = config["model"]
    encoder = module_registry.build(
        model_config["encoder"]["name"],
        model_config["encoder"],
        config["data"],
    )
    model = NewModel(encoder=encoder)
    return ModelBuild(model=model, extras={})
```

### Step 4: 新增 objective

如果只是换 loss primitive，优先复用 `losses/`。

如果损失组合方式不同，新增 `objectives/<new_objective>.py`：

```python
@objective_registry.register("new_objective")
class NewObjective:
    def __init__(self, config):
        losses = config.get("losses", {})
        self.action_loss = get_configured_loss(losses, "action", "mse")

    def __call__(self, ...):
        ...
```

### Step 5: 新增 algorithm

如果新论文训练逻辑和现有 GNM/ViNT/NoMaD 不同，新增 `algorithms/<new_algorithm>.py`。

至少实现：

- `build_model()`
- `build_objective()`
- `prepare_batch()`
- `train_step()`
- `eval_step()`

按需实现：

- `create_state()`
- `prepare_resume()`
- `model_for_eval()`
- `after_optimizer_step()`
- `heavy_metrics()`
- `visualize()`
- `state_dict()`
- `step_scheduler()`

`Trainer` 会调用这些统一接口，不应该为了新论文改训练主循环。

### Step 6: 新增 metrics 和 visualizers

普通标量 loss 直接由 objective 或 algorithm 返回即可。

如果指标需要额外采样、额外 forward、diffusion reverse process 或昂贵计算，放到 `metrics/` 并注册：

```python
@metric_registry.register("new_behavior_metric")
def compute_new_behavior_metric(...):
    ...
```

如果需要图像或轨迹可视化，放到 `visualizers/` 并注册：

```python
@visualizer_registry.register("new_visualizer")
class NewVisualizer:
    ...
```

### Step 7: 新增配置

新增 `configs/<new_model_or_paper>.yaml`。

最少要声明：

```yaml
algorithm:
  name: new_algorithm

model:
  name: new_model

objective:
  name: new_objective

optimizer:
  name: adamw
  lr: 1e-4

scheduler:
  name: cosine

metrics:
  heavy: []

visualization:
  train: []
  eval: []
```

如果使用 LMDB/DDP，要确认 runtime 部分包含：

```yaml
runtime:
  distributed: true
  require_ddp_for_multigpu: true
  require_lmdb_ready_for_ddp: true
  build_lmdb_only: false
  lmdb_cache_mode: auto
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2
```

### Step 8: 验收检查

新增或修改完成后，至少检查：

1. Python AST 通过。
2. 所有 YAML 都能加载并合并。
3. 新 registry 名字能注册成功。
4. 旧关键词没有回流到主路径：`model_type`、`task_name`、`func_name`、`DataParallel`、`training_base.domains`。
5. 单卡 smoke 能跑到一个 batch。
6. `--build-lmdb-only` 能构建或验证 LMDB。
7. 小规模 `torchrun` DDP smoke 能启动。
8. GNM/ViNT/NoMaD 原有配置没有被新改动破坏。

## 7. 判断是否应该修改 `trainer.py`

默认不要修改 `trainer.py`。

只有当新需求是所有论文都共享的训练基础设施时，才考虑修改：

- 新的 AMP 策略。
- 新的 distributed reduce 机制。
- 新的通用 checkpoint callback hook。
- 新的通用 logging cadence。
- 新的通用 gradient accumulation。

如果只是某篇论文的特殊训练步骤，应该放在 `algorithms/` 或 `objectives/`。

## 8. 最终验收标准

`training_base` 达标时应满足：

- 主架构是视觉导航领域训练基座，而不是多领域泛化框架。
- 新增同领域模型主要通过插件和配置完成。
- `Trainer` 不包含论文分支。
- GNM、ViNT、NoMaD 能力与旧训练代码对齐。
- NoMaD diffusion、EMA、heavy metrics、visualization 能正常保留。
- 多卡训练走 DDP，不走 DataParallel。
- LMDB 构建和 DDP 读取安全。
- 配置清晰、分组明确、易于审查。
- 静态检查和基础 smoke test 通过。

## 9. 当前收口规则

- GNM/ViNT 的 encoder 和 waypoint head 必须通过 `module_registry` 构建，`models/` 只负责组装 prebuilt modules。
- checkpoint 加载默认报告 `missing_keys` 和 `unexpected_keys`，但不自动迁移旧 checkpoint key。
- gradient clipping 属于 backward 之后、optimizer step 之前的训练基础设施；`mode: norm` 使用 `clip_grad_norm_`，`mode: value` 使用 `clip_grad_value_`。
- `NavigationBatch` 显式携带 `metric_scale`，visualizer 不直接读取 `data/data_config.yaml`。
- `metrics.train/eval` 只承载 cheap light metrics；需要额外 forward、diffusion sampling 或 reverse process 的指标继续放在 `metrics.heavy`。
