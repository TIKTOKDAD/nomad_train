# Extension Contract

This file defines the minimum interfaces for extending `training_base` without editing the trainer loop.

## Algorithm

Register an `Algorithm` under `algorithm_registry`. An algorithm owns paper-specific behavior:

- `build_model(config)` returns `(nn.Module, extras)`.
- `build_objective(config)` returns the loss/objective object.
- `prepare_batch(...)` moves tensors and creates the model input contract.
- `train_step(...)` and `eval_step(...)` return `StepResult`.
- Optional hooks: `light_metrics`, `heavy_metrics`, `visualize`, `state_dict`, `prepare_resume`, `step_scheduler`.

Use `state_dict` for algorithm-owned state such as EMA weights. Generic model, optimizer, scheduler, callback, RNG, and config state are saved by framework checkpointing.

Algorithms should use the shared resume helper unless they have extra auxiliary
state. New schema checkpoints resume strictly by default; legacy weight remaps
must be explicitly enabled in runtime config.

## Model and Modules

Register reusable blocks under `module_registry` and full assemblies under `model_registry`.

- Modules should be reusable neural-network components.
- Model builders should only wire modules together and return a `ModelBuild`.
- Avoid reading global files or training state from a model builder.

## Objective, Loss, Metric, Visualizer

- `objective_registry`: paper-level loss recipes.
- `loss_registry`: reusable primitive losses.
- `metric_registry`: cheap or explicitly scheduled metric functions.
- `visualizer_registry`: plotting/media helpers that consume prepared tensors and metadata.

Heavy metrics and visualizers must not run implicitly every step; they should follow logging schedules.

## Callback and Logger

Callbacks are shared lifecycle infrastructure. Implement only the hooks you need, such as `on_epoch_end`, `log_perf`, or `close`.

Stateful callbacks should implement:

```python
def state_dict(self) -> dict: ...
def load_state_dict(self, state: dict) -> None: ...
```

Logger sinks should implement `log_metrics`, `log_images`, and `close`. Optional `image(path)` may wrap local image paths for a backend such as W&B.

## Data

The current navigation data contract is image-based and temporal:

- observation: concatenated context RGB images
- goal: RGB image
- labels: action trajectory, distance, goal position, action mask, dataset index, metric scale

New modalities or non-temporal sampling should be implemented through a registered dataset/data-module adapter rather than by growing `NavigationDataset` with more mode-specific branches. The default selection is `data.module_name: navigation`.

Deterministic sample context is framework-owned: `data/sampling.py` passes `(seed, epoch, index)` to `data/labeling.sample_context`, and labeling code derives stochastic goals from that context. New datasets should either reuse that path or provide an adapter with the same deterministic contract.

Negative goal sampling is configured through `data.goal_sampling.negative`.
Dataset-level `negative_mining` is accepted for old configs, but new configs
should use the centralized goal sampling fields.
