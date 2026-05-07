# Navigation Training Base

`training_base` is the visual-navigation training base for GNM, ViNT, NoMaD, and later papers in the same domain. It is not a generic multi-domain framework.

For the full project requirements, design boundaries, directory responsibilities, and the process for adding new networks, see [REQUIREMENTS.md](REQUIREMENTS.md).

## Layers

```text
core/
  Runtime, config, DDP, AMP, checkpoint, and DataLoader helpers.

modules/
  Reusable neural-network modules such as vision encoders, transformer blocks,
  diffusion networks, and prediction heads.

losses/
  Loss primitives and reductions used by objectives.

optimizers/
  Optimizer builders and optimizer-adjacent helpers.

schedulers/
  Learning-rate scheduler builders.

noise_schedulers/
  Diffusion/noise scheduler builders such as DDPM.

data/
  Navigation dataset protocol, LMDB cache handling, and DDP-safe DataLoaders.
  Offline dataset-statistics tools do not live here.

models/
  Network definitions and model assembly. A model says what modules are connected.

objectives/
  Loss recipes built from registered loss primitives.

metrics/
  Cheap metric primitives plus low-frequency expensive behavior metrics.

visualizers/
  Visualization helpers and paper-specific visualization adapters.

callbacks/
  Shared trainer lifecycle hooks such as checkpointing and performance monitoring.

loggers/
  Logging sinks, recorders, and metric stores.

algorithms/
  Paper training recipes. An algorithm wires model, objective, metrics,
  visualization, resume state, EMA, and train/eval steps together.

trainer.py
  Paper-agnostic high-performance DDP/AMP training loop.
```

Offline maintenance utilities live under `utils/`, for example `utils/data_stats.py`.

## Registries

`core/registry.py` defines the small `Registry` class.

`registry.py` owns the global registries:

```text
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

Call `register_builtins()` before building configured objects.

Registered reusable training items include GNM/ViNT encoders, waypoint heads,
NoMaD vision modules, diffusion networks, distance heads, and DDPM noise
schedulers.

## Adding A New Paper

1. Add reusable network modules under `modules/` if the paper introduces new building blocks.
2. Add a model under `models/` if the paper needs a new network assembly.
3. Add or reuse loss primitives under `losses/`.
4. Add an objective under `objectives/` for the paper's loss recipe.
5. Add metric primitives or heavy metrics under `metrics/` when needed.
6. Add visualization helpers or adapters under `visualizers/` if the paper needs images or plots.
7. Add callbacks or loggers only when the behavior is shared training infrastructure.
8. Add an algorithm under `algorithms/` to define batch preparation, train/eval steps, resume state, EMA, metrics, and visualization calls.
9. Add a config under `configs/` selecting `algorithm.name`, `model.name`, `objective.name`, optimizer, scheduler, metrics, callbacks, and visualization.

Paper-specific EMA belongs in `algorithm`, for example `algorithm.ema`, while reusable checkpointing stays in `callbacks`.

If the new paper reuses the visual-navigation batch contract, do not change
`trainer.py`. Extend `NavigationBatch` and the paper's `Algorithm` only when
the input/output protocol genuinely changes.

## Checkpoints

New runs are saved in the `training_base` checkpoint schema with
`checkpoint_schema_version`, `model`, `optimizer`, `scheduler`,
`algorithm_state`, `config`, `global_step`, and `eval_summaries`.

Old GNM, ViNT, and NoMaD checkpoints are read-only compatible where the legacy
module names are known. Loading reports missing and unexpected model keys after
the remap attempt, so incompatible weights are visible instead of being silently
ignored. The old layout is not used for new saves.

## Visualization

Dataset visualization metadata is injected by `NavigationDataModule` through
`data.dataset_metadata`. Visualizers consume that metadata and should not read
global dataset config files directly.

Trajectory plots include the bird's-eye comparison, observation image, goal
image, distance prediction/label, and camera projection overlays when
`camera_metrics` are available. Missing camera metadata or missing OpenCV falls
back to the non-projected observation panel.

NoMaD action visualizations keep goal-conditioned samples, unconditioned samples,
and ground truth visually separated so diffusion behavior remains diagnosable.

## NoMaD Compatibility

`objective.distance_mask_mode` controls the distance loss compatibility mode:
`per_sample` is the default and masks each sample independently, while
`legacy_scalar` exists only for strict reproduction of old `train_nofix` logs.

Gradient clipping is applied after backward and before the optimizer step.
The trainer should not need paper-specific branches.
