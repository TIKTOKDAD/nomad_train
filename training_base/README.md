# Navigation Training Base

`training_base` is the visual-navigation training base for GNM, ViNT, NoMaD, and later papers in the same domain. It is not a generic multi-domain framework.

For the full project requirements, design boundaries, directory responsibilities, and the process for adding new networks, see [REQUIREMENTS.md](REQUIREMENTS.md).

For a concrete Chinese step-by-step guide with a full new-paper example, see [ADDING_NEW_PAPER.md](ADDING_NEW_PAPER.md).

For day-to-day commands, see [QUICKSTART.md](QUICKSTART.md). For extension interfaces, see [docs/extension_contract.md](docs/extension_contract.md).

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
  Deterministic epoch/index-aware sampling lives in `data/sampling.py`; label
  sampling uses `data/labeling.py`.
  Trajectory loading, LMDB image reads, goal sampling, and action label
  construction are separated internally so new datasets can plug in through the
  data-module registry.
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

## Common Commands

Build or repair LMDB caches in one process before distributed training:

```bash
python -m training_base.cli -c training_base/configs/nomad_retrain.yaml --build-lmdb-only
python -m training_base.cli -c training_base/configs/nomad_retrain.yaml --build-lmdb-only --rebuild-incomplete-lmdb
```

Run single-process training:

```bash
python -m training_base.cli -c training_base/configs/nomad_retrain.yaml
```

Run DDP training after LMDB caches are complete:

```bash
torchrun --standalone --nproc_per_node=4 -m training_base.cli -c training_base/configs/nomad_retrain.yaml
```

Resume by setting `runtime.load_run` or `runtime.load_checkpoint_path`.
`runtime.load_run` resolves to `<runtime.log_root>/<load_run>/latest.pth`;
`runtime.load_checkpoint_path` points directly at a checkpoint file. New schema
checkpoints load strictly by default through `runtime.resume_strict: true`.
Legacy GNM/ViNT/NoMaD key remapping is opt-in with
`runtime.allow_legacy_weight_remap: true`. `runtime.epochs` is the target total
epoch count, so a checkpoint saved at epoch 20 with `runtime.epochs: 100`
resumes through epoch 99 rather than adding 100 more epochs.

Set a W&B sink to `enabled: false` to disable W&B. Set `strict: true` only when
logging failures should stop training; by default W&B failures warn and training
continues. For stricter reproducibility, set `runtime.seed` and
`runtime.deterministic: true`; this disables cuDNN benchmark and asks PyTorch to
use deterministic algorithms where available.

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
dataset_registry
data_module_registry
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

The default data module is `data.module_name: navigation`. Register a new data
module under `data_module_registry` when a dataset cannot reasonably adapt to
the navigation batch contract.

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
`algorithm_state`, `callback_state`, `config`, `global_step`,
`eval_summaries`, and `rng_state`.
`latest.pth` and the auxiliary latest files are written through a temp file and
atomic replace; an existing latest file is kept as `*.backup.pth`.

Recovery is epoch-level only; mid-epoch replay is intentionally not promised.
Checkpoints restore Python, NumPy, Torch CPU, and CUDA RNG state when the field
is present. Older checkpoints without `rng_state` still load, but strict replay
after resume is not guaranteed. Callback state is saved with the checkpoint, and
epoch-aware data sampling derives stochastic labels from `seed + epoch + index`
so resumed epochs do not depend on worker-local random streams.

Old GNM, ViNT, and NoMaD checkpoints are read-only compatible where the legacy
module names are known, but remapping is explicit: set
`runtime.allow_legacy_weight_remap: true`. Loading reports missing and
unexpected model keys after the remap attempt, so incompatible weights are
visible instead of being silently ignored. The old layout is not used for new
saves.

Training light metrics are all-reduced before rank0 writes logs in DDP, so W&B
and console values represent the global step rather than only rank0's local
batch. Heavy metrics and media still follow their configured schedules.

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
