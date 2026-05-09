# Quickstart

`training_base` is a visual-navigation training framework for GNM, ViNT, NoMaD, and related papers.

## Build LMDB Caches

Build image caches in a single process before distributed training:

```bash
python -m training_base.cli -c training_base/configs/nomad_retrain.yaml --build-lmdb-only
```

If a previous cache build was interrupted:

```bash
python -m training_base.cli -c training_base/configs/nomad_retrain.yaml --build-lmdb-only --rebuild-incomplete-lmdb
```

## Train

Single process:

```bash
python -m training_base.cli -c training_base/configs/nomad_retrain.yaml
```

DDP:

```bash
torchrun --standalone --nproc_per_node=4 -m training_base.cli -c training_base/configs/nomad_retrain.yaml
```

## Resume

Set `runtime.load_run` to a run under `runtime.log_root`:

```yaml
runtime:
  log_root: logs
  load_run: navigation/nomad_2026_05_09_120000
  epochs: 100
```

Or resume from an explicit file:

```yaml
runtime:
  load_checkpoint_path: logs/navigation/nomad_2026_05_09_120000/latest.pth
  resume_strict: true
```

`runtime.epochs` is the target total epoch count, not the number of extra epochs to append. A checkpoint saved at epoch 20 resumes from epoch 21 and runs until epoch 99 when `epochs: 100`.

The framework supports epoch-level recovery. It saves model, optimizer, scheduler, algorithm state, callback state, global step, and RNG state. It does not guarantee mid-epoch replay.

Legacy GNM/ViNT/NoMaD weight layouts are not remapped silently. Use `runtime.allow_legacy_weight_remap: true` only when intentionally migrating old checkpoints.

## Data Sampling

Negative goal sampling is configured centrally:

```yaml
data:
  goal_sampling:
    negative:
      enabled: true
      policy: offset_zero
      distance_label: max_dist_cat
```

Dataset-level `negative_mining` remains accepted for old configs. New non-standard data flows should register a data module and select it with `data.module_name`.

## Reproducibility

Use:

```yaml
runtime:
  seed: 0
  deterministic: true
```

`deterministic: true` disables cuDNN benchmark and requests deterministic PyTorch algorithms where available. Checkpoints created by the hardened schema also restore Python, NumPy, Torch CPU, and CUDA RNG state.

## Disable W&B

```yaml
logging:
  sinks:
    - name: wandb
      enabled: false
```

Use `strict: true` only when W&B failures should stop training.
