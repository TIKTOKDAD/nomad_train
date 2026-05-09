# ============================================================
# Engineering hardening tests - checkpoint/config/data/runtime safety
# ============================================================
# 本文件覆盖 training_base 的工程化保护逻辑：
# 1. 检查 checkpoint 是否可恢复、是否保存随机数/GradScaler/回调状态
# 2. 检查配置规范化、数据采样可复现和 DDP/子集派生字段
# 3. 检查 Trainer 的 resume 语义、W&B 降级逻辑和分布式指标聚合

import builtins
import copy
import random
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from training_base.algorithms.base import Algorithm, StepResult
from training_base.callbacks.perf_monitor import PerfMonitorCallback
from training_base.core.checkpoint import ResumeState
from training_base.core.checkpoint import (
    atomic_torch_save,
    load_checkpoint,
    load_model_state,
    load_training_resume,
    restore_rng_state,
    save_checkpoint,
)
from training_base.core.runtime import RuntimeContext
from training_base.core.config import load_yaml, normalize_config
from training_base.core.image_size import as_torch_resize_size
from training_base.data import data_module as data_module_exports
from training_base.data.data_module import resolve_data_runtime
from training_base.data.goal_sampling import normalize_goal_sampling_config, sample_navigation_goal
from training_base.data.labeling import sample_goal
from training_base.data.sampling import EpochAwareDataset, EpochAwareSampler, stable_subset_indices
from training_base.loggers.metric_store import reduce_metric_logs_distributed
from training_base.loggers.wandb import WandBSink
from training_base.registry import (
    algorithm_registry,
    callback_registry,
    data_module_registry,
    model_registry,
    objective_registry,
    register_builtins,
)
from training_base.trainer import Trainer


# 测试用有状态回调：用于验证 callback_state 在 checkpoint 中的保存/恢复
@callback_registry.register("unit_stateful")
class _UnitStatefulCallback:
    def __init__(self, config, context):
        self.count = int(config.get("start", 0))

    def on_epoch_end(self, **kwargs):
        self.count += 1

    def state_dict(self):
        return {"count": self.count}

    def load_state_dict(self, state):
        self.count = int(state.get("count", 0))


# 采样上下文测试数据集：每次 __getitem__ 内部会调用 goal sampling
class _ContextDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 6

    def __getitem__(self, index):
        from training_base.data.labeling import sample_goal

        goals_index = [(f"goal_{item}", item) for item in range(20)]
        return sample_goal("traj", 10, 0, 2, goals_index)


# 保留原始 batch 列表，便于比较多 worker DataLoader 的采样结果
def _identity_collate(batch):
    return batch


# checkpoint 相关硬化测试：完整保存、恢复、错误检查和旧权重 remap
class CheckpointHardeningTest(unittest.TestCase):
    def test_save_checkpoint_is_loadable_and_keeps_latest_backup(self):
        # latest.pth 应可读，并在覆盖前生成 latest.backup.pth
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            latest_path = os.path.join(tmpdir, "latest.pth")
            save_checkpoint(
                latest_path,
                epoch=0,
                global_step=3,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                algorithm_state={"x": 1},
                callback_state={},
                config={"runtime": {"epochs": 1}},
            )
            payload = load_checkpoint(latest_path, torch.device("cpu"))
            self.assertEqual(payload["epoch"], 0)
            self.assertEqual(payload["global_step"], 3)
            self.assertIn("rng_state", payload)
            self.assertIn("grad_scaler", payload)

            save_checkpoint(
                latest_path,
                epoch=1,
                global_step=4,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                algorithm_state={"x": 2},
                callback_state={},
                config={"runtime": {"epochs": 2}},
            )
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "latest.backup.pth")))

    def test_enabled_grad_scaler_state_is_checkpointed(self):
        # AMP GradScaler 启用时必须进入 checkpoint，恢复后训练尺度才连续
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = mock.Mock()
        scaler.is_enabled.return_value = True
        scaler.state_dict.return_value = {"scale": 128.0}

        with tempfile.TemporaryDirectory() as tmpdir:
            latest_path = os.path.join(tmpdir, "latest.pth")
            save_checkpoint(
                latest_path,
                epoch=0,
                global_step=1,
                model=model,
                optimizer=optimizer,
                scheduler=None,
                algorithm_state={},
                callback_state={},
                config={"runtime": {"epochs": 1}},
                grad_scaler=scaler,
            )
            payload = load_checkpoint(latest_path, torch.device("cpu"))
            self.assertEqual(payload["grad_scaler"], {"scale": 128.0})

    def test_rng_state_restores_python_numpy_and_torch(self):
        # 恢复 rng_state 后，Python/NumPy/Torch 的下一次随机数应与保存时一致
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        with tempfile.TemporaryDirectory() as tmpdir:
            latest_path = os.path.join(tmpdir, "latest.pth")
            random.seed(123)
            np.random.seed(123)
            torch.manual_seed(123)
            save_checkpoint(
                latest_path,
                epoch=0,
                global_step=1,
                model=model,
                optimizer=optimizer,
                scheduler=None,
                algorithm_state={},
                callback_state={},
                config={"runtime": {"epochs": 1}},
            )
            expected = (random.random(), np.random.rand(), torch.rand(1).item())
            random.seed(999)
            np.random.seed(999)
            torch.manual_seed(999)
            payload = load_checkpoint(latest_path, torch.device("cpu"))
            self.assertTrue(restore_rng_state(payload["rng_state"], path=latest_path))
            actual = (random.random(), np.random.rand(), torch.rand(1).item())
            self.assertEqual(expected[0], actual[0])
            self.assertEqual(expected[1], actual[1])
            self.assertEqual(expected[2], actual[2])

    def test_legacy_checkpoint_without_rng_state_warns_but_loads(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "latest.pth")
            atomic_torch_save(
                {
                    "model": torch.nn.Linear(1, 1).state_dict(),
                    "epoch": 0,
                    "global_step": 0,
                },
                path,
            )
            payload = load_checkpoint(path, torch.device("cpu"))
            with self.assertWarns(RuntimeWarning):
                self.assertFalse(restore_rng_state(payload.get("rng_state"), path=path))

    def test_callback_state_is_saved_after_epoch_callbacks(self):
        # checkpoint 回调应在其他 epoch_end 回调之后执行，确保保存的是更新后的状态
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        from training_base.callbacks import CallbackManager
        from training_base.core.runtime import RuntimeContext

        context = RuntimeContext(
            device=torch.device("cpu"),
            distributed=False,
            rank=0,
            local_rank=0,
            world_size=1,
            is_main_process=True,
            gpu_ids=[0],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "runtime": {"project_folder": tmpdir},
                "callbacks": [
                    {"name": "checkpoint"},
                    {"name": "unit_stateful"},
                ],
            }
            manager = CallbackManager(config, context)
            manager.call(
                "on_epoch_end",
                epoch=0,
                global_step=1,
                model=model,
                optimizer=optimizer,
                scheduler=None,
                algorithm=mock.Mock(state_dict=lambda state: {}),
                state={},
                config=config,
                eval_summaries={},
                callback_manager=manager,
            )
            payload = load_checkpoint(os.path.join(tmpdir, "latest.pth"), torch.device("cpu"))
            self.assertEqual(payload["callback_state"]["callbacks"][0]["name"], "unit_stateful")
            self.assertEqual(payload["callback_state"]["callbacks"][0]["state"]["count"], 1)

            second_manager = CallbackManager(config, context)
            second_manager.load_state_dict(payload["callback_state"])
            self.assertEqual(second_manager.state_dict()["callbacks"][1]["state"]["count"], 1)

    def test_bad_training_checkpoint_fails_loudly(self):
        # 看起来像训练 checkpoint 但缺必要字段时，应明确失败而不是静默当权重加载
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = os.path.join(tmpdir, "latest.pth")
            atomic_torch_save({"epoch": 1}, bad_path)
            with self.assertRaises(RuntimeError):
                load_checkpoint(bad_path, torch.device("cpu"))

    def test_strict_resume_rejects_schema_model_key_mismatch(self):
        # strict resume 遇到模型结构不匹配必须报错，防止误用错误 checkpoint
        source_model = torch.nn.Linear(2, 1)
        target_model = torch.nn.Linear(3, 1)
        target_optimizer = torch.optim.SGD(target_model.parameters(), lr=0.1)
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "latest.pth")
            save_checkpoint(
                checkpoint_path,
                epoch=0,
                global_step=1,
                model=source_model,
                optimizer=torch.optim.SGD(source_model.parameters(), lr=0.1),
                scheduler=None,
                algorithm_state={},
                callback_state={},
                config={},
            )
            config = {
                "runtime": {
                    "load_checkpoint_path": checkpoint_path,
                    "resume_strict": True,
                    "allow_legacy_weight_remap": False,
                }
            }
            with self.assertRaises(RuntimeError):
                load_training_resume(
                    model=target_model,
                    optimizer=target_optimizer,
                    scheduler=None,
                    config=config,
                    device=torch.device("cpu"),
                    model_name="gnm",
                )

    def test_legacy_weight_remap_requires_explicit_flag(self):
        # 旧版 key remap 只有显式开启时才允许，避免过度宽松的兼容掩盖错误
        class LegacyGnm(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Module()
                self.encoder.obs_mobilenet = torch.nn.Linear(1, 1, bias=False)

        checkpoint = {"obs_mobilenet.weight": torch.ones(1, 1)}
        model = LegacyGnm()
        with self.assertRaises(RuntimeError):
            load_model_state(model, checkpoint, strict=True, model_name="gnm")
        load_model_state(
            model,
            checkpoint,
            strict=True,
            model_name="gnm",
            allow_legacy_remap=True,
        )
        self.assertTrue(torch.equal(model.encoder.obs_mobilenet.weight, torch.ones(1, 1)))


# 配置层硬化测试：非法字段拒绝、内置配置可解析、旧字段兼容
class ConfigHardeningTest(unittest.TestCase):
    def _default_config(self):
        return load_yaml(os.path.join("training_base", "configs", "defaults.yaml"))

    def test_rejects_unsupported_context_type(self):
        # 当前数据上下文只支持 temporal，非法值应在配置规范化阶段尽早失败
        config = self._default_config()
        config["data"]["context_type"] = "randomized"
        with self.assertRaises(ValueError):
            normalize_config(config)

    def test_rejects_non_image_obs_and_goal_type(self):
        config = self._default_config()
        config["data"]["obs_type"] = "state"
        with self.assertRaises(ValueError):
            normalize_config(copy.deepcopy(config))

        config = self._default_config()
        config["data"]["goal_type"] = "text"
        with self.assertRaises(ValueError):
            normalize_config(config)

    def test_visualization_image_size_uses_width_height_config_order(self):
        self.assertEqual(as_torch_resize_size([160, 120], "visualization.image_size"), (120, 160))

    def test_rejects_invalid_eval_fraction_and_amp_dtype(self):
        # eval fraction 和 AMP dtype 是运行时关键字段，非法值不应拖到训练中才报错
        config = self._default_config()
        config["logging"]["eval"]["schedule"]["fraction"] = 0
        with self.assertRaises(ValueError):
            normalize_config(copy.deepcopy(config))

        config = self._default_config()
        config["runtime"]["amp_dtype"] = "fp32"
        with self.assertRaises(ValueError):
            normalize_config(config)

    def test_builtin_configs_normalize_and_core_registries_resolve(self):
        # 仓库内置 YAML 都应该能规范化，并能在核心 registry 中找到对应组件
        register_builtins()
        defaults_path = os.path.join("training_base", "configs", "defaults.yaml")
        for filename in os.listdir(os.path.join("training_base", "configs")):
            if not filename.endswith(".yaml"):
                continue
            config = load_yaml(defaults_path)
            user_config = load_yaml(os.path.join("training_base", "configs", filename))
            from training_base.core.config import deep_merge

            normalized = normalize_config(deep_merge(config, user_config))
            self.assertIn(normalized["algorithm"]["name"], algorithm_registry.names())
            self.assertIn(normalized["model"]["name"], model_registry.names())
            self.assertIn(normalized["objective"]["name"], objective_registry.names())
            self.assertIn(normalized["data"]["module_name"], data_module_registry.names())

    def test_legacy_negative_mining_alias_can_disable_goal_sampling(self):
        # 旧字段 negative_mining=False 应迁移为新版 goal_sampling.negative.enabled=False
        config = self._default_config()
        config["data"].pop("goal_sampling", None)
        config["data"]["datasets"] = {"tiny": {"negative_mining": False}}
        normalized = normalize_config(config)
        self.assertFalse(normalized["data"]["goal_sampling"]["negative"]["enabled"])

    def test_project_folder_uses_configured_log_root(self):
        # run 输出目录必须使用 runtime.log_root，而不是默认落到当前工作目录
        from training_base.cli import _prepare_project_folder

        context = RuntimeContext(
            device=torch.device("cpu"),
            distributed=False,
            rank=0,
            local_rank=0,
            world_size=1,
            is_main_process=True,
            gpu_ids=[0],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            config = {"runtime": {"log_root": tmpdir, "project_name": "proj", "run_name": "run"}}
            _prepare_project_folder(config, context)
            project_folder = config["runtime"]["project_folder"]
            self.assertTrue(os.path.isabs(project_folder))
            self.assertTrue(project_folder.startswith(os.path.abspath(tmpdir)))
            self.assertIn(os.path.join("proj", "run_"), project_folder)


# 数据采样与 DataLoader 派生字段测试
class DataSubsetTest(unittest.TestCase):
    def test_sampling_module_and_legacy_data_module_exports_match(self):
        # data_module 旧导出路径应继续指向 sampling 模块中的新实现
        self.assertIs(data_module_exports.EpochAwareDataset, EpochAwareDataset)
        self.assertIs(data_module_exports.EpochAwareSampler, EpochAwareSampler)
        self.assertIs(data_module_exports.stable_subset_indices, stable_subset_indices)

    def test_stable_subset_indices_are_seeded(self):
        # 同一 seed 下抽取的训练子集应完全一致
        first = stable_subset_indices(10, 0.3, seed=7)
        second = stable_subset_indices(10, 0.3, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertEqual(len(set(first)), 3)

    def test_stable_subset_indices_change_with_seed(self):
        # 不同 seed 应得到不同子集，证明 seed 确实进入采样路径
        first = stable_subset_indices(100, 0.25, seed=7)
        second = stable_subset_indices(100, 0.25, seed=8)
        self.assertNotEqual(first, second)

    def test_sample_goal_is_seed_epoch_index_deterministic(self):
        # goal sampling 应由 seed/epoch/index 唯一决定，支持 persistent_workers 可复现
        goals_index = [("a", 1), ("b", 2), ("c", 3)]
        dataset = EpochAwareDataset(list(range(3)), seed=11)
        sampler = EpochAwareSampler(dataset, shuffle=False, seed=11)

        def sample_for(epoch, index):
            dataset.set_epoch(epoch)
            with mock.patch.object(dataset, "dataset", [index]):
                # Use the same context path that DataLoader uses: sampler yields (epoch, index).
                dataset[(epoch, 0)]
            from training_base.data.labeling import sample_context

            with sample_context(seed=11, epoch=epoch, index=index):
                return sample_goal("traj", 10, 5, 2, goals_index)

        self.assertEqual(sample_for(3, 7), sample_for(3, 7))
        self.assertNotEqual(sample_for(3, 7), sample_for(4, 7))
        self.assertEqual(len(list(iter(sampler))), 3)

    def test_goal_sampling_new_config_preserves_legacy_default_and_can_disable_negative(self):
        # 新配置默认应保持旧负样本行为，同时允许显式关闭负样本
        goals_index = [("other", 5)]

        class ZeroRng:
            def integers(self, low, high):
                return 0

        legacy = sample_goal("traj", 10, 5, 2, goals_index, rng=ZeroRng())
        configured = sample_navigation_goal(
            "traj",
            10,
            5,
            2,
            goals_index,
            config=normalize_goal_sampling_config({"negative": {"enabled": True}}),
            rng=ZeroRng(),
        )
        disabled = sample_navigation_goal(
            "traj",
            10,
            5,
            2,
            goals_index,
            config=normalize_goal_sampling_config({"negative": {"enabled": False}}),
            rng=ZeroRng(),
        )
        self.assertEqual(legacy, configured)
        self.assertEqual(disabled, ("traj", 12, False))

    def test_epoch_aware_dataloader_is_deterministic_with_persistent_workers(self):
        # persistent_workers 下 worker 不重启，仍必须随 epoch 更新采样上下文
        def collect(seed, epoch):
            dataset = EpochAwareDataset(_ContextDataset(), seed=seed)
            sampler = EpochAwareSampler(dataset, shuffle=False, seed=seed)
            loader = DataLoader(
                dataset,
                batch_size=2,
                sampler=sampler,
                num_workers=2,
                persistent_workers=True,
                collate_fn=_identity_collate,
            )
            dataset.set_epoch(epoch)
            sampler.set_epoch(epoch)
            try:
                return list(loader)
            finally:
                iterator = getattr(loader, "_iterator", None)
                if iterator is not None:
                    iterator._shutdown_workers()

        self.assertEqual(collect(seed=11, epoch=3), collect(seed=11, epoch=3))
        self.assertNotEqual(collect(seed=11, epoch=3), collect(seed=11, epoch=4))
        self.assertNotEqual(collect(seed=11, epoch=3), collect(seed=12, epoch=3))

    def test_resolve_data_runtime_centralizes_effective_fields(self):
        # 数据 runtime 派生字段应由 resolve_data_runtime 一处集中计算
        runtime = {"batch_size": 8, "eval_batch_size": 3}
        effective = resolve_data_runtime(
            runtime,
            distributed=True,
            world_size=2,
            train_subset_size=5,
            train_subset_total_size=10,
            train_num_workers=4,
        )
        self.assertEqual(
            effective,
            {
                "global_batch_size": 8,
                "batch_size": 8,
                "per_device_batch_size": 4,
                "train_subset_size": 5,
                "train_subset_total_size": 10,
                "num_workers_per_rank": 4,
                "eval_batch_size": 3,
            },
        )
        self.assertEqual(runtime, {"batch_size": 8, "eval_batch_size": 3})


# 最小数据模块桩：用于验证 Trainer 在 resume 已完成时不会训练
class _DummyDataModule:
    def __init__(self):
        self.setup_calls = 0
        self.train_sampler = None
        self.train_loader = []
        self.test_dataloaders = {}
        self.transform = None

    def setup(self, build_lmdb_only=False):
        self.setup_calls += 1


# 单 batch 训练数据模块桩：用于验证 resume 后继续跑剩余 epoch
class _TinyTrainDataModule:
    def __init__(self):
        self.setup_calls = 0
        self.train_sampler = None
        self.train_loader = [torch.ones(1, 1)]
        self.test_dataloaders = {}
        self.transform = None
        self.epochs = []

    def setup(self, build_lmdb_only=False):
        self.setup_calls += 1

    def set_train_epoch(self, epoch):
        self.epochs.append(epoch)


class _ConfigMutatingDataModule:
    def __init__(self, config):
        self.config = config
        self.setup_calls = 0
        self.train_sampler = None
        self.train_loader = []
        self.test_dataloaders = {}
        self.transform = None
        self.loader_summary = "loader summary"

    def setup(self, build_lmdb_only=False):
        self.setup_calls += 1
        runtime = self.config["runtime"]
        runtime["per_device_batch_size"] = 7
        runtime["num_workers_per_rank"] = 3
        runtime["test_num_workers_per_rank"] = 2
        self.config["data"]["dataset_metadata"] = {"0": {"dataset_name": "tiny"}}


class _CaptureRecorder:
    def __init__(self):
        self.logged = {}

    def log_metrics(self, data, *, step=None, commit=True):
        self.logged.update(data)


# resume 语义测试算法：恢复点到达目标 epoch 时 train_step 不应被调用
class _ResumeOnlyAlgorithm(Algorithm):
    name = "dummy"

    def __init__(self, current_epoch):
        self.current_epoch = current_epoch
        self.train_calls = 0

    def build_model(self, config):
        return torch.nn.Linear(1, 1), {}

    def build_objective(self, config):
        return object()

    def configure_optimizers(self, model, config):
        return torch.optim.SGD(model.parameters(), lr=0.1), None

    def prepare_resume(self, model, optimizer, scheduler, config, device):
        return ResumeState(current_epoch=self.current_epoch, extra={"global_step": 5})

    def create_state(self, model, model_extras, objective, config, device, resume_state):
        return {}

    def train_step(self, model, prepared, state, config):
        self.train_calls += 1
        raise AssertionError("train_step should not run when resume already reached target epoch")

    def prepare_batch(self, batch, transform, device, mode, should_log_images):
        return batch


# 真正跑一小步训练的算法桩：用于验证 checkpoint resume 能延续训练状态
class _CheckpointResumeAlgorithm(Algorithm):
    name = "checkpoint_resume"

    def __init__(self):
        self.model = None

    def build_model(self, config):
        self.model = torch.nn.Linear(1, 1)
        return self.model, {}

    def build_objective(self, config):
        return object()

    def configure_optimizers(self, model, config):
        return torch.optim.SGD(model.parameters(), lr=0.1), None

    def prepare_resume(self, model, optimizer, scheduler, config, device):
        checkpoint_path = config["runtime"].get("load_checkpoint_path")
        if not checkpoint_path:
            return ResumeState(extra={"global_step": 0})
        checkpoint = load_checkpoint(checkpoint_path, device)
        restore_rng_state(checkpoint.get("rng_state"), path=checkpoint_path)
        load_model_state(model, checkpoint, strict=True, model_name=None)
        optimizer.load_state_dict(checkpoint["optimizer"])
        return ResumeState(
            current_epoch=checkpoint["epoch"] + 1,
            latest_checkpoint=checkpoint,
            load_project_folder=os.path.dirname(checkpoint_path),
            extra={"global_step": checkpoint["global_step"]},
        )

    def create_state(self, model, model_extras, objective, config, device, resume_state):
        return {}

    def prepare_batch(self, batch, transform, device, mode, should_log_images):
        return batch.to(device)

    def train_step(self, model, prepared, state, config):
        loss = model(prepared).pow(2).mean()
        return StepResult(loss=loss, logs={"total_loss": loss}, batch_size=int(prepared.shape[0]))


# Trainer resume 语义测试：目标总 epoch 与 checkpoint current_epoch 的关系
class ResumeSemanticsTest(unittest.TestCase):
    def _context(self):
        return RuntimeContext(
            device=torch.device("cpu"),
            distributed=False,
            rank=0,
            local_rank=0,
            world_size=1,
            is_main_process=True,
            gpu_ids=[0],
        )

    def test_resume_exits_when_current_epoch_reaches_target_total_epochs(self):
        # runtime.epochs 表示目标总轮数，不是 resume 后追加轮数
        config = load_yaml(os.path.join("training_base", "configs", "defaults.yaml"))
        config["runtime"]["epochs"] = 3
        config["runtime"]["gpu_ids"] = [0]
        config["runtime"]["distributed"] = False
        config["logging"]["sinks"] = []
        config["callbacks"] = []

        context = RuntimeContext(
            device=torch.device("cpu"),
            distributed=False,
            rank=0,
            local_rank=0,
            world_size=1,
            is_main_process=True,
            gpu_ids=[0],
        )
        datamodule = _DummyDataModule()
        algorithm = _ResumeOnlyAlgorithm(current_epoch=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            config["runtime"]["project_folder"] = tmpdir
            trainer = Trainer(config, algorithm, datamodule, context)

            trainer.fit()

        self.assertEqual(datamodule.setup_calls, 1)
        self.assertEqual(algorithm.train_calls, 0)
        self.assertEqual(trainer.global_step, 5)

    def test_effective_config_is_saved_after_datamodule_setup(self):
        config = load_yaml(os.path.join("training_base", "configs", "defaults.yaml"))
        config["runtime"]["epochs"] = 1
        config["runtime"]["gpu_ids"] = [0]
        config["runtime"]["distributed"] = False
        config["logging"]["sinks"] = []
        config["callbacks"] = []

        context = self._context()
        with tempfile.TemporaryDirectory() as tmpdir:
            config["runtime"]["project_folder"] = tmpdir
            datamodule = _ConfigMutatingDataModule(config)
            trainer = Trainer(config, _ResumeOnlyAlgorithm(current_epoch=1), datamodule, context)
            trainer.fit()

            resolved_path = os.path.join(tmpdir, "config.resolved.yaml")
            with open(resolved_path, "r", encoding="utf-8") as f:
                saved = yaml.safe_load(f)

        self.assertEqual(datamodule.setup_calls, 1)
        self.assertEqual(saved["runtime"]["per_device_batch_size"], 7)
        self.assertEqual(saved["runtime"]["num_workers_per_rank"], 3)
        self.assertEqual(saved["runtime"]["test_num_workers_per_rank"], 2)
        self.assertEqual(saved["data"]["dataset_metadata"], {"0": {"dataset_name": "tiny"}})

    def test_two_epoch_checkpoint_resume_continues_one_more_epoch(self):
        # 从 2 epoch checkpoint 恢复到 runtime.epochs=3 时，只应继续训练第 3 轮
        config = load_yaml(os.path.join("training_base", "configs", "defaults.yaml"))
        config["runtime"]["epochs"] = 2
        config["runtime"]["gpu_ids"] = [0]
        config["runtime"]["distributed"] = False
        config["runtime"]["train"] = True
        config["logging"]["sinks"] = []
        config["callbacks"] = [
            {"name": "checkpoint"},
            {"name": "unit_stateful"},
        ]

        context = RuntimeContext(
            device=torch.device("cpu"),
            distributed=False,
            rank=0,
            local_rank=0,
            world_size=1,
            is_main_process=True,
            gpu_ids=[0],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            config["runtime"]["project_folder"] = tmpdir
            first_data = _TinyTrainDataModule()
            first_trainer = Trainer(config, _CheckpointResumeAlgorithm(), first_data, context)
            first_trainer.fit()

            latest_path = os.path.join(tmpdir, "latest.pth")
            payload = load_checkpoint(latest_path, torch.device("cpu"))
            self.assertEqual(payload["epoch"], 1)
            self.assertEqual(payload["global_step"], 2)
            self.assertEqual(payload["callback_state"]["callbacks"][0]["state"]["count"], 2)

            resumed_config = copy.deepcopy(config)
            resumed_config["runtime"]["epochs"] = 3
            resumed_config["runtime"]["load_checkpoint_path"] = latest_path
            resumed_data = _TinyTrainDataModule()
            resumed_trainer = Trainer(resumed_config, _CheckpointResumeAlgorithm(), resumed_data, context)
            resumed_trainer.fit()

            resumed_payload = load_checkpoint(latest_path, torch.device("cpu"))
            self.assertEqual(resumed_data.epochs, [2])
            self.assertEqual(resumed_trainer.global_step, 3)
            self.assertEqual(resumed_payload["epoch"], 2)
            self.assertEqual(resumed_payload["callback_state"]["callbacks"][0]["state"]["count"], 3)


# W&B sink 降级行为测试
class PerfMonitorTest(unittest.TestCase):
    def test_runtime_config_logs_configured_and_effective_workers(self):
        context = RuntimeContext(
            device=torch.device("cpu"),
            distributed=False,
            rank=0,
            local_rank=0,
            world_size=1,
            is_main_process=True,
            gpu_ids=[0],
        )
        config = {
            "runtime": {
                "num_workers": 8,
                "num_workers_per_rank": 4,
                "test_num_workers": 2,
                "test_num_workers_per_rank": 1,
                "amp": False,
            },
            "logging": {},
        }
        recorder = _CaptureRecorder()
        PerfMonitorCallback({}, context).log_runtime_config(recorder=recorder, config=config, global_step=0)

        self.assertEqual(recorder.logged["runtime/dataloader/train_num_workers_configured"], 8)
        self.assertEqual(recorder.logged["runtime/dataloader/train_num_workers_per_rank"], 4)
        self.assertEqual(recorder.logged["runtime/dataloader/test_num_workers_configured"], 2)
        self.assertEqual(recorder.logged["runtime/dataloader/test_num_workers_per_rank"], 1)


class WandBSinkTest(unittest.TestCase):
    def test_disabled_wandb_does_not_import(self):
        # disabled sink 不应导入 wandb，方便无 wandb 环境运行测试
        context = mock.Mock(is_main_process=True)
        sink = WandBSink({"enabled": False}, context)
        self.assertIsNone(sink.run)


# 指标聚合与 W&B 导入失败处理测试
class MetricReduceTest(unittest.TestCase):
    def test_reduce_metric_logs_no_distributed_returns_clean_floats(self):
        # 非分布式模式下应过滤 None/NaN，并把 Tensor 转成 Python float
        logs = {
            "loss": torch.tensor(2.0),
            "none": None,
            "nan": float("nan"),
        }
        self.assertEqual(reduce_metric_logs_distributed(logs, torch.device("cpu")), {"loss": 2.0})

    def test_reduce_metric_logs_uses_all_reduce_payload(self):
        # 分布式聚合使用 sum/count payload，保证各 rank 均值正确
        captured = []

        def fake_all_reduce(payload, op=None):
            captured.append((payload.clone(), op))
            payload[0] += 4.0
            payload[1] += 1.0

        with mock.patch("training_base.loggers.metric_store.dist.is_available", return_value=True), \
             mock.patch("training_base.loggers.metric_store.dist.is_initialized", return_value=True), \
             mock.patch("training_base.loggers.metric_store.dist.all_reduce", side_effect=fake_all_reduce):
            reduced = reduce_metric_logs_distributed({"loss": 2.0}, torch.device("cpu"))

        self.assertEqual(len(captured), 1)
        self.assertEqual(reduced["loss"], 3.0)

    def test_reduce_metric_logs_keeps_collective_keys_for_nan_values(self):
        # 即使某个 rank 的值是 NaN，也要参与同名 collective，避免不同 rank 调用次数不一致
        calls = []

        def fake_all_reduce(payload, op=None):
            calls.append(payload.clone())
            payload[0] += 4.0
            payload[1] += 1.0

        with mock.patch("training_base.loggers.metric_store.dist.is_available", return_value=True), \
             mock.patch("training_base.loggers.metric_store.dist.is_initialized", return_value=True), \
             mock.patch("training_base.loggers.metric_store.dist.all_reduce", side_effect=fake_all_reduce):
            reduced = reduce_metric_logs_distributed({"loss": 2.0, "nan": float("nan")}, torch.device("cpu"))

        self.assertEqual(len(calls), 2)
        self.assertEqual(reduced["loss"], 3.0)
        self.assertEqual(reduced["nan"], 4.0)

    def test_non_strict_wandb_import_failure_does_not_raise(self):
        # strict=False 时 wandb 导入失败应降级禁用，不影响训练继续
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "wandb":
                raise RuntimeError("wandb unavailable")
            return real_import(name, *args, **kwargs)

        context = mock.Mock(is_main_process=True)
        with mock.patch("builtins.__import__", side_effect=fake_import):
            sink = WandBSink({"enabled": True, "project": "test", "strict": False}, context)
        self.assertIsNone(sink.run)


if __name__ == "__main__":
    unittest.main()
