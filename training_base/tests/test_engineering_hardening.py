import builtins
import copy
import random
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from training_base.algorithms.base import Algorithm, StepResult
from training_base.core.checkpoint import ResumeState
from training_base.core.checkpoint import atomic_torch_save, load_checkpoint, load_model_state, restore_rng_state, save_checkpoint
from training_base.core.runtime import RuntimeContext
from training_base.core.config import load_yaml, normalize_config
from training_base.data.data_module import EpochAwareDataset, EpochAwareSampler, stable_subset_indices
from training_base.data.labeling import sample_goal
from training_base.loggers.wandb import WandBSink
from training_base.registry import callback_registry
from training_base.trainer import Trainer


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


class CheckpointHardeningTest(unittest.TestCase):
    def test_save_checkpoint_is_loadable_and_keeps_latest_backup(self):
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

    def test_rng_state_restores_python_numpy_and_torch(self):
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
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = os.path.join(tmpdir, "latest.pth")
            atomic_torch_save({"epoch": 1}, bad_path)
            with self.assertRaises(RuntimeError):
                load_checkpoint(bad_path, torch.device("cpu"))


class ConfigHardeningTest(unittest.TestCase):
    def _default_config(self):
        return load_yaml(os.path.join("training_base", "configs", "defaults.yaml"))

    def test_rejects_unsupported_context_type(self):
        config = self._default_config()
        config["data"]["context_type"] = "randomized"
        with self.assertRaises(ValueError):
            normalize_config(config)

    def test_rejects_invalid_eval_fraction_and_amp_dtype(self):
        config = self._default_config()
        config["logging"]["eval"]["schedule"]["fraction"] = 0
        with self.assertRaises(ValueError):
            normalize_config(copy.deepcopy(config))

        config = self._default_config()
        config["runtime"]["amp_dtype"] = "fp32"
        with self.assertRaises(ValueError):
            normalize_config(config)


class DataSubsetTest(unittest.TestCase):
    def test_stable_subset_indices_are_seeded(self):
        first = stable_subset_indices(10, 0.3, seed=7)
        second = stable_subset_indices(10, 0.3, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertEqual(len(set(first)), 3)

    def test_stable_subset_indices_change_with_seed(self):
        first = stable_subset_indices(100, 0.25, seed=7)
        second = stable_subset_indices(100, 0.25, seed=8)
        self.assertNotEqual(first, second)

    def test_sample_goal_is_seed_epoch_index_deterministic(self):
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


class _DummyDataModule:
    def __init__(self):
        self.setup_calls = 0
        self.train_sampler = None
        self.train_loader = []
        self.test_dataloaders = {}
        self.transform = None

    def setup(self, build_lmdb_only=False):
        self.setup_calls += 1


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


class ResumeSemanticsTest(unittest.TestCase):
    def test_resume_exits_when_current_epoch_reaches_target_total_epochs(self):
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

    def test_two_epoch_checkpoint_resume_continues_one_more_epoch(self):
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


class WandBSinkTest(unittest.TestCase):
    def test_disabled_wandb_does_not_import(self):
        context = mock.Mock(is_main_process=True)
        sink = WandBSink({"enabled": False}, context)
        self.assertIsNone(sink.run)

    def test_non_strict_wandb_import_failure_does_not_raise(self):
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
