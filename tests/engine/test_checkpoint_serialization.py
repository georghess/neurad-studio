"""Round-trip full training checkpoints containing NumPy scheduler state."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from nerfstudio.engine.optimizers import AdamOptimizerConfig, Optimizers
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import Trainer, TrainerConfig
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.eval_utils import eval_load_checkpoint


class CheckpointPipeline(Pipeline):
    """Small CPU pipeline that exercises the real checkpoint serialization paths."""

    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        self.loaded_step = None

    def load_pipeline(self, loaded_state, step, strict=True):
        """Restore the parameter without allocating a dataset or renderer."""
        nn.Module.load_state_dict(self, loaded_state, strict=strict)
        self.loaded_step = step


def make_trainer(checkpoint_dir: Path) -> Trainer:
    """Initialize training state independently of Trainer.setup's optimizer ordering."""
    trainer = Trainer.__new__(Trainer)
    trainer.config = TrainerConfig(load_optimizer=True, load_scheduler=True)
    trainer.pipeline = CheckpointPipeline()
    trainer.checkpoint_dir = checkpoint_dir
    trainer.checkpoint_saving_tracker = SimpleNamespace(did_degrade=lambda: False, reset_latest=lambda: None)
    trainer.grad_scaler = torch.cuda.amp.GradScaler(enabled=False)
    trainer.optimizers = Optimizers(
        {
            "model": {
                "optimizer": AdamOptimizerConfig(lr=0.01),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=0.001, max_steps=100),
            }
        },
        {"model": [trainer.pipeline.param]},
    )
    return trainer


def take_step(trainer: Trainer) -> None:
    """Populate Adam moments and advance the repository's NumPy-based scheduler."""
    optimizer = trainer.optimizers.optimizers["model"]
    optimizer.zero_grad()
    trainer.pipeline.param.square().sum().backward()
    optimizer.step()
    trainer.optimizers.schedulers["model"].step()


@pytest.fixture
def saved_trainer(tmp_path):
    """Save through Trainer.save_checkpoint, including actual optimizer/scheduler state."""
    trainer = make_trainer(tmp_path / "checkpoints")
    take_step(trainer)
    assert isinstance(trainer.optimizers.optimizers["model"].param_groups[0]["lr"], np.floating)
    assert isinstance(trainer.optimizers.schedulers["model"].state_dict()["_last_lr"][0], np.floating)
    trainer.save_checkpoint(step=7)
    return trainer


@pytest.mark.parametrize("mode", ["file", "directory"])
def test_resume_checkpoint_with_numpy_scheduler_state(saved_trainer, mode):
    """Both training load paths restore full state and continue the same optimization."""
    resumed = make_trainer(saved_trainer.checkpoint_dir)
    if mode == "file":
        resumed.config.load_checkpoint = saved_trainer.checkpoint_dir / "step-000000007.ckpt"
    else:
        resumed.config.load_dir = saved_trainer.checkpoint_dir
    with torch.no_grad():
        resumed.pipeline.param.zero_()
    resumed._load_checkpoint()

    assert resumed._start_step == 8
    assert resumed.pipeline.loaded_step == 7
    torch.testing.assert_close(resumed.pipeline.param, saved_trainer.pipeline.param)
    saved_optimizer = saved_trainer.optimizers.optimizers["model"]
    resumed_optimizer = resumed.optimizers.optimizers["model"]
    assert resumed_optimizer.param_groups[0]["lr"] == saved_optimizer.param_groups[0]["lr"]
    assert (
        resumed.optimizers.schedulers["model"].state_dict() == saved_trainer.optimizers.schedulers["model"].state_dict()
    )
    for key in ("step", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(
            resumed_optimizer.state[resumed.pipeline.param][key],
            saved_optimizer.state[saved_trainer.pipeline.param][key],
        )
    take_step(saved_trainer)
    take_step(resumed)
    torch.testing.assert_close(resumed.pipeline.param, saved_trainer.pipeline.param)


def test_eval_loads_full_training_checkpoint(saved_trainer):
    """Evaluation/rendering must deserialize a checkpoint even when discarding its optimizer."""
    pipeline = CheckpointPipeline()
    with torch.no_grad():
        pipeline.param.zero_()
    config = TrainerConfig(load_dir=saved_trainer.checkpoint_dir)
    path, step = eval_load_checkpoint(config, pipeline)
    assert path == saved_trainer.checkpoint_dir / "step-000000007.ckpt"
    assert step == pipeline.loaded_step == 7
    torch.testing.assert_close(pipeline.param, saved_trainer.pipeline.param)
