"""Test optimizer initialization and state restoration after checkpoint parameters are loaded."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from nerfstudio.engine.trainer import Trainer


class ResumeOrderTest(unittest.TestCase):
    """Verify parameter rebinding and resumed updates using real Adam state."""

    def check_resume(self, mode: str, restore_optimizer: bool = True) -> None:
        """Check optimizer initialization for file, directory, and fresh training paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)
            saved_param = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
            saved_optimizer = torch.optim.Adam([saved_param], lr=0.01)
            saved_scheduler = torch.optim.lr_scheduler.ExponentialLR(saved_optimizer, gamma=0.9)
            saved_param.sum().backward()
            saved_optimizer.step()
            saved_scheduler.step()
            saved = {
                "step": 28000,
                "pipeline": saved_param.detach().clone(),
                "optimizers": {"means": saved_optimizer.state_dict()},
                "schedulers": {"means": saved_scheduler.state_dict()},
                "scalers": {},
            }
            checkpoint_path = checkpoint_dir / "step-000028000.ckpt"
            torch.save(saved, checkpoint_path)
            events = []
            pipeline = SimpleNamespace(param=torch.nn.Parameter(torch.zeros(1)))

            def load_pipeline(state: torch.Tensor, step: int) -> None:
                """Replace the parameter to simulate a changed Gaussian count."""
                events.append("model")
                pipeline.param = torch.nn.Parameter(state.clone())

            pipeline.load_pipeline = load_pipeline
            trainer = Trainer.__new__(Trainer)
            trainer._start_step = 0
            trainer.pipeline = pipeline
            trainer.config = SimpleNamespace(
                load_dir=checkpoint_dir if mode == "directory" else None,
                load_checkpoint=checkpoint_path if mode == "file" else None,
                load_step=None,
                training_start_step=None,
                load_optimizer=restore_optimizer,
                load_scheduler=restore_optimizer,
            )
            trainer.grad_scaler = SimpleNamespace(load_state_dict=lambda state: None)

            def setup_optimizers() -> SimpleNamespace:
                """Bind the loaded parameter before restoring optimizer state."""
                events.append("optimizer")
                optimizer = torch.optim.Adam([pipeline.param], lr=0.01)
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
                return SimpleNamespace(
                    optimizer=optimizer,
                    scheduler=scheduler,
                    load_optimizers=lambda state: optimizer.load_state_dict(state["means"]),
                    load_schedulers=lambda state: scheduler.load_state_dict(state["means"]),
                )

            trainer.setup_optimizers = setup_optimizers
            trainer._load_checkpoint()
            expected_events = ["optimizer"] if mode == "fresh" else ["model", "optimizer"]
            self.assertEqual(events, expected_events)
            optimizer = trainer.optimizers.optimizer
            self.assertIs(optimizer.param_groups[0]["params"][0], pipeline.param)
            self.assertEqual(trainer._start_step, 0 if mode == "fresh" else 28001)
            if mode != "fresh" and restore_optimizer:
                state = optimizer.state[pipeline.param]
                self.assertEqual(int(state["step"]), 1)
                torch.testing.assert_close(state["exp_avg"], saved_optimizer.state[saved_param]["exp_avg"])
                torch.testing.assert_close(state["exp_avg_sq"], saved_optimizer.state[saved_param]["exp_avg_sq"])
                self.assertEqual(trainer.optimizers.scheduler.state_dict(), saved_scheduler.state_dict())
                self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.009)
            else:
                self.assertEqual(len(optimizer.state), 0)
            before = pipeline.param.detach().clone()
            pipeline.param.sum().backward()
            optimizer.step()
            self.assertFalse(torch.equal(before, pipeline.param.detach()))

    def test_file_resume(self) -> None:
        """Restore optimizer and scheduler state from a checkpoint file."""
        self.check_resume("file")

    def test_directory_resume(self) -> None:
        """Restore optimizer and scheduler state from a checkpoint directory."""
        self.check_resume("directory")

    def test_fresh_training(self) -> None:
        """Initialize optimizers when training without a checkpoint."""
        self.check_resume("fresh")

    def test_resume_without_optimizer_state(self) -> None:
        """Bind loaded parameters even when optimizer restoration is disabled."""
        self.check_resume("file", restore_optimizer=False)


if __name__ == "__main__":
    unittest.main()
