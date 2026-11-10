"""Test gradient-free standalone evaluation and checkpoint metadata output."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from nerfstudio.scripts.eval import ComputePSNR


class EvaluationNoGradTest(unittest.TestCase):
    """Prevent standalone evaluation from retaining unnecessary backward graphs."""

    def test_eval_disables_grad_and_writes_metrics(self) -> None:
        """Disable gradients only during evaluation and preserve the checkpoint in JSON."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "metrics.json"
            checkpoint_path = Path(temp_dir) / "step-000030000.ckpt"

            def evaluate(**kwargs: object) -> dict[str, float]:
                """Evaluate a trainable parameter without constructing a backward graph."""
                self.assertFalse(torch.is_grad_enabled())
                parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
                rendered = parameter.square()
                self.assertFalse(rendered.requires_grad)
                return {"psnr": 27.0}

            pipeline = SimpleNamespace(get_average_eval_image_metrics=evaluate)
            config = SimpleNamespace(experiment_name="test-eval", method_name="splatad")
            with torch.enable_grad():
                with patch(
                    "nerfstudio.scripts.eval.eval_setup",
                    return_value=(config, pipeline, checkpoint_path, 30000),
                ):
                    ComputePSNR(load_config=Path("unused.yml"), output_path=output_path).main()
                self.assertTrue(torch.is_grad_enabled())
            metrics = json.loads(output_path.read_text())
            self.assertEqual(metrics["checkpoint"], str(checkpoint_path))
            self.assertEqual(metrics["results"], {"psnr": 27.0})


if __name__ == "__main__":
    unittest.main()
