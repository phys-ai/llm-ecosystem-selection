import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

import log_collect_tool as collector
from analysis.ai_social_experiment_replay import prepare_log_streaming


ROOT = Path(__file__).resolve().parents[1]
SUBSET = ROOT / "benchmarks" / "bfcl" / "tasks.jsonl"


class ToolReplayContractTests(unittest.TestCase):
    def test_fitness_is_direct_and_safety_channel_is_disabled(self) -> None:
        config = collector.LogGenerationConfig(
            subset_path=str(SUBSET),
            n_tasks=5,
            mock_mode="empty",
            api_max_workers=1,
            save_prompt=False,
            save_raw_response=False,
        )
        instance = collector.BFCLLogCollector(config)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_dir = Path(temporary) / "checkpoints"
            with redirect_stdout(io.StringIO()):
                instance.collect(checkpoint_dir)
            prepared = prepare_log_streaming(
                checkpoint_dir,
                run_name="tool_contract",
                score_weights={"like": 1.0},
            )

        self.assertEqual(prepared.base_score.shape, (5, 9))
        self.assertAlmostEqual(float(prepared.base_score.mean()), 0.2)
        self.assertTrue(np.array_equal(prepared.quality_score, prepared.base_score))
        self.assertTrue(np.all(prepared.safety_score == 1.0))
        self.assertTrue(np.all(prepared.risk_penalty == 0.0))


if __name__ == "__main__":
    unittest.main()
