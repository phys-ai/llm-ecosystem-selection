import json
from collections import Counter
import tempfile
import unittest
from pathlib import Path

import log_collect_tool as collector


ROOT = Path(__file__).resolve().parents[1]
SUBSET = ROOT / "benchmarks" / "bfcl" / "tasks.jsonl"


class BFCLSubsetTests(unittest.TestCase):
    def test_pinned_subset_is_balanced_and_hash_valid(self) -> None:
        tasks = collector.load_bfcl_tasks(SUBSET)
        counts = {category: 0 for category in collector.BFCL_CATEGORIES}
        for task in tasks:
            counts[task.category] += 1
        self.assertEqual(len(tasks), 100)
        self.assertEqual(set(counts.values()), {20})

    def test_gold_calls_score_correct_for_every_task(self) -> None:
        for task in collector.load_bfcl_tasks(SUBSET):
            tools, api_to_original = collector.compile_openai_tools(task.functions)
            original_to_api = {
                original: api_name
                for api_name, original in api_to_original.items()
            }
            raw_calls = collector._gold_mock_calls(task, original_to_api)
            predicted, parse_errors = collector.normalize_predicted_tool_calls(
                raw_calls,
                api_to_original,
            )
            self.assertEqual(parse_errors, [], task.task_id)
            self.assertTrue(
                collector.score_bfcl_tool_calls(task, predicted)["valid"],
                task.task_id,
            )
            self.assertEqual(len(tools), len(task.functions))

    def test_empty_calls_only_pass_irrelevance(self) -> None:
        for task in collector.load_bfcl_tasks(SUBSET):
            valid = collector.score_bfcl_tool_calls(task, [])["valid"]
            self.assertEqual(valid, task.category == "irrelevance", task.task_id)

    def test_nested_dict_and_list_of_dict_arguments(self) -> None:
        task = collector.BFCLTask(
            task_id="nested_test",
            category="simple_python",
            messages=({"role": "user", "content": "Process the structured values."},),
            functions=(
                {
                    "name": "nested.process",
                    "parameters": {
                        "type": "dict",
                        "required": ["area", "records"],
                        "properties": {
                            "area": {
                                "type": "dict",
                                "properties": {
                                    "width": {"type": "integer"},
                                    "height": {"type": "integer"},
                                },
                            },
                            "records": {
                                "type": "array",
                                "items": {
                                    "type": "dict",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "score": {"type": "integer"},
                                        "note": {"type": "string"},
                                    },
                                },
                            },
                        },
                    },
                },
            ),
            ground_truth=(
                {
                    "nested.process": {
                        "area": [{"width": [20], "height": [12]}],
                        "records": [
                            [
                                {"name": ["A"], "score": [1], "note": [""]},
                                {"name": ["B"], "score": [2], "note": ["ok", ""]},
                            ]
                        ],
                    }
                },
            ),
        )
        correct = [
            {
                "name": "nested.process",
                "arguments": {
                    "area": {"width": 20, "height": 12},
                    "records": [
                        {"name": "A", "score": 1},
                        {"name": "B", "score": 2, "note": "ok"},
                    ],
                },
            }
        ]
        self.assertTrue(collector.score_bfcl_tool_calls(task, correct)["valid"])
        incorrect = json.loads(json.dumps(correct))
        incorrect[0]["arguments"]["area"]["width"] = 19
        self.assertFalse(collector.score_bfcl_tool_calls(task, incorrect)["valid"])

        tools, api_to_original = collector.compile_openai_tools(task.functions)
        original_to_api = {
            original: api_name for api_name, original in api_to_original.items()
        }
        raw = collector._gold_mock_calls(task, original_to_api)
        predicted, errors = collector.normalize_predicted_tool_calls(
            raw,
            api_to_original,
        )
        self.assertEqual(errors, [])
        self.assertEqual(predicted, correct)
        self.assertTrue(collector.score_bfcl_tool_calls(task, predicted)["valid"])
        self.assertEqual(tools[0]["function"]["name"], "nested_process")


class CollectorPipelineTests(unittest.TestCase):
    def test_mock_collection_and_resume(self) -> None:
        config = collector.LogGenerationConfig(
            subset_path=str(SUBSET),
            n_tasks=2,
            mock_mode="gold",
            save_prompt=False,
            save_raw_response=False,
        )
        instance = collector.BFCLLogCollector(config)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_dir = Path(temporary) / "checkpoints"
            first = instance.collect(checkpoint_dir)
            second = instance.collect(checkpoint_dir)
        expected_rows = 2 * len(instance.agents)
        self.assertEqual(len(first["contents"]), expected_rows)
        self.assertEqual(len(second["contents"]), expected_rows)
        self.assertTrue(all(row["correct"] for row in second["contents"]))

    def test_at_config_file_wrapper(self) -> None:
        wrapped = {
            "description": "test",
            "config_json": {"n_tasks": 3, "mock_mode": "empty"},
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.json"
            path.write_text(json.dumps(wrapped), encoding="utf-8")
            parsed = collector.parse_config_overrides(f"@{path}")
        self.assertEqual(parsed["n_tasks"], 3)
        self.assertEqual(parsed["mock_mode"], "empty")

    def test_checkpoint_rows_follow_social_schema_contract(self) -> None:
        config = collector.LogGenerationConfig(
            subset_path=str(SUBSET),
            n_tasks=1,
            mock_mode="gold",
            api_max_workers=1,
            save_prompt=True,
            save_raw_response=True,
        )
        instance = collector.BFCLLogCollector(config)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_dir = Path(temporary) / "checkpoints"
            instance.collect(checkpoint_dir)
            metadata = collector.load_json(checkpoint_dir / "metadata.json")
            bundle = collector.load_json(checkpoint_dir / "timestep_0000.json")

        self.assertEqual(len(metadata["evaluators"]), 1)
        self.assertFalse(metadata["eval_only"])
        self.assertFalse(metadata["reused_contents"])
        self.assertTrue(
            {
                "post_provider",
                "post_model",
                "post_prompt",
                "post_raw_response_text",
                "trait_bins",
                "source_subreddit",
                "source_post_id",
                *collector.SOCIAL_CONTEXT_AXES,
            }.issubset(bundle["contents"][0])
        )
        self.assertTrue(
            {
                "evaluator_trait",
                "agent_trait",
                "agent_trait_coords",
                "agent_trait_bins",
                "agent_trait_labels",
                "eval_provider",
                "eval_model",
                "eval_prompt",
                "eval_raw_response_text",
                "eval_parse_error",
                *collector.SOCIAL_CONTEXT_AXES,
            }.issubset(bundle["evaluations"][0])
        )
        self.assertTrue(
            {
                "agent_trait",
                "trait_coords",
                "trait_bins",
                "trait_labels",
                "audit_provider",
                "audit_model",
                "audit_prompt",
                "audit_raw_response_text",
                "audit_parse_error",
                "reason",
                *collector.SOCIAL_CONTEXT_AXES,
            }.issubset(bundle["audits"][0])
        )

    def test_task_truncation_is_stratified(self) -> None:
        config = collector.LogGenerationConfig(
            subset_path=str(SUBSET),
            n_tasks=13,
            task_sample_seed=7,
        )
        instance = collector.BFCLLogCollector(config)
        counts = Counter(task.category for task in instance.tasks)
        self.assertEqual(sum(counts.values()), 13)
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)


if __name__ == "__main__":
    unittest.main()
