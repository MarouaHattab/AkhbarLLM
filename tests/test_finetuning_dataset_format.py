import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.utils import jsonl
from src import templates


class JsonWritingTests(unittest.TestCase):
    def test_write_json_preserves_arabic_text(self) -> None:
        self.assertTrue(hasattr(jsonl, "write_json"))

        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "dataset.json"
            jsonl.write_json(output_path, [{"title": "خبر عربي"}])

            self.assertEqual(
                output_path.read_text(encoding="utf-8"),
                '[\n  {\n    "title": "خبر عربي"\n  }\n]',
            )


class LlamaFactoryFormattingTests(unittest.TestCase):
    def test_formatter_uses_system_message_from_templates(self) -> None:
        self.assertTrue(hasattr(templates, "FINETUNING_SYSTEM_MESSAGE"))
        self.assertEqual(
            templates.FINETUNING_SYSTEM_MESSAGE,
            "\n".join(
                (
                    "You are a professional NLP data parser.",
                    "Follow the provided Task and Output Schema to generate the Output JSON.",
                    "Do not generate any introduction or conclusion.",
                )
            ),
        )

        formatter = importlib.import_module(
            "src.workflows.format_finetuning_dataset"
        )
        result = formatter.format_finetuning_records(
            [
                {
                    "story": "story",
                    "task": "task",
                    "output_schema": {},
                    "response": {},
                }
            ],
            print_fn=lambda _: None,
        )
        self.assertEqual(
            result.samples[0]["system"],
            templates.FINETUNING_SYSTEM_MESSAGE,
        )

    def test_configure_stdout_encoding_uses_utf8_when_supported(self) -> None:
        module_name = "src.workflows.format_finetuning_dataset"
        formatter = importlib.import_module(module_name)
        self.assertTrue(hasattr(formatter, "configure_stdout_encoding"))

        class ReconfigurableStream:
            def __init__(self) -> None:
                self.encoding: str | None = None

            def reconfigure(self, *, encoding: str) -> None:
                self.encoding = encoding

        stdout = ReconfigurableStream()
        formatter.configure_stdout_encoding(stdout)
        self.assertEqual(stdout.encoding, "utf-8")

    def test_format_records_uses_schema_fallback_and_expected_fields(self) -> None:
        module_name = "src.workflows.format_finetuning_dataset"
        self.assertIsNotNone(importlib.util.find_spec(module_name))

        formatter = importlib.import_module(module_name)
        result = formatter.format_finetuning_records(
            [
                {
                    "story": "قصة عربية",
                    "task": "Extract details.",
                    "output_scheme": {"title": "string"},
                    "response": {"title": "نتيجة"},
                }
            ],
            shuffle_seed=101,
        )

        self.assertEqual(result.skipped, 0)
        self.assertEqual(result.samples[0]["input"], "")
        self.assertEqual(result.samples[0]["history"], [])
        self.assertIn(
            '# Output Schema:\n{"title": "string"}',
            result.samples[0]["instruction"],
        )
        self.assertEqual(result.samples[0]["output"], '{"title": "نتيجة"}')

    def test_format_records_skips_missing_schema_or_response(self) -> None:
        module_name = "src.workflows.format_finetuning_dataset"
        self.assertIsNotNone(importlib.util.find_spec(module_name))

        formatter = importlib.import_module(module_name)
        result = formatter.format_finetuning_records(
            [
                {"story": "one", "task": "task", "response": {}},
                {"story": "two", "task": "task", "output_schema": {}},
            ]
        )

        self.assertEqual(result.samples, [])
        self.assertEqual(result.skipped, 2)

    def test_workflow_writes_dataset_and_matching_registration(self) -> None:
        module_name = "src.workflows.format_finetuning_dataset"
        formatter = importlib.import_module(module_name)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sft.jsonl"
            output_path = root / "custom_train.json"
            registration_path = root / "dataset_info.json"
            source.write_text(
                '{"story":"B","task":"T","output_schema":{},"response":{}}\n'
                '{"story":"A","task":"T","output_schema":{},"response":{}}\n',
                encoding="utf-8",
            )

            stats = formatter.format_finetuning_dataset(
                source,
                output_path,
                registration_path,
                print_fn=lambda _: None,
            )

            self.assertEqual(stats.converted, 2)
            self.assertEqual(stats.skipped, 0)
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8"))[0]["input"], "")
            registration = json.loads(registration_path.read_text(encoding="utf-8"))
            self.assertEqual(
                registration["news_finetuning"]["file_name"],
                "custom_train.json",
            )
