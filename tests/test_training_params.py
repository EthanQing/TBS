from __future__ import annotations

import unittest

from train_platform.utils.training_params import (
    AUTO_BATCH_SIZE,
    extract_selected_gpu_ids,
    normalize_batch_size,
    normalize_device_spec,
    validate_training_params_for_engine,
)


class TrainingParamsTestCase(unittest.TestCase):
    def test_normalize_batch_size_supports_auto_batch(self) -> None:
        self.assertEqual(normalize_batch_size(AUTO_BATCH_SIZE), AUTO_BATCH_SIZE)
        self.assertEqual(normalize_batch_size(16), 16)

    def test_normalize_batch_size_rejects_zero(self) -> None:
        with self.assertRaises(ValueError):
            normalize_batch_size(0)

    def test_normalize_device_spec_supports_common_gpu_forms(self) -> None:
        self.assertEqual(normalize_device_spec("gpu"), "0")
        self.assertEqual(normalize_device_spec("cuda:0, 1"), "0,1")
        self.assertEqual(normalize_device_spec([0, 1]), "0,1")
        self.assertEqual(extract_selected_gpu_ids("cuda:0,1"), [0, 1])

    def test_ultralytics_multi_gpu_requires_fixed_divisible_batch(self) -> None:
        normalized = validate_training_params_for_engine(
            "ultralytics-yolo",
            {"batch_size": 16, "device": "cuda:0,1"},
        )
        self.assertEqual(normalized["device"], "0,1")

        with self.assertRaises(ValueError):
            validate_training_params_for_engine(
                "ultralytics-yolo",
                {"batch_size": AUTO_BATCH_SIZE, "device": "0,1"},
            )

        with self.assertRaises(ValueError):
            validate_training_params_for_engine(
                "ultralytics-yolo",
                {"batch_size": 15, "device": "0,1"},
            )

    def test_non_ultralytics_rejects_auto_batch_and_multi_gpu(self) -> None:
        with self.assertRaises(ValueError):
            validate_training_params_for_engine(
                "paddle-det",
                {"batch_size": AUTO_BATCH_SIZE, "device": "0"},
            )

        with self.assertRaises(ValueError):
            validate_training_params_for_engine(
                "paddle-det",
                {"batch_size": 8, "device": "0,1"},
            )


if __name__ == "__main__":
    unittest.main()
