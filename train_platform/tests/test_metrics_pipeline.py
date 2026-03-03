from __future__ import annotations

import unittest
from unittest.mock import patch

from train_platform.training.plugins.paddle_det import _apply_metric_aliases
from train_platform.workers.training.train_entry import _merge_metrics_payload
from train_platform.api.v2.training_runs import list_training_run_epoch_metrics


class MetricsPipelineTests(unittest.TestCase):
    def test_apply_metric_aliases_adds_unified_keys(self) -> None:
        raw = {
            "AP50": 0.62,
            "mAP": 0.41,
            "precision": 0.7,
            "recall": 0.55,
        }
        out = _apply_metric_aliases(raw)
        self.assertAlmostEqual(out["metrics/mAP50(B)"], 0.62)
        self.assertAlmostEqual(out["metrics/mAP50-95(B)"], 0.41)
        self.assertAlmostEqual(out["metrics/precision(B)"], 0.7)
        self.assertAlmostEqual(out["metrics/recall(B)"], 0.55)
        # Raw keys should still be preserved.
        self.assertAlmostEqual(out["AP50"], 0.62)
        self.assertAlmostEqual(out["mAP"], 0.41)

    def test_merge_metrics_payload_keeps_existing_and_overrides_duplicates(self) -> None:
        merged = _merge_metrics_payload(
            {"loss": 1.0, "metrics/mAP50(B)": 0.1},
            {"loss": 0.9, "lr": 0.001},
        )
        self.assertEqual(merged["loss"], 0.9)
        self.assertEqual(merged["lr"], 0.001)
        self.assertEqual(merged["metrics/mAP50(B)"], 0.1)

    @patch("train_platform.api.v2.training_runs.TrainingRunService.list_epoch_metrics")
    @patch("train_platform.api.v2.training_runs.fetch_mlflow_epoch_metrics")
    def test_metrics_source_auto_falls_back_to_db_when_mlflow_empty(
        self,
        mock_fetch_mlflow,
        mock_list_epoch_metrics,
    ) -> None:
        db_rows = [{"epoch": 0, "metrics": {"loss": 1.2}}]
        mock_fetch_mlflow.return_value = []
        mock_list_epoch_metrics.return_value = db_rows

        got = list_training_run_epoch_metrics("run-1", limit=100, source="auto", db=object())
        self.assertEqual(got, db_rows)
        mock_list_epoch_metrics.assert_called_once()

    @patch("train_platform.api.v2.training_runs.TrainingRunService.list_epoch_metrics")
    @patch("train_platform.api.v2.training_runs.fetch_mlflow_epoch_metrics")
    def test_metrics_source_mlflow_does_not_fallback_when_empty(
        self,
        mock_fetch_mlflow,
        mock_list_epoch_metrics,
    ) -> None:
        mock_fetch_mlflow.return_value = []
        got = list_training_run_epoch_metrics("run-1", limit=100, source="mlflow", db=object())
        self.assertEqual(got, [])
        mock_list_epoch_metrics.assert_not_called()


if __name__ == "__main__":
    unittest.main()

