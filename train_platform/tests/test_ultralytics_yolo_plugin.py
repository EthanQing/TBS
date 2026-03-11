from __future__ import annotations

import types
import unittest
import shutil
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import patch

from train_platform.training.plugins.ultralytics_yolo import UltralyticsYOLOTrainer


class _FakeModelBase:
    init_paths: list[str] = []
    train_kwargs: list[dict] = []

    def __init__(self, model_path: str) -> None:
        self.__class__.init_paths.append(str(model_path))

    def add_callback(self, _event: str, _fn) -> None:
        return None

    def train(self, **kwargs) -> None:
        self.__class__.train_kwargs.append(dict(kwargs))

    def val(self) -> None:
        return None

    @classmethod
    def reset(cls) -> None:
        cls.init_paths = []
        cls.train_kwargs = []


class _FakeYOLO(_FakeModelBase):
    pass


class _FakeRTDETR(_FakeModelBase):
    pass


class UltralyticsYOLOPluginTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeYOLO.reset()
        _FakeRTDETR.reset()
        self.root = Path.cwd() / "temp_vdl_test" / f"ultra_plugin_{uuid4().hex}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.dataset_dir = self.root / "dataset"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        (self.dataset_dir / "data.yaml").write_text("train: images/train\nval: images/val\nnames: ['a']\n", encoding="utf-8")

        self.settings = SimpleNamespace(
            training_dir=self.root / "training",
            temp_dir=self.root / "temp",
            pretrain_models_dir=self.root / "pretrain",
        )
        self.settings.training_dir.mkdir(parents=True, exist_ok=True)
        self.settings.temp_dir.mkdir(parents=True, exist_ok=True)
        self.settings.pretrain_models_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _build_ctx(self, *, variant: str, add: dict | None = None, run_id: str = "run-1"):
        params = SimpleNamespace(
            epochs=5,
            batch_size=2,
            learning_rate=0.01,
            image_size=640,
            optimizer="AdamW",
            workers=2,
            patience=20,
            device="cpu",
            use_pretrained=True,
            additional_params=add or {},
        )
        arch = SimpleNamespace(variant=variant, family="YOLO")
        dataset = SimpleNamespace(name="dummy-dataset")
        project = SimpleNamespace(dataset=dataset)
        job = SimpleNamespace(parameters=params, architecture=arch, project=project)
        run_dir = self.root / "runs" / run_id
        return SimpleNamespace(
            job_id=run_id,
            job=job,
            dataset_path=self.dataset_dir,
            run_dir=run_dir,
            cancel_requested=lambda: False,
            upsert_epoch_metrics=lambda _epoch, _metrics: None,
        )

    def _patch_runtime_modules(self):
        fake_torch = types.ModuleType("torch")
        fake_torch.cuda = SimpleNamespace(is_available=lambda: False)

        fake_ultralytics = types.ModuleType("ultralytics")
        fake_ultralytics.YOLO = _FakeYOLO
        fake_ultralytics.RTDETR = _FakeRTDETR
        fake_ultralytics.settings = SimpleNamespace(update=lambda *_args, **_kwargs: None)

        return patch.dict(
            "sys.modules",
            {
                "torch": fake_torch,
                "ultralytics": fake_ultralytics,
            },
            clear=False,
        )

    def test_yolo_pretrained_defaults_to_pt(self) -> None:
        trainer = UltralyticsYOLOTrainer()
        ctx = self._build_ctx(variant="yolo11n", add={"use_pretrained": True})

        with self._patch_runtime_modules(), patch(
            "train_platform.training.plugins.ultralytics_yolo.settings", self.settings
        ), patch(
            "train_platform.training.plugins.ultralytics_yolo._apply_torch_safe_load_patches", return_value=None
        ), patch("train_platform.training.plugins.ultralytics_yolo._ensure_amp_check_weight", return_value=True):
            trainer.run(ctx)

        self.assertTrue(_FakeYOLO.init_paths[-1].endswith("yolo11n.pt"))
        args = _FakeYOLO.train_kwargs[-1]
        self.assertEqual(args.get("pretrained"), True)
        self.assertIn("momentum", args)
        self.assertIn("warmup_epochs", args)

    def test_yolo_without_pretrained_uses_yaml(self) -> None:
        trainer = UltralyticsYOLOTrainer()
        ctx = self._build_ctx(variant="yolo12n", add={"use_pretrained": False})

        with self._patch_runtime_modules(), patch(
            "train_platform.training.plugins.ultralytics_yolo.settings", self.settings
        ), patch(
            "train_platform.training.plugins.ultralytics_yolo._apply_torch_safe_load_patches", return_value=None
        ), patch("train_platform.training.plugins.ultralytics_yolo._ensure_amp_check_weight", return_value=True):
            trainer.run(ctx)

        self.assertTrue(_FakeYOLO.init_paths[-1].endswith("yolo12n.yaml"))
        args = _FakeYOLO.train_kwargs[-1]
        self.assertEqual(args.get("pretrained"), False)

    def test_rtdetr_uses_rtdetr_loader_and_filtered_args(self) -> None:
        trainer = UltralyticsYOLOTrainer()
        ctx = self._build_ctx(variant="rtdetr-l", add={"use_pretrained": True})

        with self._patch_runtime_modules(), patch(
            "train_platform.training.plugins.ultralytics_yolo.settings", self.settings
        ), patch(
            "train_platform.training.plugins.ultralytics_yolo._apply_torch_safe_load_patches", return_value=None
        ), patch("train_platform.training.plugins.ultralytics_yolo._ensure_amp_check_weight", return_value=True):
            trainer.run(ctx)

        self.assertTrue(_FakeRTDETR.init_paths[-1].endswith("rtdetr-l.pt"))
        args = _FakeRTDETR.train_kwargs[-1]
        self.assertNotIn("momentum", args)
        self.assertNotIn("warmup_bias_lr", args)
        self.assertIn("weight_decay", args)

    def test_resume_path_takes_precedence_over_pretrained_path(self) -> None:
        trainer = UltralyticsYOLOTrainer()
        resume_job_id = "resume-source"
        resume_weights = self.settings.training_dir / resume_job_id / "weights" / "last.pt"
        resume_weights.parent.mkdir(parents=True, exist_ok=True)
        resume_weights.write_bytes(b"pt")

        explicit_pretrain = self.root / "custom.pt"
        explicit_pretrain.write_bytes(b"pt")
        ctx = self._build_ctx(
            variant="yolo26n",
            add={
                "resume_training": True,
                "resume_job_id": resume_job_id,
                "pretrained_model_path": str(explicit_pretrain),
                "use_pretrained": True,
            },
            run_id="run-resume",
        )

        with self._patch_runtime_modules(), patch(
            "train_platform.training.plugins.ultralytics_yolo.settings", self.settings
        ), patch(
            "train_platform.training.plugins.ultralytics_yolo._apply_torch_safe_load_patches", return_value=None
        ), patch("train_platform.training.plugins.ultralytics_yolo._ensure_amp_check_weight", return_value=True):
            trainer.run(ctx)

        self.assertEqual(_FakeYOLO.init_paths[-1], str(resume_weights))
        args = _FakeYOLO.train_kwargs[-1]
        self.assertEqual(args.get("resume"), True)
        self.assertNotIn("pretrained", args)

    def test_disable_amp_when_probe_weights_unavailable(self) -> None:
        trainer = UltralyticsYOLOTrainer()
        ctx = self._build_ctx(variant="yolo11s", add={"use_pretrained": False, "amp": True})

        with self._patch_runtime_modules(), patch(
            "train_platform.training.plugins.ultralytics_yolo.settings", self.settings
        ), patch(
            "train_platform.training.plugins.ultralytics_yolo._apply_torch_safe_load_patches", return_value=None
        ), patch("train_platform.training.plugins.ultralytics_yolo._ensure_amp_check_weight", return_value=False):
            trainer.run(ctx)

        args = _FakeYOLO.train_kwargs[-1]
        self.assertEqual(args.get("amp"), False)


if __name__ == "__main__":
    unittest.main()
