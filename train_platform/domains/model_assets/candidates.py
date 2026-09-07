from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from sqlalchemy.orm import Session

from train_platform.domains.model_assets.runtime import resolve_architecture_config_path
from train_platform.models.v3.architecture import ModelArchitecture
from train_platform.models.v3.enums import TrainingRunStatus
from train_platform.models.v3.model_registry import ModelVersion
from train_platform.models.v3.training_run import TrainingRun
from train_platform.schemas.v3.inference_jobs import InferenceModelCandidate
from train_platform.platform.filesystem.locations import resolve_training_path


class ModelCandidateService:
    """Build model candidates shared by inference and evaluation APIs."""

    def _weights_ext_ok(self, engine: str, weights_path: Path) -> bool:
        ext = weights_path.suffix.lower()
        if engine == "paddle-det":
            return ext == ".pdparams"
        return ext in {".pt", ".pth"}

    def _build_candidate(
        self,
        *,
        source: str,
        model_version: ModelVersion | None,
        run: TrainingRun,
        arch: ModelArchitecture | None,
    ) -> Optional[InferenceModelCandidate]:
        engine = str(getattr(arch, "engine", "") or "ultralytics-yolo").strip().lower()
        family = str(getattr(arch, "family", "") or "").strip() or None
        variant = str(getattr(arch, "variant", "") or "").strip() or None

        weights_rel = None
        if model_version and model_version.weights_path:
            weights_rel = model_version.weights_path
        elif run.result:
            weights_rel = run.result.best_weights_path or run.result.last_weights_path
        if not weights_rel:
            return None

        weights_abs = resolve_training_path(weights_rel)
        if not weights_abs.exists() or not weights_abs.is_file():
            return None
        if not self._weights_ext_ok(engine, weights_abs):
            return None

        config_path = None
        if engine == "paddle-det":
            cfg = resolve_architecture_config_path(arch)
            if cfg is None:
                return None
            config_path = str(cfg)

        label_parts = []
        if family:
            label_parts.append(family)
        if variant:
            label_parts.append(variant)
        if model_version and model_version.version:
            label_parts.append(f"v:{model_version.version}")
        else:
            label_parts.append(f"run:{str(run.run_id)[:8]}")
        label = " / ".join(label_parts)

        created_at = None
        if model_version and getattr(model_version, "created_at", None):
            created_at = model_version.created_at
        elif getattr(run, "finished_at", None):
            created_at = run.finished_at
        else:
            created_at = getattr(run, "created_at", None)

        return InferenceModelCandidate(
            source="model_version" if source == "model_version" else "training_run",
            model_version_id=int(model_version.model_version_id) if model_version else None,
            run_id=str(run.run_id),
            project_id=int(run.project_id),
            architecture_id=int(run.architecture_id),
            engine=engine,
            family=family,
            variant=variant,
            version=str(model_version.version) if model_version else None,
            label=label,
            weights_path=str(weights_rel),
            config_path=config_path,
            inferable=True,
            created_at=created_at,
        )

    def list_inferable_models(self, db: Session, *, project_id: int | None = None) -> List[InferenceModelCandidate]:
        q_runs = db.query(TrainingRun).filter(TrainingRun.status == TrainingRunStatus.COMPLETED)
        if project_id is not None:
            q_runs = q_runs.filter(TrainingRun.project_id == int(project_id))
        runs = q_runs.all()
        run_map = {str(r.run_id): r for r in runs}
        arch_ids = {int(r.architecture_id) for r in runs}
        arch_rows = []
        if arch_ids:
            arch_rows = db.query(ModelArchitecture).filter(ModelArchitecture.architecture_id.in_(sorted(arch_ids))).all()
        arch_map = {int(a.architecture_id): a for a in arch_rows}

        q_mvs = db.query(ModelVersion)
        if project_id is not None:
            q_mvs = q_mvs.filter(ModelVersion.project_id == int(project_id))
        mvs = q_mvs.order_by(ModelVersion.created_at.desc()).all()

        out: List[InferenceModelCandidate] = []
        run_with_mv: set[str] = set()

        for mv in mvs:
            run = run_map.get(str(mv.run_id))
            if not run:
                continue
            arch = arch_map.get(int(run.architecture_id))
            cand = self._build_candidate(source="model_version", model_version=mv, run=run, arch=arch)
            if not cand:
                continue
            out.append(cand)
            run_with_mv.add(str(run.run_id))

        for run in runs:
            rid = str(run.run_id)
            if rid in run_with_mv:
                continue
            arch = arch_map.get(int(run.architecture_id))
            cand = self._build_candidate(source="training_run", model_version=None, run=run, arch=arch)
            if not cand:
                continue
            out.append(cand)

        out.sort(key=lambda x: x.created_at or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
        return out


__all__ = ["ModelCandidateService"]
