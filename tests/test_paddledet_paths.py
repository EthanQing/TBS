from pathlib import Path
from types import SimpleNamespace


def _make_repo(root: Path) -> Path:
    (root / "ppdet").mkdir(parents=True)
    (root / "ppdet" / "__init__.py").write_text("", encoding="utf-8")
    cfg = root / "configs" / "ppyoloe" / "ppyoloe_plus_crn_s_80e_coco.yml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("architecture: PPYOLOE\n", encoding="utf-8")
    return root


def test_resolve_paddledet_repo_accepts_complete_checkout(tmp_path, monkeypatch):
    from train_platform.utils import paddledet_paths

    repo = _make_repo(tmp_path / "PaddleDetection")
    monkeypatch.setattr(
        paddledet_paths,
        "settings",
        SimpleNamespace(paddle_det_dir=repo, home_dir=tmp_path),
    )

    assert paddledet_paths.resolve_paddledet_repo() == repo


def test_resolve_paddledet_repo_rejects_incomplete_checkout(tmp_path, monkeypatch):
    from train_platform.utils import paddledet_paths

    incomplete = tmp_path / "PaddleDetection"
    (incomplete / "configs").mkdir(parents=True)
    monkeypatch.setattr(
        paddledet_paths,
        "settings",
        SimpleNamespace(paddle_det_dir=incomplete, home_dir=tmp_path),
    )

    assert paddledet_paths.resolve_paddledet_repo() is None


def test_resolve_paddledet_config_path_uses_repo_relative_path(tmp_path, monkeypatch):
    from train_platform.utils import paddledet_paths

    repo = _make_repo(tmp_path / "PaddleDetection")
    monkeypatch.setattr(
        paddledet_paths,
        "settings",
        SimpleNamespace(paddle_det_dir=repo, home_dir=tmp_path),
    )

    resolved = paddledet_paths.resolve_paddledet_config_path(
        "configs/ppyoloe/ppyoloe_plus_crn_s_80e_coco.yml"
    )

    assert resolved == repo / "configs" / "ppyoloe" / "ppyoloe_plus_crn_s_80e_coco.yml"
