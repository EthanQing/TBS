from __future__ import annotations

import os
import shutil
import stat
from pathlib import Path
from typing import Iterable


def _make_writable(path: str | os.PathLike[str]) -> None:
    try:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD | stat.S_IEXEC)
    except OSError:
        pass


def remove_tree(path: Path, *, ignore_errors: bool = False) -> None:
    target = Path(path)
    if target.is_symlink():
        target.unlink(missing_ok=True)
        return
    if not target.exists():
        return

    def _onerror(func, raw_path, _exc_info):
        _make_writable(raw_path)
        func(raw_path)

    try:
        shutil.rmtree(target, onerror=_onerror)
    except OSError:
        if not ignore_errors:
            raise


def remove_path(path: Path) -> None:
    target = Path(path)
    if target.is_symlink() or target.is_file():
        target.unlink(missing_ok=True)
    else:
        remove_tree(target)


def clear_directory(path: Path) -> None:
    target = Path(path)
    remove_tree(target)
    target.mkdir(parents=True, exist_ok=True)


def overlay_tree(source: Path, destination: Path, *, skip_names: Iterable[str] | None = None) -> None:
    source = Path(source)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    skipped = {str(name).lower() for name in (skip_names or set())}
    for item in source.iterdir():
        if item.name.lower() in skipped:
            continue
        target = destination / item.name
        if item.is_dir() and not item.is_symlink():
            overlay_tree(item, target, skip_names=skip_names)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def copy_tree(source: Path, destination: Path) -> None:
    clear_directory(Path(destination))
    overlay_tree(Path(source), Path(destination))


def merge_tree(source: Path, destination: Path, *, skip_names: Iterable[str] | None = None) -> None:
    """Copy missing files from source into destination without overwriting."""
    source = Path(source)
    destination = Path(destination)
    if not source.is_dir():
        return
    destination.mkdir(parents=True, exist_ok=True)
    skipped = {str(name).lower() for name in (skip_names or set())}
    for item in source.iterdir():
        if item.name.lower() in skipped:
            continue
        target = destination / item.name
        if item.is_dir() and not item.is_symlink():
            merge_tree(item, target, skip_names=skip_names)
        elif not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
