from pathlib import Path
from types import SimpleNamespace

import pytest

from train_platform.platform.runtime import process_scope


SCOPE = {"boot_id": "boot-a", "pid_namespace": {"device": 4, "inode": 12345}}


def test_linux_scope_reads_kernel_boot_and_namespace_identity(monkeypatch):
    monkeypatch.setattr(process_scope.sys, "platform", "linux")
    read_paths = []

    def read(path, **kwargs):
        read_paths.append(path.as_posix())
        return "boot-a\n"

    def stat(path):
        read_paths.append(str(path))
        return SimpleNamespace(st_dev=4, st_ino=12345)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(process_scope.os, "stat", stat)
    assert process_scope.get_process_scope() == SCOPE
    assert read_paths == ["/proc/sys/kernel/random/boot_id", "/proc/self/ns/pid"]


@pytest.mark.parametrize("failure", ["boot", "namespace"])
def test_unreadable_linux_scope_is_unknown(monkeypatch, failure):
    monkeypatch.setattr(process_scope.sys, "platform", "linux")

    def unreadable(*args, **kwargs):
        raise PermissionError("proc unavailable")

    monkeypatch.setattr(Path, "read_text", unreadable if failure == "boot" else lambda *args, **kwargs: "boot-a")
    monkeypatch.setattr(process_scope.os, "stat", unreadable)
    assert process_scope.get_process_scope() is None


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_unsupported_environment_does_not_invent_scope(monkeypatch, platform):
    monkeypatch.setattr(process_scope.sys, "platform", platform)
    assert process_scope.get_process_scope() is None


@pytest.mark.parametrize("recorded,current,expected", [
    (SCOPE, SCOPE, "same"),
    (SCOPE, {**SCOPE, "boot_id": "boot-b"}, "different"),
    (SCOPE, {**SCOPE, "pid_namespace": {"device": 4, "inode": 12346}}, "different"),
    (SCOPE, {**SCOPE, "pid_namespace": {"device": 5, "inode": 12345}}, "different"),
    (None, SCOPE, "missing"),
    ({"boot_id": "boot-a"}, SCOPE, "missing"),
    (SCOPE, None, "unavailable"),
])
def test_scope_comparison_requires_both_identifiers(recorded, current, expected):
    assert process_scope.compare_process_scope(recorded, current) == expected


def test_conflicting_direct_and_owner_scope_is_not_local():
    identity = {
        "process_scope": SCOPE,
        "execution_owner": {"process_scope": {**SCOPE, "boot_id": "boot-b"}},
    }
    assert process_scope.identity_process_scope(identity) is None
