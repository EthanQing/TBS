from __future__ import annotations

import base64

import pytest

from train_platform.core.license import LicenseError, assert_valid_license, license_required


LICENSE_ENV = (
    "TRAIN_PLATFORM_LICENSE_REQUIRED",
    "TRAIN_PLATFORM_LICENSE_PATH",
    "TRAIN_PLATFORM_LICENSE_DATA",
    "TRAIN_PLATFORM_LICENSE_DATA_B64",
)


def _clear_license_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in LICENSE_ENV:
        monkeypatch.delenv(name, raising=False)


def test_development_mode_can_disable_license(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_license_env(monkeypatch)
    monkeypatch.setenv("TRAIN_PLATFORM_LICENSE_REQUIRED", "0")

    assert license_required() is False
    assert assert_valid_license() is None


@pytest.mark.parametrize("value", ["1", "true", "yes", "y", "on", "TRUE"])
def test_license_required_accepts_enabled_values(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    _clear_license_env(monkeypatch)
    monkeypatch.setenv("TRAIN_PLATFORM_LICENSE_REQUIRED", value)

    assert license_required() is True


def test_required_license_rejects_missing_file(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _clear_license_env(monkeypatch)
    monkeypatch.setenv("TRAIN_PLATFORM_LICENSE_REQUIRED", "1")
    monkeypatch.setenv("TRAIN_PLATFORM_LICENSE_PATH", str(tmp_path / "missing.dat"))

    with pytest.raises(LicenseError, match="License file not found"):
        assert_valid_license()


def test_base64_input_has_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_license_env(monkeypatch)
    monkeypatch.setenv("TRAIN_PLATFORM_LICENSE_REQUIRED", "1")
    monkeypatch.setenv("TRAIN_PLATFORM_LICENSE_DATA", "not selected")
    monkeypatch.setenv(
        "TRAIN_PLATFORM_LICENSE_DATA_B64",
        base64.b64encode(b"not json").decode("ascii"),
    )

    with pytest.raises(LicenseError, match="Invalid license file format"):
        assert_valid_license()


def test_invalid_base64_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_license_env(monkeypatch)
    monkeypatch.setenv("TRAIN_PLATFORM_LICENSE_REQUIRED", "1")
    monkeypatch.setenv("TRAIN_PLATFORM_LICENSE_DATA_B64", "not base64")

    with pytest.raises(LicenseError, match="Invalid base64 value"):
        assert_valid_license()
