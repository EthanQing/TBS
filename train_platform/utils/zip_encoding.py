from __future__ import annotations

import os
import unicodedata
import zipfile
from pathlib import Path

from train_platform.utils.exceptions import ValidationError


_UTF8_FLAG = 0x800


def _candidate_encodings() -> tuple[str, ...]:
    raw = os.getenv("ZIP_FILENAME_ENCODINGS", "utf-8,gbk,cp936,big5,cp437")
    items = [item.strip() for item in str(raw or "").split(",") if item.strip()]
    out: list[str] = []
    for item in items:
        key = item.lower()
        if key not in {x.lower() for x in out}:
            out.append(item)
    return tuple(out or ["utf-8", "gbk", "cp936", "cp437"])


def decode_zip_member_name(info: zipfile.ZipInfo) -> str:
    name = str(info.filename or "")
    if info.flag_bits & _UTF8_FLAG:
        return unicodedata.normalize("NFC", name)

    try:
        raw = name.encode("cp437")
    except UnicodeEncodeError:
        return unicodedata.normalize("NFC", name)

    for encoding in _candidate_encodings():
        try:
            decoded = raw.decode(encoding)
        except UnicodeDecodeError:
            continue
        if decoded:
            return unicodedata.normalize("NFC", decoded)
    return unicodedata.normalize("NFC", name)


def safe_zip_member_relpath(info: zipfile.ZipInfo) -> Path:
    raw_name = decode_zip_member_name(info).replace("\\", "/")
    if not raw_name or raw_name.startswith("/"):
        raise ValidationError("ZIP contains an unsafe absolute path")
    if "\x00" in raw_name:
        raise ValidationError("ZIP contains an unsafe null byte path")
    rel = Path(raw_name)
    if rel.is_absolute() or ".." in rel.parts or (rel.parts and ":" in rel.parts[0]):
        raise ValidationError("ZIP contains an unsafe path")
    return rel
