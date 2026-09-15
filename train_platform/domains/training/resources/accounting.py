from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Mapping


ACTIVE_ALLOCATION_STATES = frozenset({"reserved", "starting", "running", "releasing"})


@dataclass(frozen=True)
class ActiveCommitment:
    allocation_id: str
    reserved_memory_mib: int


@dataclass(frozen=True)
class GpuAccountingResult:
    status: str
    total_mib: int | None
    used_mib: int | None
    free_mib: int | None
    safety_mib: int
    external_used_mib: int | None
    committed_mib: int | None
    reliable_training_used_mib: int | None
    reserved_mib: int
    available_budget_mib: int

    def admits(self, requested_mib: int) -> bool:
        return requested_mib > 0 and requested_mib <= self.available_budget_mib


def _aware(value: datetime | None) -> datetime | None:
    if value is None or value.tzinfo is not None:
        return value
    return value.replace(tzinfo=timezone.utc)


def calculate_gpu_accounting(
    snapshot: Mapping[str, object] | None,
    active_commitments: Iterable[ActiveCommitment | Mapping[str, object]],
    safety_mib: int,
    *,
    reliable_usage_by_allocation: Mapping[str, int | float | None] | None = None,
    process_mapping_complete: bool = False,
    process_mapping_has_duplicates: bool = False,
    stale_after_seconds: int = 20,
    now: datetime | None = None,
) -> GpuAccountingResult:
    commitments = [
        item if isinstance(item, ActiveCommitment) else ActiveCommitment(str(item["allocation_id"]), int(item["reserved_memory_mib"]))
        for item in active_commitments
    ]
    reserved = sum(max(0, item.reserved_memory_mib) for item in commitments)
    safety = max(0, int(safety_mib))
    if not snapshot:
        return GpuAccountingResult("unavailable", None, None, None, safety, None, None, None, reserved, 0)
    total = snapshot.get("memory_total_mib")
    used = snapshot.get("memory_used_mib")
    free = snapshot.get("memory_free_mib")
    sampled_at = _aware(snapshot.get("sampled_at")) if isinstance(snapshot.get("sampled_at"), datetime) else None
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in (total, used, free)):
        return GpuAccountingResult("unavailable", total if isinstance(total, int) else None, used if isinstance(used, int) else None, free if isinstance(free, int) else None, safety, None, None, None, reserved, 0)
    instant = _aware(now) or datetime.now(timezone.utc)
    if sampled_at is None or sampled_at > instant or instant - sampled_at > timedelta(seconds=stale_after_seconds):
        return GpuAccountingResult("stale", total, used, free, safety, None, None, None, reserved, 0)

    reliable: dict[str, int] = {}
    mapping_ok = process_mapping_complete and not process_mapping_has_duplicates
    if mapping_ok:
        for item in commitments:
            raw = (reliable_usage_by_allocation or {}).get(item.allocation_id)
            if raw is not None and math.isfinite(float(raw)) and float(raw) >= 0:
                reliable[item.allocation_id] = math.floor(float(raw))
        if sum(reliable.values()) > used:
            reliable.clear()
            mapping_ok = False
    reliable_total = sum(reliable.values())
    external = max(0, used - reliable_total)
    committed = external + sum(max(item.reserved_memory_mib, reliable.get(item.allocation_id, 0)) for item in commitments)
    unused_promises = sum(max(item.reserved_memory_mib - reliable.get(item.allocation_id, 0), 0) for item in commitments)
    by_total = total - safety - committed
    by_free = free - safety - unused_promises
    available = max(0, min(by_total, by_free))
    return GpuAccountingResult(
        "verified" if mapping_ok else "conservative",
        total,
        used,
        free,
        safety,
        external,
        committed,
        reliable_total if mapping_ok else None,
        reserved,
        available,
    )
