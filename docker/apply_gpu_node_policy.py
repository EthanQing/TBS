"""Apply the deployment's explicit GPU node policy inside the backend container."""
from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping

from sqlalchemy.exc import SQLAlchemyError

from train_platform.db.session import SessionLocal
from train_platform.models.v3.gpu_allocation import GpuAllocation, GpuNodeSchedulingState


def read_policy(environment: Mapping[str, str]) -> dict:
    node_id = environment.get("GPU_NODE_ID", "").strip()
    if not node_id or len(node_id) > 128:
        raise ValueError("GPU_NODE_ID must contain 1 to 128 characters")

    def boolean(name: str) -> bool:
        value = environment.get(name, "").strip().lower()
        if value not in {"true", "false", "1", "0"}:
            raise ValueError(f"{name} must be true, false, 1 or 0")
        return value in {"true", "1"}

    def integer(name: str, minimum: int) -> int:
        raw = environment.get(name, "").strip()
        if not raw.isascii() or not raw.isdecimal():
            raise ValueError(f"{name} must be an integer >= {minimum}")
        value = int(raw)
        if not minimum <= value <= 2147483647:
            raise ValueError(f"{name} must be between {minimum} and 2147483647")
        return value

    if not boolean("GPU_SCHEDULER_ENABLED"):
        raise ValueError("GPU_SCHEDULER_ENABLED must be enabled to apply managed policy")
    return {
        "node_id": node_id,
        "managed": True,
        "accepting_allocations": True,
        "shared_execution_enabled": boolean("GPU_SHARED_EXECUTION_ENABLED"),
        "max_shared_tasks_per_device": integer("GPU_MAX_SHARED_TASKS_PER_DEVICE", 1),
        "memory_safety_mib": integer("GPU_MEMORY_SAFETY_MIB", 0),
    }


def apply_policy(policy: dict, session_factory=SessionLocal) -> dict:
    with session_factory() as db, db.begin():
        node = db.query(GpuNodeSchedulingState).filter_by(
            node_id=policy["node_id"]
        ).with_for_update().first()
        # This must be the first consistent read after acquiring the node lock.
        # Locking allocations here would reverse the allocator's lock order.
        active = db.query(GpuAllocation.allocation_id).filter(
            GpuAllocation.node_id == policy["node_id"],
            GpuAllocation.state != "released",
        ).first()
        if active is not None:
            raise ValueError("Target node has unfinished allocations; policy was not changed")
        if node is None:
            node = GpuNodeSchedulingState(node_id=policy["node_id"])
            db.add(node)
        for name, value in policy.items():
            setattr(node, name, value)
    return dict(policy)


def main() -> int:
    try:
        policy = read_policy(os.environ)
        result = apply_policy(policy)
    except ValueError as exc:
        print(f"GPU node policy rejected: {exc}", file=sys.stderr)
        return 1
    except SQLAlchemyError:
        # Database exceptions may include connection credentials or SQL values.
        print("GPU node policy failed: database transaction rolled back; check database availability and concurrent policy changes", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
