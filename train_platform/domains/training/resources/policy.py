from __future__ import annotations

from sqlalchemy.orm import Session

from train_platform.models.v3.gpu_allocation import GpuAllocation, GpuNodeSchedulingState


def apply_node_policy(db: Session, policy: dict) -> dict:
    """Apply an explicit, complete managed policy in the caller's transaction."""
    fields = {
        "node_id", "managed", "accepting_allocations", "shared_execution_enabled",
        "max_shared_tasks_per_device", "memory_safety_mib",
    }
    if set(policy) != fields:
        raise ValueError("Node policy must contain exactly the complete policy fields")
    node_id = policy["node_id"]
    if not isinstance(node_id, str) or not node_id.strip() or len(node_id) > 128:
        raise ValueError("node_id must contain 1 to 128 characters")
    for name in ("managed", "accepting_allocations"):
        if policy[name] is not True:
            raise ValueError(f"{name} must be enabled to apply managed policy")
    if not isinstance(policy["shared_execution_enabled"], bool):
        raise ValueError("shared_execution_enabled must be a boolean")
    for name, minimum in (("max_shared_tasks_per_device", 1), ("memory_safety_mib", 0)):
        value = policy[name]
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= 2147483647:
            raise ValueError(f"{name} must be an integer between {minimum} and 2147483647")

    node = db.query(GpuNodeSchedulingState).filter_by(node_id=node_id).with_for_update().first()
    # This must be the first consistent read after acquiring the node lock.
    # Locking allocations here would reverse the allocator's lock order.
    active = db.query(GpuAllocation.allocation_id).filter(
        GpuAllocation.node_id == node_id,
        GpuAllocation.state != "released",
    ).first()
    if active is not None:
        raise ValueError("Target node has unfinished allocations; policy was not changed")
    if node is None:
        node = GpuNodeSchedulingState(node_id=node_id)
        db.add(node)
    for name, value in policy.items():
        setattr(node, name, value)
    return dict(policy)
