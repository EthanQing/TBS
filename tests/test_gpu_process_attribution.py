from train_platform.platform.runtime import gpu_processes


GPU = "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
SCOPE = {"boot_id": "boot", "pid_namespace": {"device": 1, "inode": 2}}
OWNER = {"allocation_id": "a", "guard_pid": 10, "guard_create_time": 11.0, "process_scope": SCOPE}


def registration(**kwargs):
    return dict(allocation_id="a", pid=10, create_time=11.0, start_ticks=100,
                clock_ticks_per_second=100, boot_time=10.0, process_scope=SCOPE,
                execution_owner=OWNER, assigned_gpu_uuids=[GPU], **kwargs)


def attribute(monkeypatch, registrations, rows=None, scope=SCOPE):
    monkeypatch.setattr(gpu_processes, "_proc_identity", lambda root, pid: {
        "start_ticks": 100, "nspid": [500, 10], "process_scope": scope,
    })
    return gpu_processes.attribute_driver_processes(
        rows or [{"driver_pid": 500, "memory_used_mib": 14336}], registrations,
        host_proc_root="/host/proc", gpu_uuid=GPU, active_owners={"a": OWNER},
    )


def test_host_pid_maps_through_namespace_and_exact_start_ticks(monkeypatch):
    result = attribute(monkeypatch, [registration()])
    assert result.complete
    assert result.usage_by_allocation == {"a": 14336}


def test_same_pid_wrong_namespace_is_not_attributed(monkeypatch):
    other = {"boot_id": "boot", "pid_namespace": {"device": 1, "inode": 3}}
    assert attribute(monkeypatch, [registration()], scope=other).usage_by_allocation == {}


def test_reused_pid_with_different_ticks_is_not_attributed(monkeypatch):
    item = registration()
    item["start_ticks"] = 99
    assert attribute(monkeypatch, [item]).usage_by_allocation == {}


def test_wrong_gpu_and_previous_allocation_cannot_receive_credit(monkeypatch):
    item = registration()
    item["assigned_gpu_uuids"] = ["GPU-other"]
    assert attribute(monkeypatch, [item]).usage_by_allocation == {}
    item = registration()
    item["execution_owner"] = {**OWNER, "allocation_id": "old"}
    assert attribute(monkeypatch, [item]).usage_by_allocation == {}


def test_duplicate_driver_and_registry_entries_are_not_added_twice(monkeypatch):
    row = {"driver_pid": 500, "memory_used_mib": 14336}
    result = attribute(monkeypatch, [registration(), registration()], [row, row])
    assert result.usage_by_allocation == {"a": 14336}


def test_no_host_view_never_guesses_pid_mapping():
    result = gpu_processes.attribute_driver_processes(
        [{"driver_pid": 10, "memory_used_mib": 14336}], [registration()],
        host_proc_root=None, gpu_uuid=GPU, active_owners={"a": OWNER},
    )
    assert not result.complete
    assert result.usage_by_allocation == {}
