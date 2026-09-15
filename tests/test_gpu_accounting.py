from datetime import datetime, timedelta, timezone

from train_platform.domains.training.resources.accounting import calculate_gpu_accounting


NOW = datetime.now(timezone.utc)


def snapshot(used=16384, free=32768):
    return dict(memory_total_mib=49152, memory_used_mib=used,
                memory_free_mib=free, sampled_at=NOW)


def calculate(data=None, usage=None, **kwargs):
    return calculate_gpu_accounting(
        data or snapshot(), [{"allocation_id": "a", "reserved_memory_mib": 20480}],
        4096, reliable_usage_by_allocation=usage,
        process_mapping_complete=usage is not None, now=NOW, **kwargs,
    )


def test_verified_usage_preserves_external_usage_and_full_promise():
    result = calculate(usage={"a": 14336})
    assert result.external_used_mib == 2048
    assert result.committed_mib == 22528
    assert result.available_budget_mib == 22528
    assert result.admits(18432)


def test_unknown_usage_does_not_credit_actual_memory():
    result = calculate()
    assert result.status == "conservative"
    assert result.external_used_mib == 16384
    assert result.available_budget_mib == 8192
    assert not result.admits(18432)


def test_free_memory_independently_limits_new_budget():
    result = calculate(snapshot(free=16384), usage={"a": 14336})
    assert result.available_budget_mib == 6144


def test_over_budget_execution_keeps_actual_usage():
    result = calculate(snapshot(used=30000, free=19152), usage={"a": 25000})
    assert result.committed_mib == 30000
    assert result.external_used_mib == 5000
    assert result.available_budget_mib == 15056


def test_inconsistent_attribution_discards_all_credit():
    result = calculate(usage={"a": 17000})
    assert result.status == "conservative"
    assert result.available_budget_mib == calculate().available_budget_mib


def test_fractional_credit_is_rounded_down():
    assert calculate(usage={"a": 14336.9}).available_budget_mib == calculate(usage={"a": 14336}).available_budget_mib


def test_stale_or_missing_fields_cannot_admit():
    data = snapshot()
    data["sampled_at"] = NOW - timedelta(seconds=21)
    assert calculate(data).status == "stale"
    assert not calculate(data).admits(1)
    data["memory_free_mib"] = None
    assert calculate(data).status == "unavailable"
