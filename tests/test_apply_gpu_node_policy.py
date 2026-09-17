import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from train_platform.models.v3 import V3Base
from train_platform.models.v3.gpu_allocation import GpuAllocation, GpuNodeSchedulingState

SCRIPT = Path(__file__).resolve().parents[1] / 'docker' / 'apply_gpu_node_policy.py'
spec = importlib.util.spec_from_file_location('apply_gpu_node_policy', SCRIPT)
policy_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(policy_script)
ENV = {
    'GPU_NODE_ID': 'existing-node',
    'GPU_SCHEDULER_ENABLED': 'true',
    'GPU_SHARED_EXECUTION_ENABLED': 'true',
    'GPU_MAX_SHARED_TASKS_PER_DEVICE': '2',
    'GPU_MEMORY_SAFETY_MIB': '4096',
}


@pytest.fixture
def sessions(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'policy.db'}")
    V3Base.metadata.create_all(engine)
    yield sessionmaker(bind=engine)
    engine.dispose()


def test_create_update_idempotent_and_node_isolation(sessions):
    policy = policy_script.read_policy(ENV)
    assert policy_script.apply_policy(policy, sessions) == policy
    with sessions.begin() as db:
        db.add(GpuNodeSchedulingState(node_id='other', managed=False))
        node = db.get(GpuNodeSchedulingState, 'existing-node')
        node.managed = False
        node.accepting_allocations = False
        node.memory_safety_mib = 0
    policy_script.apply_policy(policy, sessions)
    policy_script.apply_policy(policy, sessions)
    with sessions() as db:
        assert db.query(GpuNodeSchedulingState).count() == 2
        node = db.get(GpuNodeSchedulingState, 'existing-node')
        assert {name: getattr(node, name) for name in policy} == policy
        assert not db.get(GpuNodeSchedulingState, 'other').managed


def add_allocation(sessions, state, node_id='existing-node'):
    now = datetime.now(timezone.utc)
    with sessions.begin() as db:
        db.add(GpuAllocation(allocation_id='allocation', run_id='run',
            worker_instance_id='instance', worker_id='worker', node_id=node_id,
            state=state, request_snapshot={}, reserved_at=now, launch_deadline_at=now))


@pytest.mark.parametrize('state', ['reserved', 'starting', 'running', 'releasing', 'unknown'])
@pytest.mark.parametrize('existing', [False, True])
def test_unfinished_allocation_rejects_even_identical_policy(sessions, state, existing):
    policy = policy_script.read_policy(ENV)
    if existing:
        policy_script.apply_policy(policy, sessions)
    add_allocation(sessions, state)
    with pytest.raises(ValueError, match='unfinished allocations'):
        policy_script.apply_policy(policy, sessions)
    with sessions() as db:
        assert db.query(GpuNodeSchedulingState).count() == int(existing)
        assert db.get(GpuAllocation, 'allocation').state == state


@pytest.mark.parametrize(('state', 'node_id'), [('released', 'existing-node'), ('running', 'other')])
def test_released_or_other_node_does_not_block(sessions, state, node_id):
    add_allocation(sessions, state, node_id)
    policy_script.apply_policy(policy_script.read_policy(ENV), sessions)
    with sessions() as db:
        assert db.get(GpuNodeSchedulingState, 'existing-node').managed
        assert db.get(GpuAllocation, 'allocation').state == state


@pytest.mark.parametrize(('name', 'value'), [
    ('GPU_NODE_ID', ''), ('GPU_NODE_ID', 'x' * 129),
    ('GPU_SCHEDULER_ENABLED', 'false'), ('GPU_SCHEDULER_ENABLED', 'yes'),
    ('GPU_SHARED_EXECUTION_ENABLED', 'maybe'),
    ('GPU_MAX_SHARED_TASKS_PER_DEVICE', '0'),
    ('GPU_MAX_SHARED_TASKS_PER_DEVICE', '2.5'),
    ('GPU_MEMORY_SAFETY_MIB', '-1'), ('GPU_MEMORY_SAFETY_MIB', '2147483648'),
])
def test_invalid_environment(name, value):
    with pytest.raises(ValueError, match=name):
        policy_script.read_policy({**ENV, name: value})


def test_missing_parameters_rejected():
    for name in ENV:
        with pytest.raises(ValueError, match=name):
            policy_script.read_policy({key: value for key, value in ENV.items() if key != name})


def test_zero_safety_and_disabled_shared_are_valid():
    policy = policy_script.read_policy({**ENV, 'GPU_MEMORY_SAFETY_MIB': '0',
                                       'GPU_SHARED_EXECUTION_ENABLED': 'false'})
    assert policy['memory_safety_mib'] == 0
    assert policy['shared_execution_enabled'] is False


def test_commit_failure_rolls_back(sessions):
    policy = policy_script.read_policy(ENV)
    policy_script.apply_policy(policy, sessions)
    def fail_flush(session, context, instances):
        raise OperationalError('secret SQL', {}, Exception('secret password'))
    event.listen(sessions, 'before_flush', fail_flush)
    try:
        with pytest.raises(OperationalError):
            policy_script.apply_policy({**policy, 'memory_safety_mib': 1}, sessions)
    finally:
        event.remove(sessions, 'before_flush', fail_flush)
    with sessions() as db:
        assert db.get(GpuNodeSchedulingState, 'existing-node').memory_safety_mib == 4096


def test_database_error_does_not_print_credentials(monkeypatch, capsys):
    for name, value in ENV.items():
        monkeypatch.setenv(name, value)
    def fail(policy):
        raise OperationalError('secret SQL', {}, Exception('secret password'))
    monkeypatch.setattr(policy_script, 'apply_policy', fail)
    assert policy_script.main() == 1
    output = capsys.readouterr()
    assert 'rolled back' in output.err
    assert 'secret' not in output.err
    assert not output.out


def test_stdin_entrypoint_rejects_invalid_config_before_database_access():
    result = subprocess.run([sys.executable, '-'], input=SCRIPT.read_text(encoding='utf-8'),
        text=True, capture_output=True, cwd=SCRIPT.parents[1],
        env={**os.environ, **ENV, 'GPU_NODE_ID': ''})
    assert result.returncode == 1
    assert 'GPU_NODE_ID' in result.stderr
    assert not result.stdout


def test_main_prints_only_policy(monkeypatch, capsys, sessions):
    for name, value in ENV.items():
        monkeypatch.setenv(name, value)
    apply = policy_script.apply_policy
    monkeypatch.setattr(policy_script, 'apply_policy', lambda policy: apply(policy, sessions))
    assert policy_script.main() == 0
    assert json.loads(capsys.readouterr().out) == policy_script.read_policy(ENV)
