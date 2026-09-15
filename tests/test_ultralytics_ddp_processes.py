"""Real Linux process checks, runnable without the optional training dependencies."""

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import types
from contextlib import ExitStack
from unittest.mock import patch

import psutil


RUNTIME_DIR = Path(__file__).resolve().parents[1] / "train_platform/platform/runtime"
MODULE_PATH = RUNTIME_DIR / "ultralytics_ddp.py"
SCOPE_PATH = RUNTIME_DIR / "process_scope.py"
try:
    from train_platform.platform.runtime import process_scope
except ImportError:
    train_platform_package = sys.modules.setdefault("train_platform", types.ModuleType("train_platform"))
    platform_package = sys.modules.setdefault("train_platform.platform", types.ModuleType("train_platform.platform"))
    runtime_package = sys.modules.setdefault("train_platform.platform.runtime", types.ModuleType("train_platform.platform.runtime"))
    train_platform_package.__path__ = []
    platform_package.__path__ = []
    runtime_package.__path__ = []
    scope_spec = importlib.util.spec_from_file_location("train_platform.platform.runtime.process_scope", SCOPE_PATH)
    process_scope = importlib.util.module_from_spec(scope_spec)
    sys.modules[scope_spec.name] = process_scope
    scope_spec.loader.exec_module(process_scope)
    runtime_package.process_scope = process_scope
SPEC = importlib.util.spec_from_file_location("ddp_process_test_runtime", MODULE_PATH)
runtime = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runtime
SPEC.loader.exec_module(runtime)

LAUNCHER = r'''
import json, os, pathlib, subprocess, sys, time
import psutil
context = json.loads(pathlib.Path(sys.argv[1]).read_text())
mode = sys.argv[2]
event = {"type": "epoch_metrics", "run_id": context["run_id"],
         "attempt_id": context["attempt_id"], "epoch": 0,
         "metrics": {"train/box_loss": 1.25, "lr/pg0": 0.001}}
line = json.dumps(event) + "\n"
with open(context["metrics_path"], "a") as stream:
    stream.write(line[:20]); stream.flush()
    time.sleep(0.1)
    stream.write(line[20:]); stream.flush()
if mode == "success":
    sys.exit(0)
rank_code = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); time.sleep(60)"
rank = subprocess.Popen([sys.executable, "-c", rank_code])
identity = {"pid": rank.pid, "create_time": psutil.Process(rank.pid).create_time(),
            "pgid": os.getpgid(rank.pid), "role": "rank", "rank": 0,
            "run_id": context["run_id"], "attempt_id": context["attempt_id"],
            "execution_owner": context["execution_owner"]}
directory = pathlib.Path(context["processes_dir"])
temporary = directory / "rank-0.json.tmp"
temporary.write_text(json.dumps(identity))
temporary.replace(directory / "rank-0.json")
time.sleep(0.6)
if mode == "failure":
    sys.exit(7)
time.sleep(60)
'''


@unittest.skipUnless(sys.platform == "linux", "requires Linux process groups")
class DistributedProcessTests(unittest.TestCase):
    def run_supervisor(self, mode, cancel=None, fault=None):
        temporary = tempfile.TemporaryDirectory(prefix="tbs-ddp-test-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        owner = {
            "guard_pid": os.getpid(),
            "guard_create_time": psutil.Process().create_time(),
            "worker_id": "process-test",
            "process_scope": process_scope.get_process_scope(),
        }
        context = {
            "run_id": "process-test", "run_root": str(root),
            "runtime_dir": str(root / "runtime"), "output_dir": str(root / "output"),
            "world_size": 2, "cuda_visible_devices": "2,5",
            "execution_owner": owner,
        }
        events = []
        real_popen = subprocess.Popen
        launchers = []

        def launch(command, **kwargs):
            self.assertEqual(command[:3], [sys.executable, "-m", "torch.distributed.run"])
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "2,5")
            process = real_popen([sys.executable, "-c", LAUNCHER, command[-1], mode], **kwargs)
            launchers.append(process)
            return process

        def emit(epoch, metrics):
            if fault == "metrics":
                raise OSError("metric callback failed")
            events.append((epoch, metrics))

        failure = None
        started = time.monotonic()
        with ExitStack() as patches:
            patches.enter_context(patch.object(runtime.subprocess, "Popen", side_effect=launch))
            if fault == "registration":
                patches.enter_context(patch.object(runtime, "register_process", side_effect=OSError("registry failed")))
            try:
                runtime.run_ultralytics_ddp(
                    context,
                    cancel_requested=lambda: cancel(started) if cancel else False,
                    upsert_epoch_metrics=emit,
                    poll_seconds=0.05,
                )
            except (runtime.UltralyticsDDPError, runtime.UltralyticsDDPCancelled, OSError) as exc:
                failure = exc
        self.assertLess(time.monotonic() - started, 15)
        for launcher in launchers:
            self.assertIsNotNone(launcher.poll(), "launcher was not reaped")
        if fault is None:
            self.assertEqual(events, [(0, {"train/box_loss": 1.25, "lr/pg0": 0.001})])
        identities = [json.loads(path.read_text()) for path in root.glob("runtime/ddp/*/processes/*.json")]
        try:
            for identity in identities:
                try:
                    process = psutil.Process(identity["pid"])
                    same = process.create_time() == identity["create_time"]
                    self.assertFalse(same and process.status() != psutil.STATUS_ZOMBIE, identity)
                except psutil.NoSuchProcess:
                    pass
        finally:
            runtime.terminate_registered_processes(root, run_id="process-test", owner=owner, grace_seconds=0)
        return failure, identities

    def test_success_drains_last_metric(self):
        failure, _ = self.run_supervisor("success")
        self.assertIsNone(failure)

    def test_launcher_failure_cleans_registered_rank_and_observed_child(self):
        failure, identities = self.run_supervisor("failure")
        self.assertIsInstance(failure, runtime.UltralyticsDDPError)
        self.assertIn("7", str(failure))
        self.assertGreaterEqual(len({item["pid"] for item in identities}), 3)

    def test_cancellation_cleans_entire_attempt(self):
        failure, _ = self.run_supervisor("cancel", lambda started: time.monotonic() - started > 0.8)
        self.assertIsInstance(failure, runtime.UltralyticsDDPCancelled)

    def test_external_signal_is_failure_and_restores_handler(self):
        previous = signal.getsignal(signal.SIGTERM)
        sent = False

        def terminate(started):
            nonlocal sent
            if not sent and time.monotonic() - started > 0.8:
                sent = True
                os.kill(os.getpid(), signal.SIGTERM)
            return False

        failure, _ = self.run_supervisor("cancel", terminate)
        self.assertIsInstance(failure, runtime.UltralyticsDDPError)
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous)

    def test_registration_failure_still_reaps_launcher(self):
        failure, _ = self.run_supervisor("cancel", fault="registration")
        self.assertIsInstance(failure, OSError)

    def test_metric_callback_failure_still_cleans_processes(self):
        failure, _ = self.run_supervisor("cancel", fault="metrics")
        self.assertIsInstance(failure, OSError)

    def test_cleanup_rejects_other_execution_and_reused_pid(self):
        with tempfile.TemporaryDirectory(prefix="tbs-ddp-owner-") as directory:
            root = Path(directory)
            process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
            try:
                attempt = root / "runtime/ddp/attempt"
                attempt.mkdir(parents=True)
                local_scope = process_scope.get_process_scope()
                context = {"run_id": "run", "attempt_id": "attempt", "execution_owner": {"guard_pid": 11, "process_scope": local_scope}}
                (attempt / "context.json").write_text(json.dumps(context))
                identity = runtime.process_identity(process.pid, **context)
                runtime.register_process(attempt / "processes", "rank-0", **identity)
                runtime.terminate_registered_processes(root, run_id="run", owner={"guard_pid": 22, "process_scope": local_scope}, grace_seconds=0)
                self.assertIsNone(process.poll())
                self.assertIsNone(runtime._matching_process({**identity, "create_time": identity["create_time"] - 1}))
                runtime.register_process(attempt / "processes", "rank-0", **{**identity, "attempt_id": "old"})
                runtime.terminate_registered_processes(root, run_id="run", owner={"guard_pid": 11, "process_scope": local_scope}, grace_seconds=0)
                self.assertIsNone(process.poll())
            finally:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
