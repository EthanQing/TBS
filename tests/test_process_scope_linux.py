"""Linux scope checks runnable with only the Python standard library."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "train_platform/platform/runtime/process_scope.py"
SPEC = importlib.util.spec_from_file_location("linux_scope_test", MODULE_PATH)
scope = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scope)


@unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux procfs")
class ProcessScopeTests(unittest.TestCase):
    def test_scope_matches_kernel_and_namespace(self):
        actual = scope.get_process_scope()
        namespace = os.stat("/proc/self/ns/pid")
        self.assertEqual(actual, {
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "pid_namespace": {"device": namespace.st_dev, "inode": namespace.st_ino},
        })

    def test_child_in_different_pid_namespace_has_different_scope(self):
        unshare = shutil.which("unshare")
        if unshare is None:
            self.skipTest("unshare is unavailable")
        command = [unshare]
        if os.geteuid() != 0:
            command.extend(["--user", "--map-root-user"])
        # Mount the child's procfs as well, so /proc/self refers to its own PID namespace.
        command.extend(["--pid", "--fork", "--mount-proc", sys.executable, str(Path(__file__).resolve()), "--scope-child"])
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        if result.returncode and "not permitted" in result.stderr.lower():
            self.skipTest("environment does not permit creating a PID namespace")
        self.assertEqual(result.returncode, 0, result.stderr)
        child_scope = json.loads(result.stdout)
        parent_scope = scope.get_process_scope()
        self.assertEqual(child_scope["boot_id"], parent_scope["boot_id"])
        self.assertNotEqual(child_scope["pid_namespace"], parent_scope["pid_namespace"])
        self.assertEqual(scope.compare_process_scope(child_scope, parent_scope), "different")


if __name__ == "__main__":
    if sys.argv[1:] == ["--scope-child"]:
        print(json.dumps(scope.get_process_scope()))
    else:
        unittest.main()
