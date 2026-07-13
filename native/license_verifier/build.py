from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the native Train Platform license verifier.")
    parser.add_argument("--output", type=Path, required=True, help="Protected runtime root.")
    parser.add_argument("--enforce", action="store_true", help="Compile out the runtime bypass switch.")
    args = parser.parse_args()

    command = ["cargo", "build", "--manifest-path", str(ROOT / "Cargo.toml"), "--locked", "--release"]
    if args.enforce:
        command.extend(["--features", "enforce-license"])

    env = os.environ.copy()
    env["PYO3_PYTHON"] = sys.executable
    subprocess.run(command, check=True, env=env)

    source_name = "license.dll" if os.name == "nt" else "liblicense.so"
    source = ROOT / "target" / "release" / source_name
    if not source.is_file():
        raise SystemExit(f"Native license artifact not found: {source}")

    suffix = str(sysconfig.get_config_var("EXT_SUFFIX") or (".pyd" if os.name == "nt" else ".so"))
    destination_dir = args.output.resolve() / "train_platform" / "core"
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"license{suffix}"
    shutil.copy2(source, destination)

    plaintext = destination_dir / "license.py"
    if plaintext.exists():
        plaintext.unlink()

    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
