from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.build_core_checkpoint_release import SCRIPT_FILES

ROOT = Path(__file__).resolve().parents[1]
CHECK_ENTRYPOINT_ORIGIN = """
import runpy
import sys
from pathlib import Path

entrypoint = Path(sys.argv[1])
sys.argv = [str(entrypoint), "--help"]
try:
    runpy.run_path(str(entrypoint), run_name="__main__")
except SystemExit as exit_status:
    if exit_status.code != 0:
        raise

import tsunami_surrogate
from tsunami_surrogate.utils import hashing

root = entrypoint.parents[1]
assert Path(tsunami_surrogate.__file__).resolve() == root / "tsunami_surrogate/__init__.py"
assert Path(hashing.__file__).resolve() == root / "src/utils/hashing.py"
"""


@pytest.fixture(scope="module")
def exported_source(tmp_path_factory):
    checkout = tmp_path_factory.mktemp("checkout")
    shutil.copytree(
        ROOT / "src", checkout / "src", ignore=shutil.ignore_patterns("__pycache__")
    )
    package = checkout / "tsunami_surrogate"
    package.mkdir()
    shutil.copyfile(ROOT / "tsunami_surrogate/__init__.py", package / "__init__.py")
    scripts = checkout / "scripts"
    scripts.mkdir()
    for script in (*SCRIPT_FILES, "build_core_checkpoint_release"):
        entrypoint = f"{script}.py"
        shutil.copyfile(ROOT / "scripts" / entrypoint, scripts / entrypoint)
    return checkout


@pytest.mark.parametrize(
    "script, option",
    [
        ("build_core_checkpoint_release", "--replay"),
        ("eval_accuracy", "--checkpoint"),
        ("eval_calibration", "--val-dataset"),
        ("eval_pooled_reference_ablation", "--reference-dataset"),
        ("eval_reference_analysis", "--output-dir"),
        ("eval_seed_metrics", "--expected-seeds"),
    ],
)
def test_python_entrypoints_import_from_uninstalled_checkout(
    exported_source, tmp_path, script, option
):
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            CHECK_ENTRYPOINT_ORIGIN,
            str(exported_source / "scripts" / f"{script}.py"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert option in result.stdout
