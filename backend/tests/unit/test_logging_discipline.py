"""Library discipline: importing the agent must never configure logging.

Python's logging guidance reserves handler configuration for the application
entrypoint. `research_system` is both a library (imported by `app`) and an
application (its own CLI), so the rule is enforced by call site rather than by
package: only entrypoints may call `configure_logging`.
"""

import ast
import subprocess
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2] / "research_system"
ENTRYPOINTS = {"cli.py"}


def _modules_calling_configure_logging() -> set[str]:
    offenders = set()
    for path in PACKAGE.rglob("*.py"):
        if path.name in ENTRYPOINTS or path.name == "logging.py":
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = getattr(func, "id", None) or getattr(func, "attr", None)
                if name == "configure_logging":
                    offenders.add(path.name)
    return offenders


def test_no_library_module_calls_configure_logging() -> None:
    """Only entrypoints configure logging. Agents and nodes just get a logger."""
    assert _modules_calling_configure_logging() == set()


def test_importing_the_package_installs_no_handler() -> None:
    """A bare import must not attach a handler to the root logger."""
    code = (
        "import logging, research_system; "
        "print(len([h for h in logging.getLogger().handlers "
        "if not isinstance(h, logging.NullHandler)]))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "0"
