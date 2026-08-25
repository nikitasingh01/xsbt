"""Packaging checks, so a clean clone installs and runs the same way this one did.

Nothing here imports the package. These are checks on the files a reviewer or a CI
runner reads before any code executes.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def pinned() -> dict[str, str]:
    """constraints.txt as a lookup, normalised the way pip compares names."""
    out = {}
    for line in (ROOT / "constraints.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        name, _, rest = line.partition("==")
        out[normalise(name)] = rest.split(";")[0].strip()
    return out


def normalise(name: str) -> str:
    """PEP 503: underscores, dots and hyphens are the same character to pip."""
    return re.sub(r"[-_.]+", "-", name).lower()


def declared(pyproject: dict) -> list[Requirement]:
    project = pyproject["project"]
    raw = [*project["dependencies"], *project["optional-dependencies"]["dev"]]
    return [Requirement(r) for r in raw]


def test_every_declared_dependency_is_pinned(pyproject: dict, pinned: dict[str, str]) -> None:
    """The failure this catches is adding a dependency and forgetting the lock.

    It would not break anything on the machine that added it, which is exactly why it
    survives to the reviewer, who then cannot reproduce the numbers.
    """
    missing = [r.name for r in declared(pyproject) if normalise(r.name) not in pinned]

    assert not missing, f"declared in pyproject.toml but not pinned: {missing}"


def test_the_pins_satisfy_the_ranges(pyproject: dict, pinned: dict[str, str]) -> None:
    """Two files, one answer. A pin outside its own range means one of them is stale."""
    for requirement in declared(pyproject):
        version = pinned[normalise(requirement.name)]
        assert requirement.specifier.contains(version, prereleases=True), (
            f"{requirement.name} is pinned at {version}, outside {requirement.specifier}"
        )


def test_the_pinned_set_is_a_full_closure(pyproject: dict, pinned: dict[str, str]) -> None:
    """Direct dependencies alone do not pin a build; their dependencies move too.

    A rough count is enough here. The real check is the pinned CI job, which installs
    against this file and fails if it no longer resolves.
    """
    direct = {normalise(r.name) for r in declared(pyproject)}

    assert len(pinned) > 2 * len(direct)
    for transitive in ("python-dateutil", "pytz", "urllib3", "markupsafe"):
        assert transitive in pinned


def test_docker_and_ci_install_against_the_same_file() -> None:
    """A lockfile nobody installs from is documentation, not a lockfile."""
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "-c constraints.txt" in dockerfile
    assert "-c constraints.txt" in workflow


def test_the_version_matrix_still_installs_unpinned() -> None:
    """The matrix exists to prove the ranges resolve. Pinning it would test nothing."""
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    matrix, _, _ = workflow.partition("  pinned:")

    assert "python-version:" in matrix
    assert "-c constraints.txt" not in matrix
