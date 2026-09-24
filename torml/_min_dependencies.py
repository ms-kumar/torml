"""Module to get minimum dependency versions.

For version checking in tests and validation.
"""

from __future__ import annotations


def get_min_version(package: str) -> tuple[int, int, int]:
    """Get minimum version tuple for a package from pyproject.toml.

    Parameters
    ----------
    package : str
        Package name to find in pyproject.toml dependencies.

    Returns
    -------
    tuple of int
        Version as (major, minor, patch) tuple.

    Raises
    ------
    ValueError
        If package not found in pyproject.toml.
    """
    from importlib.resources import files

    toml_file = files("torml").joinpath("pyproject.toml").as_file()
    content = toml_file.read()

    import re

    # Find dependency line: package>=X.Y.Z or package>=X.Y.*
    match = re.search(rf"{package}\s*=\s*['\"]([^'\"]+)['\"]", content)
    if not match:
        raise ValueError(f"Package {package} not found in pyproject.toml")

    version_str = match.group(1)
    # Handle .* wildcard
    version_match = re.match(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?", version_str)

    major = int(version_match.group(1))
    minor = int(version_match.group(2)) if version_match.group(2) else 0
    patch = int(version_match.group(3)) if version_match.group(3) else 0

    return (major, minor, patch)


PYTHON_MIN_VERSION = get_min_version("python_version")
TORCH_MIN_VERSION = get_min_version("torch")
PYTEST_MIN_VERSION = get_min_version("pytest")
