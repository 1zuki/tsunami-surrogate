"""Expose the source tree under the package name used by project scripts."""

from pathlib import Path

# Keep the existing source layout while supporting imports from a source checkout.
__path__ = [str(Path(__file__).resolve().parent.parent / "src")]
