"""pytest-llm: A pytest plugin for testing LLM outputs with success rate thresholds."""

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # Python < 3.8
    from importlib_metadata import version, PackageNotFoundError

try:
    __version__ = version("pytest-llm")
except PackageNotFoundError:
    # Package is not installed
    __version__ = "unknown"
