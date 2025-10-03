"""pytest-llm: A pytest plugin for testing LLM outputs with success rate thresholds."""

import dataclasses

import pytest
from _pytest import nodes
from _pytest.reports import TestReport
from _pytest.runner import runtestprotocol

from . import _version

__version__ = _version.version
VERSION = _version.version_tuple


@dataclasses.dataclass
class LLMMarker:
    """Marker data for LLM tests."""

    prompt: str
    min_success_rate: float = 0.9  # Default to 90% success rate
    num_synthetic_prompts: int = 2

    def __post_init__(self):
        if not (0.0 <= self.min_success_rate <= 1.0):
            raise ValueError("min_success_rate must be between 0.0 and 1.0")


def pytest_configure(config):
    """Register the llm marker."""
    config.addinivalue_line(
        "markers",
        "llm(prompt, success_rate): mark test to run multiple times with LLM, "
        "requiring only a given success rate (0.0-1.0)",
    )


def pytest_generate_tests(metafunc: pytest.Metafunc):
    if llm_marker := metafunc.definition.get_closest_marker("llm"):
        marker = LLMMarker(*llm_marker.args)
        if "prompt" in metafunc.fixturenames:
            if fns := metafunc.definition.ihook.pytest_llm_complete(
                config=metafunc.config
            ):
                fn = fns[0]
                metafunc.parametrize(
                    "prompt",
                    [
                        marker.prompt,
                        *(
                            fn(
                                marker.prompt,
                                system="You MUST ONLY repeat the user statement but with different words while maintaining its meaning.",
                            )
                            for _ in range(marker.num_synthetic_prompts)
                        ),
                    ],
                )


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item: nodes.Item, nextitem: nodes.Item):
    """Run LLM tests multiple times based on success rate."""
    if llm_marker := item.get_closest_marker("llm"):
        llm_marker_data = LLMMarker(*llm_marker.args)
    else:
        # Not an LLM test, run normally - return None to let other hooks handle it
        return None

    # For LLM tests, we handle the execution ourselves
    # Calculate number of runs - use at least 10 for meaningful statistics
    min_runs = 10
    num_runs = max(min_runs, int(10 / max(1 - llm_marker_data.min_success_rate, 0.1)))

    # Initialize tracking
    item.llm_test_results = []
    ihook = item.ihook

    # Run the test multiple times
    passed_count = 0
    failed_count = 0

    for _ in range(num_runs):
        # Execute the test
        reports = runtestprotocol(item, nextitem=nextitem, log=False)

        # Check the outcome of the 'call' phase
        for report in reports:
            if report.when == "call":
                if report.passed:
                    passed_count += 1
                else:
                    failed_count += 1
                item.llm_test_results.append(report)
                break

    # Calculate success rate
    total_runs = passed_count + failed_count
    actual_success_rate = passed_count / total_runs if total_runs > 0 else 0

    # Determine overall outcome
    if actual_success_rate >= llm_marker_data.min_success_rate:
        # Create a passing report
        final_outcome = "passed"
    else:
        # Create a failing report
        final_outcome = "failed"

    # Store summary info
    item.llm_summary = {
        "passed": passed_count,
        "failed": failed_count,
        "total": total_runs,
        "actual_rate": actual_success_rate,
        "required_rate": llm_marker_data.min_success_rate,
        "outcome": final_outcome,
    }

    # Report the aggregated result
    # Create custom reports for setup, call, and teardown
    for when in ["setup", "call", "teardown"]:
        if when == "call":
            # Main call phase - use our aggregated result
            if final_outcome == "passed":
                report = TestReport(
                    nodeid=item.nodeid,
                    location=item.location,
                    keywords=item.keywords,
                    outcome="passed",
                    longrepr=None,
                    when="call",
                )
            else:
                longrepr = (
                    f"LLM test failed: {passed_count}/{total_runs} passed "
                    f"({actual_success_rate:.1%}), required {llm_marker_data.min_success_rate:.1%}"
                )
                report = TestReport(
                    nodeid=item.nodeid,
                    location=item.location,
                    keywords=item.keywords,
                    outcome="failed",
                    longrepr=longrepr,
                    when="call",
                )
        else:
            # Setup and teardown always pass for now
            report = TestReport(
                nodeid=item.nodeid,
                location=item.location,
                keywords=item.keywords,
                outcome="passed",
                longrepr=None,
                when=when,
            )

        ihook.pytest_runtest_logreport(report=report)

    # Return True to indicate we handled this test
    return True


def pytest_addhooks(pluginmanager):
    from . import hooks

    pluginmanager.add_hookspecs(hooks)
