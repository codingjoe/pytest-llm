"""Pytest plugin for LLM reliability testing."""
import pytest
from _pytest.runner import runtestprotocol
from _pytest.reports import TestReport


class LLMMarker:
    """Marker data for LLM tests."""
    
    def __init__(self, prompt: str, success_rate: float):
        self.prompt = prompt
        self.success_rate = success_rate
        
        if not 0 <= success_rate <= 1:
            raise ValueError(f"success_rate must be between 0 and 1, got {success_rate}")


class LLM:
    """Default LLM implementation for testing."""
    
    def complete(self, prompt: str) -> str:
        """
        Complete the given prompt.
        
        This is a mock implementation that returns the prompt itself.
        Users should override this fixture with their own LLM implementation.
        """
        return prompt


@pytest.fixture
def llm():
    """
    Fixture providing an LLM instance.
    
    This default implementation returns a mock LLM.
    Users can override this fixture in their conftest.py to provide
    their own LLM implementation.
    """
    return LLM()


@pytest.fixture
def input(request):
    """
    Fixture providing the input prompt for LLM tests.
    
    For tests marked with @pytest.mark.llm, this will be the prompt
    specified in the marker.
    """
    llm_marker = request.node.get_closest_marker("llm")
    if llm_marker and len(llm_marker.args) >= 1:
        return llm_marker.args[0]
    return None


def pytest_configure(config):
    """Register the llm marker."""
    config.addinivalue_line(
        "markers",
        "llm(prompt, success_rate): mark test to run multiple times with LLM, "
        "requiring only a given success rate (0.0-1.0)"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test items to handle llm marker."""
    for item in items:
        llm_marker = item.get_closest_marker("llm")
        if llm_marker:
            # Extract marker arguments
            if len(llm_marker.args) < 2:
                raise ValueError(
                    f"llm marker requires 2 arguments (prompt, success_rate), "
                    f"got {len(llm_marker.args)}"
                )
            
            prompt = llm_marker.args[0]
            success_rate = llm_marker.args[1]
            
            # Store marker data on the item
            item.llm_marker_data = LLMMarker(prompt, success_rate)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_protocol(item, nextitem):
    """Run LLM tests multiple times based on success rate."""
    llm_marker_data = getattr(item, 'llm_marker_data', None)
    
    if llm_marker_data is None:
        # Not an LLM test, run normally - return None to let other hooks handle it
        return None
    
    # For LLM tests, we handle the execution ourselves
    # Calculate number of runs - use at least 10 for meaningful statistics
    min_runs = 10
    num_runs = max(min_runs, int(10 / max(1 - llm_marker_data.success_rate, 0.1)))
    
    # Initialize tracking
    item.llm_test_results = []
    ihook = item.ihook
    
    # Run the test multiple times
    passed_count = 0
    failed_count = 0
    
    for run_num in range(num_runs):
        # Execute the test
        reports = runtestprotocol(item, nextitem=nextitem, log=False)
        
        # Check the outcome of the 'call' phase
        for report in reports:
            if report.when == 'call':
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
    if actual_success_rate >= llm_marker_data.success_rate:
        # Create a passing report
        final_outcome = 'passed'
    else:
        # Create a failing report
        final_outcome = 'failed'
    
    # Store summary info
    item.llm_summary = {
        'passed': passed_count,
        'failed': failed_count,
        'total': total_runs,
        'actual_rate': actual_success_rate,
        'required_rate': llm_marker_data.success_rate,
        'outcome': final_outcome
    }
    
    # Report the aggregated result
    # Create custom reports for setup, call, and teardown
    for when in ['setup', 'call', 'teardown']:
        if when == 'call':
            # Main call phase - use our aggregated result
            if final_outcome == 'passed':
                report = TestReport(
                    nodeid=item.nodeid,
                    location=item.location,
                    keywords=item.keywords,
                    outcome='passed',
                    longrepr=None,
                    when='call'
                )
            else:
                longrepr = (
                    f"LLM test failed: {passed_count}/{total_runs} passed "
                    f"({actual_success_rate:.1%}), required {llm_marker_data.success_rate:.1%}"
                )
                report = TestReport(
                    nodeid=item.nodeid,
                    location=item.location,
                    keywords=item.keywords,
                    outcome='failed',
                    longrepr=longrepr,
                    when='call'
                )
        else:
            # Setup and teardown always pass for now
            report = TestReport(
                nodeid=item.nodeid,
                location=item.location,
                keywords=item.keywords,
                outcome='passed',
                longrepr=None,
                when=when
            )
        
        ihook.pytest_runtest_logreport(report=report)
    
    # Return True to indicate we handled this test
    return True
