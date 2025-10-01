"""Edge case tests for the pytest-llm plugin."""
import pytest


def test_regular_test_without_marker():
    """Regular tests without the llm marker should work normally."""
    assert True


@pytest.mark.llm("Test prompt", 0.0)
def test_zero_success_rate(input, llm):
    """Test with 0% success rate - should pass even if all runs fail."""
    # This will always pass because 0% of runs need to succeed
    assert False


@pytest.mark.llm("Test prompt", 1.0)
def test_full_success_rate(input, llm):
    """Test with 100% success rate - all runs must pass."""
    # This will pass because the default LLM just returns the prompt
    assert len(llm.complete(input)) > 0
