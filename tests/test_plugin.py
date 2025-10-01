"""Test the pytest-llm plugin."""
import pytest


@pytest.mark.llm("Tell me a joke.", 0.9)
def test_dadjoke(input, llm):
    """Test that verifies LLM doesn't generate inappropriate jokes."""
    result = llm.complete(input).lower()
    assert "yo mama" not in result


@pytest.mark.llm("What is 2+2?", 1.0)
def test_always_pass(input, llm):
    """Test that should always pass."""
    # The default LLM just returns the prompt, which doesn't contain "wrong"
    assert "wrong" not in llm.complete(input).lower()


@pytest.mark.llm("Generate a number", 0.5)
def test_partial_success(input, llm):
    """Test that only needs to pass 50% of the time."""
    # This will always pass with the default mock LLM
    result = llm.complete(input)
    assert isinstance(result, str)
