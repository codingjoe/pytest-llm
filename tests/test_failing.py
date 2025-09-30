"""Test the pytest-llm plugin with failing scenarios."""
import pytest
import random


# Set a seed for reproducibility
random.seed(42)


# Create a global counter to track calls across fixture instances
_call_counter = {'count': 0}


class SometimesFailingLLM:
    """LLM that fails based on a pattern."""
    
    def complete(self, prompt: str) -> str:
        """Return different responses based on global call count."""
        _call_counter['count'] += 1
        count = _call_counter['count']
        # Fail 50% of the time (every other call)
        if count % 2 == 0:
            return "Yo mama joke"
        return prompt


@pytest.fixture
def llm():
    """Override the default llm fixture."""
    return SometimesFailingLLM()


@pytest.mark.llm("Tell me a joke.", 0.6)
def test_should_fail_due_to_low_success_rate(input, llm):
    """
    This test should fail because it only passes 50% of the time,
    but requires 60% success rate.
    """
    result = llm.complete(input).lower()
    assert "yo mama" not in result


# Reset counter for next test
_call_counter2 = {'count': 0}


class AlwaysFailingLLM:
    """LLM that always fails."""
    
    def complete(self, prompt: str) -> str:
        """Always return inappropriate content."""
        return "Yo mama is so fat"


@pytest.fixture
def llm_always_fail():
    """Fixture that always fails."""
    return AlwaysFailingLLM()


@pytest.mark.llm("Tell me a joke.", 0.1)
def test_should_fail_always(input):
    """
    This test should fail because it never passes,
    and requires 10% success rate.
    """
    # Using AlwaysFailingLLM directly without a fixture
    llm_obj = AlwaysFailingLLM()
    result = llm_obj.complete(input).lower()
    assert "yo mama" not in result
