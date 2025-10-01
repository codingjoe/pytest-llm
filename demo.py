#!/usr/bin/env python
"""
Demonstration of the pytest-llm plugin.

This script shows how to use the @pytest.mark.llm decorator to test
LLM outputs with configurable success rate thresholds.
"""

import pytest


# Example 1: High success rate requirement (90%)
@pytest.mark.llm("Tell me a joke.", 0.9)
def test_joke_quality(input, llm):
    """This test requires 90% of responses to pass."""
    result = llm.complete(input).lower()
    assert "yo mama" not in result


# Example 2: Perfect success rate (100%)
@pytest.mark.llm("What is 2+2?", 1.0)
def test_math_answer(input, llm):
    """This test requires 100% of responses to pass."""
    result = llm.complete(input)
    # With default mock LLM, this will always pass
    assert len(result) > 0


# Example 3: Lower success rate (50%)
@pytest.mark.llm("Generate a creative response", 0.5)
def test_creative_output(input, llm):
    """This test only requires 50% of responses to pass."""
    result = llm.complete(input)
    # Very lenient check
    assert isinstance(result, str)


if __name__ == "__main__":
    # Run this file as a test
    pytest.main([__file__, "-v"])
