"""Test the pytest-llm plugin."""

import pytest


@pytest.mark.llm("Tell me a joke.", 0.9)
def test_dadjoke(prompt, llm):
    assert "yo mama" not in llm(prompt).lower(), "Expected a joke without 'yo mama'."


@pytest.mark.llm("What is 2+2?", 1.0)
def test_always_pass(prompt, llm):
    assert "wrong" not in llm(prompt).lower()


@pytest.mark.llm("Generate a number", 1)
def test_partial_success(prompt, llm):
    assert "42" in llm(prompt)


@pytest.mark.llm("How many R's are in the word 'Strawberry'?")
def test_counting(prompt, llm):
    assert "three" in llm(prompt)
