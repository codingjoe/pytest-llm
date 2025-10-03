import pytest


@pytest.mark.llm("How many R's are in the Word 'Strawberry'?", 0.9)
def test_counting(prompt, llm):
    result = llm(prompt).lower()
    assert ("3" in result) or ("three" in result)
