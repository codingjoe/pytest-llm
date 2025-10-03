# pytest-llm

Fast AI reliability test suite

A pytest plugin that provides a custom marker for testing LLM (Large Language Model) outputs with configurable success rate thresholds.

## Usage

```python
import pytest

@pytest.mark.llm("How many R's are in the Word 'Strawberry'?", 0.9)
def test_counting(prompt, llm):
    result = llm(prompt).lower()
    assert ("3" in result) or ("three" in result)
```

## Setup

```bash
python3 -m pip install -e pytest-llm
```

```python
# conftest.py
import os
import typing

import httpx
import ollama
import pytest


def github_models_complete(prompt, model=None, system=None) -> str:
    messages = [{"role": "system", "content": system}] if system else []
    messages += [{"role": "user", "content": prompt}]
    if GITHUB_TOKEN := os.getenv("GITHUB_TOKEN"):
        response = httpx.post(
            "https://models.github.ai/inference/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {GITHUB_TOKEN}",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            json={
                "model": model or "openai/gpt-5-nano",
                "messages": messages,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    else:
        return ollama.generate(model="llama3.2", prompt=prompt).response


@pytest.fixture
def llm() -> typing.Callable[[str], str]:
    # This fixture isn't needed but might be convenient
    return github_models_complete


def pytest_llm_complete(config):
    # This pytest hook will provide enable the plugin to generate random prompts
    return github_models_complete
```

## Running Tests

```bash
pytest -m llm
# or to run non-llm tests
pytest -m "not llm"
```
