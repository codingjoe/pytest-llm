# pytest-llm

Fast AI reliability test suite

A pytest plugin that provides a custom marker for testing LLM (Large Language Model) outputs with configurable success rate thresholds.

## Installation

```bash
pip install pytest-llm
```

## Usage

The plugin provides a `@pytest.mark.llm` marker that runs a test function multiple times and only requires a specified percentage of runs to pass.

### Basic Example

```python
import pytest

@pytest.mark.llm("Tell me a joke.", 0.9)
def test_dadjoke(input, llm):
    """Test passes if 90% of LLM responses don't contain inappropriate content."""
    result = llm.complete(input).lower()
    assert "yo mama" not in result
```

### How it Works

1. **Marker**: `@pytest.mark.llm(prompt, success_rate)`
   - `prompt`: The input text to send to the LLM
   - `success_rate`: Float between 0.0 and 1.0 indicating the minimum required success rate

2. **Fixtures**:
   - `input`: Automatically provides the prompt from the marker
   - `llm`: Provides an LLM instance (defaults to a mock, override in your `conftest.py`)

3. **Test Execution**:
   - The test runs multiple times (at least 10, more for higher success rates)
   - Results are aggregated
   - Test passes if actual success rate >= required success rate

### Custom LLM Implementation

Override the `llm` fixture in your `conftest.py` to use your own LLM:

```python
import pytest
from openai import OpenAI

class MyLLM:
    def __init__(self):
        self.client = OpenAI()
    
    def complete(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

@pytest.fixture
def llm():
    return MyLLM()
```

### Examples

```python
# Test that requires 100% success (all runs must pass)
@pytest.mark.llm("What is 2+2?", 1.0)
def test_math(input, llm):
    result = llm.complete(input)
    assert "4" in result

# Test that only needs 50% success rate
@pytest.mark.llm("Generate a random number", 0.5)
def test_random(input, llm):
    result = llm.complete(input)
    assert result.strip().isdigit()

# Test with lower success threshold for non-deterministic outputs
@pytest.mark.llm("Tell me a creative story", 0.7)
def test_creative(input, llm):
    result = llm.complete(input)
    assert len(result) > 50  # Story should be substantial
```

## Features

- **Probabilistic Testing**: Tests LLM outputs that may vary between runs
- **Configurable Success Rates**: Define acceptable failure rates per test
- **Automatic Retries**: Runs tests multiple times automatically
- **Clear Reporting**: Shows aggregated results (e.g., "13/25 passed (52.0%)")

## License

BSD 2-Clause License
