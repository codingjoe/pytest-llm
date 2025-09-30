# Examples

## Basic Usage

```python
import pytest

@pytest.mark.llm("Tell me a joke.", 0.9)
def test_joke(input, llm):
    """Test passes if 90% of responses meet the criteria."""
    result = llm.complete(input).lower()
    assert "yo mama" not in result
```

## Custom LLM Implementation

Create a `conftest.py` file in your test directory:

```python
import pytest
from openai import OpenAI

class OpenAILLM:
    def __init__(self, model="gpt-4"):
        self.client = OpenAI()
        self.model = model
    
    def complete(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content

@pytest.fixture
def llm():
    """Override the default LLM with OpenAI."""
    return OpenAILLM()
```

## Using Different Models

```python
import pytest

class MultiModelLLM:
    def __init__(self, provider="openai", model="gpt-4"):
        self.provider = provider
        self.model = model
        # Initialize based on provider
        if provider == "openai":
            from openai import OpenAI
            self.client = OpenAI()
        elif provider == "anthropic":
            from anthropic import Anthropic
            self.client = Anthropic()
    
    def complete(self, prompt: str) -> str:
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

@pytest.fixture
def llm(request):
    """Configurable LLM fixture."""
    # Get model from marker if specified
    marker = request.node.get_closest_marker("llm_config")
    if marker:
        provider = marker.kwargs.get("provider", "openai")
        model = marker.kwargs.get("model", "gpt-4")
    else:
        provider = "openai"
        model = "gpt-4"
    
    return MultiModelLLM(provider, model)
```

## Testing Different Success Rates

```python
import pytest

# Strict test - must always pass
@pytest.mark.llm("What is 2+2?", 1.0)
def test_deterministic(input, llm):
    assert "4" in llm.complete(input)

# Moderate test - 80% success rate
@pytest.mark.llm("Tell me a fact.", 0.8)
def test_factual(input, llm):
    result = llm.complete(input)
    assert len(result) > 10  # Should be a substantial fact

# Lenient test - 50% success rate
@pytest.mark.llm("Be creative!", 0.5)
def test_creative(input, llm):
    result = llm.complete(input)
    assert result is not None
```

## Testing Multiple Prompts

```python
import pytest

test_prompts = [
    ("What is the capital of France?", 0.95, "Paris"),
    ("What is 10 + 5?", 1.0, "15"),
    ("Name a color", 0.7, None),  # Just check format
]

@pytest.mark.parametrize("prompt,success_rate,expected", test_prompts)
def test_llm_knowledge(prompt, success_rate, expected, llm):
    """Note: parametrize doesn't work directly with @pytest.mark.llm"""
    # For parametrized tests, you'd need to call llm.complete directly
    # and implement your own retry logic, or create separate test functions
    result = llm.complete(prompt)
    if expected:
        assert expected.lower() in result.lower()
```

## Advanced: Custom Retry Logic

```python
import pytest

@pytest.fixture
def llm_with_retry():
    """LLM with built-in retry logic."""
    class RetryLLM:
        def __init__(self, max_retries=3):
            from openai import OpenAI
            self.client = OpenAI()
            self.max_retries = max_retries
        
        def complete(self, prompt: str) -> str:
            for attempt in range(self.max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "user", "content": prompt}]
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise
                    continue
    
    return RetryLLM()
```

## Testing for Specific Patterns

```python
import pytest
import re

@pytest.mark.llm("Generate a valid email address", 0.8)
def test_email_format(input, llm):
    result = llm.complete(input)
    # Check if result contains a valid email
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    assert re.search(email_pattern, result)

@pytest.mark.llm("Generate valid JSON", 0.9)
def test_json_output(input, llm):
    import json
    result = llm.complete(input)
    # Should be parseable JSON
    json.loads(result)  # Will raise if invalid
```
