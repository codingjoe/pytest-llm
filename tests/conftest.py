import os
import typing

import httpx
import ollama
import pytest


def github_models_complete(prompt, model=None, system=None) -> str:
    messages = [{"role": "system", "content": system}] if system else []
    messages += [{"role": "user", "content": prompt}]
    if GITHUB_API_KEY := os.getenv("GITHUB_API_KEY"):
        response = httpx.post(
            "https://models.github.ai/inference/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {GITHUB_API_KEY}",
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
    return github_models_complete
