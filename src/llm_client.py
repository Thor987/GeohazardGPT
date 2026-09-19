"""Provider-neutral OpenAI-compatible client for optional teacher generation.

The client reads credentials from the environment and contains no endpoint,
key, or account-specific configuration. It can be replaced by a local adapter
when a different provider is used.
"""

from __future__ import annotations

import os

from openai import OpenAI


class OpenAICompatibleClient:
    """Small chat-completion adapter for a user-supplied provider."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY or pass an API key explicitly.")
        base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)
        self.model = model

    def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if response.choices[0].finish_reason == "length":
            raise RuntimeError("Teacher response was truncated; increase max_tokens.")
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("The teacher model returned an empty response.")
        return content.strip()
