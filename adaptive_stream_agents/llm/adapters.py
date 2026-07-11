from __future__ import annotations

import json
import os
from urllib import error, request
from typing import Protocol


class LLMAdapter(Protocol):
    """Interface for replacing deterministic reasoning with a real LLM."""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        ...


class LocalTemplateLLM:
    """Small local stand-in used when no external LLM is configured."""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        return f"{system_prompt.strip()} {user_prompt.strip()}".strip()


class DeterministicJsonLLM:
    """Offline LLM stand-in that returns JSON for tests and demos."""

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        data = json.loads(user_prompt)
        baseline = data["deterministic_baseline"]
        task = data["task"]
        if baseline["accepted"]:
            opening = f"I can host the {task['stream_type']} task based on my local state."
        else:
            opening = f"I cannot safely host the {task['stream_type']} task under my local constraints."
        return json.dumps(
            {
                "accepted": baseline["accepted"],
                "confidence": baseline["confidence"],
                "rationale": baseline["rationale"],
                "natural_language": f"{opening} Reasoning: {baseline['rationale']}.",
            }
        )


class OpenAIChatLLM:
    """OpenAI chat adapter. Requires the optional `openai` package and OPENAI_API_KEY."""

    def __init__(self, model: str) -> None:
        self.model = model

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install the optional dependency `openai` to use OpenAIChatLLM.") from exc

        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_API_KEY before using OpenAIChatLLM.")

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        return response.choices[0].message.content or "{}"


class OllamaChatLLM:
    """Ollama chat adapter for local or Ollama cloud-routed models."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434") -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.2},
        }
        data = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            f"{self.base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=120) as response:
                body = json.loads(response.read().decode("utf-8"))
        except error.URLError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.base_url}. "
                "Start Ollama or change --ollama-url."
            ) from exc

        message = body.get("message", {})
        content = message.get("content")
        if not content:
            raise RuntimeError(f"Ollama returned no message content for model {self.model}.")
        return content
