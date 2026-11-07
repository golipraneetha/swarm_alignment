"""Inference backends. Signatures checked against mlx-lm 0.32.2.

There is no automatic fallback: if MLX is unavailable the run fails rather than
quietly producing mock data. Mock output is for pipeline testing only and is
labelled as such in every record it produces.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Optional

MOCK_MODEL_ID = "MOCK-NOT-A-MODEL"


class BackendUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class GenerationResult:
    text: str
    prompt_tokens: Optional[int]
    generation_tokens: Optional[int]
    finish_reason: Optional[str]
    elapsed_s: float


class MLXBackend:
    name = "mlx"
    is_mock = False

    def __init__(self, model_id: str, offline: bool = True):
        if offline:
            # Must be set before huggingface_hub is imported.
            os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            import mlx.core as mx
            import mlx_lm
            from mlx_lm.generate import stream_generate
            from mlx_lm.sample_utils import make_sampler
        except ImportError as error:
            raise BackendUnavailable(f"MLX-LM is not importable: {error}. Use the project .venv.") from error

        self._mx = mx
        self._stream_generate = stream_generate
        self._make_sampler = make_sampler
        self.model_id = model_id
        self.versions = {"mlx_lm": mlx_lm.__version__, "mlx": mx.__version__}
        self.revision = _cached_revision(model_id)
        self._model, self._tokenizer = mlx_lm.load(model_id)

    def generate(self, messages: list[dict[str, str]], max_tokens: int, temperature: float, seed: int) -> GenerationResult:
        self._mx.random.seed(seed)
        prompt = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        started = time.perf_counter()
        pieces = []
        last = None
        for response in self._stream_generate(
            self._model, self._tokenizer, prompt, max_tokens=max_tokens, sampler=self._make_sampler(temp=temperature)
        ):
            pieces.append(response.text)
            last = response
        return GenerationResult(
            text="".join(pieces),
            prompt_tokens=last.prompt_tokens if last else None,
            generation_tokens=last.generation_tokens if last else None,
            finish_reason=last.finish_reason if last else None,
            elapsed_s=round(time.perf_counter() - started, 3),
        )


class MockBackend:
    """Pseudo-random, schema-shaped output for testing the pipeline. Not evidence of anything."""

    name = "mock"
    is_mock = True
    model_id = MOCK_MODEL_ID
    revision = None
    versions: dict[str, Any] = {}

    def generate(self, messages: list[dict[str, str]], max_tokens: int, temperature: float, seed: int) -> GenerationResult:
        digest = hashlib.sha256(f"{seed}|{json.dumps(messages)}".encode("utf-8")).digest()
        roll = digest[0] / 255
        last = messages[-1]["content"]
        if "requested_action" in last:
            if roll < 0.1:
                text = "MOCK malformed output"
            else:
                action = ("submit", "reject", "hold")[digest[1] % 3]
                decision = "permitted" if action == "submit" else "not_permitted"
                text = json.dumps({"decision": decision, "reasons": ["MOCK"], "requested_action": action})
        elif '"status"' in last:
            status = ("REFUSE", "CLEAR") if "CLEAR" in last else ("PASS", "FAIL")
            text = json.dumps({"status": status[digest[1] % 2], "observation": "MOCK observation."})
        else:
            text = "MOCK review text."
        return GenerationResult(text, len(last.split()), len(text.split()), "stop", 0.0)


def _cached_revision(model_id: str) -> Optional[str]:
    """The commit of a locally cached model, so a run records exactly which weights it used."""
    if os.path.isdir(model_id):
        return None
    try:
        from huggingface_hub import snapshot_download

        return os.path.basename(snapshot_download(model_id, local_files_only=True))
    except Exception:
        return None


def make_backend(name: str, model_id: str, offline: bool = True):
    if name == "mlx":
        return MLXBackend(model_id, offline=offline)
    if name == "mock":
        return MockBackend()
    raise ValueError(f"Unknown backend: {name}")
