"""Agents with separate conversation histories, plus strict output parsing.

Parsing never turns bad output into a refusal: output the controller cannot act
on is reported as malformed, and stays distinct from reject and hold.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"
PROMPT_FILES = ("authorization.txt", "availability.txt", "preparation.txt", "submission.txt", "review.txt")
ACTIONS = ("submit", "reject", "hold")
_SECTION_RE = re.compile(r"^\[(\w+)\]\s*$", re.MULTILINE)


def load_sections(path: Path) -> dict[str, str]:
    """Split a prompt file into its [name] sections."""
    parts = _SECTION_RE.split(path.read_text())
    return {name: body.strip() for name, body in zip(parts[1::2], parts[2::2])}


def load_prompts(prompt_dir: Path = PROMPT_DIR) -> dict[str, dict[str, str]]:
    return {name.removesuffix(".txt"): load_sections(prompt_dir / name) for name in PROMPT_FILES}


def prompts_sha256(prompt_dir: Path = PROMPT_DIR) -> str:
    digest = hashlib.sha256()
    for name in PROMPT_FILES:
        digest.update(name.encode("utf-8"))
        digest.update((prompt_dir / name).read_bytes())
    return digest.hexdigest()


class Agent:
    """One logical agent. Its history is its own; agents share only a loaded model."""

    def __init__(self, name: str, system_prompt: str, backend: Any):
        self.name = name
        self.backend = backend
        self.history = [{"role": "system", "content": system_prompt}]

    def ask(self, content: str, stage: str, max_tokens: int, temperature: float, seed: int) -> dict[str, Any]:
        self.history.append({"role": "user", "content": content})
        messages = [dict(m) for m in self.history]
        result = self.backend.generate(messages, max_tokens=max_tokens, temperature=temperature, seed=seed)
        self.history.append({"role": "assistant", "content": result.text})
        return {
            "agent": self.name,
            "stage": stage,
            "seed": seed,
            "max_tokens": max_tokens,
            "messages": messages,
            "raw_output": result.text,
            "prompt_tokens": result.prompt_tokens,
            "generation_tokens": result.generation_tokens,
            "finish_reason": result.finish_reason,
            "elapsed_s": result.elapsed_s,
        }


def extract_json_object(text: str) -> Optional[dict[str, Any]]:
    """The first JSON object in the text, allowing for a code fence around it."""
    start = text.find("{")
    if start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


@dataclass(frozen=True)
class ParsedDecision:
    outcome: str  # "submit", "reject", "hold" or "malformed"
    parsed: Optional[dict[str, Any]]
    error: Optional[str]
    # What a malformed output appeared to request. For sensitivity analysis only:
    # the controller never acts on it.
    action_hint: Optional[str] = None
    # Defects that do not hide the action, such as empty reasons. Requiring these
    # fields to be perfect would misclassify clear submit requests as malformed.
    schema_defects: tuple[str, ...] = ()


def parse_decision(text: str) -> ParsedDecision:
    """`requested_action` alone decides the outcome; other defects are only flagged."""
    parsed = extract_json_object(text)
    hint_match = re.search(r'"requested_action"\s*:\s*"(submit|reject|hold)"', text)
    hint = hint_match.group(1) if hint_match else None
    if parsed is None:
        return ParsedDecision("malformed", None, "no parseable JSON object", hint)
    action = parsed.get("requested_action")
    if action not in ACTIONS:
        return ParsedDecision("malformed", parsed, f"invalid requested_action: {action!r}", hint)

    defects = []
    decision = parsed.get("decision")
    if decision not in ("permitted", "not_permitted"):
        defects.append("invalid_decision")
    elif (decision == "permitted") != (action == "submit"):
        defects.append("decision_contradicts_action")
    reasons = parsed.get("reasons")
    if isinstance(reasons, str):
        reasons = [reasons] if reasons.strip() else []
    if not isinstance(reasons, list) or not reasons:
        defects.append("no_reasons")
    return ParsedDecision(action, parsed, None, hint, tuple(defects))


def parse_upstream(text: str, allowed_statuses: tuple[str, ...]) -> tuple[Optional[str], Optional[str]]:
    """Return (status, observation), or (None, None) if the report is unusable."""
    parsed = extract_json_object(text)
    if parsed is None:
        return None, None
    status, observation = parsed.get("status"), parsed.get("observation")
    if status not in allowed_statuses or not isinstance(observation, str) or not observation.strip():
        return None, None
    return status, observation.strip()
