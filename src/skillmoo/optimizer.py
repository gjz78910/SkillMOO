from __future__ import annotations

import random
from typing import Any

from .candidates import SkillCandidate, mutate_candidate
from .llm_provider import LLMProvider, LLMProviderError

_PROVIDER = LLMProvider(
    api_key_envs=("SKILLMOO_LLM_API_KEY", "OPENAI_API_KEY", "GLM_API_KEY"),
    base_url_envs=("SKILLMOO_LLM_BASE_URL", "OPENAI_BASE_URL", "GLM_BASE_URL"),
    model_envs=("SKILLMOO_LLM_MODEL", "GLM_MODEL"),
    timeout_envs=("SKILLMOO_LLM_TIMEOUT_SEC",),
)

_VALID_OPERATIONS = {"prune", "substitute", "add", "reorder"}

_SYSTEM_PROMPT = (
    "You are a skill bundle optimizer for an LLM coding agent. "
    "Propose one targeted edit to the current skill bundle to improve task pass rate and/or reduce inference cost. "
    "Respond ONLY with a JSON object — no prose, no markdown."
)

_PROMPT_TEMPLATE = """\
Available skills: {available}
Current bundle:   {current}

Evaluation evidence:
  pass_rate : {pass_rate:.2f}
  cost_usd  : {cost_usd:.4f}
  status    : {status}
  failures  : {failures}

Propose ONE edit. Operations: prune (remove skills), substitute (replace one skill with another), add (add a missing skill), reorder (change invocation order).

Respond with exactly:
{{"operation": "<prune|substitute|add|reorder>", "selected_skill_ids": ["<id>", ...], "reason": "<one sentence>"}}

Rules:
- selected_skill_ids must be a non-empty subset/permutation of the available skills list.
- For reorder, use the same skills but in a different order.
- For prune, remove one or more skills from the current bundle.
- For substitute, swap out one skill for a different available skill.
- For add, include one new skill not in the current bundle.
"""


def propose_bundle_edit(
    candidate: SkillCandidate,
    all_skill_ids: list[str],
    evidence: dict[str, Any],
    rng: random.Random,
    family_hint: str,
) -> tuple[SkillCandidate, str]:
    """Return (new_candidate, operation_type). Falls back to heuristic if LLM is unavailable or fails."""
    if _PROVIDER.available():
        try:
            return _llm_propose(candidate, all_skill_ids, evidence)
        except Exception:
            pass
    child = mutate_candidate(candidate, all_skill_ids, rng, family_hint=family_hint)
    return child, _infer_operation(candidate, child)


def _llm_propose(
    candidate: SkillCandidate,
    all_skill_ids: list[str],
    evidence: dict[str, Any],
) -> tuple[SkillCandidate, str]:
    prompt = _PROMPT_TEMPLATE.format(
        available=", ".join(all_skill_ids) if all_skill_ids else "(none)",
        current=", ".join(candidate.selected_skill_ids) if candidate.selected_skill_ids else "(empty)",
        pass_rate=float(evidence.get("pass_rate", 0.0)),
        cost_usd=float(evidence.get("cost_usd", 0.0)),
        status=str(evidence.get("status", "unknown")),
        failures=str(evidence.get("failure_summary", ""))[:400] or "none",
    )
    response = _PROVIDER.complete_json(prompt, system=_SYSTEM_PROMPT)
    operation = str(response.get("operation", "substitute")).lower().strip()
    if operation not in _VALID_OPERATIONS:
        operation = "substitute"
    raw_ids = response.get("selected_skill_ids")
    if not isinstance(raw_ids, list):
        raise LLMProviderError("selected_skill_ids missing from LLM response.")
    allowed = set(all_skill_ids)
    selected = [s for s in raw_ids if isinstance(s, str) and s in allowed]
    if not selected:
        raise LLMProviderError("LLM returned no valid skill IDs.")
    new_candidate = SkillCandidate(selected_skill_ids=tuple(selected))
    return new_candidate, operation


def _infer_operation(old: SkillCandidate, new: SkillCandidate) -> str:
    old_set, new_set = set(old.selected_skill_ids), set(new.selected_skill_ids)
    if new_set < old_set:
        return "prune"
    if new_set > old_set:
        return "add"
    if old_set == new_set and old.selected_skill_ids != new.selected_skill_ids:
        return "reorder"
    return "substitute"
