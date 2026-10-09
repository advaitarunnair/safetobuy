"""LLM verbalizer with a numeric guardrail.

The LLM only rewords numbers that the system computed. It is given the display
strings from explain/facts.py and nothing else. Hard constraint, enforced in
code rather than by prompt alone: every number in the LLM's output must be one
of the supplied numbers, after the same rounding. Any line that fails, any API
problem, a refusal, or a missing API key falls back to explain/templates.py.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

from cspa.explain.templates import explain_with_templates

PROMPT_VERSION = "v1"
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
_SKU_LIKE = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")
_NUMBER_WORDS = re.compile(
    r"\b(two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    r"hundred|thousand|million|half|halve|double|doubles|twice|triple|dozen|quarter|third|thirds)\b",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """You write short, plain-language purchasing explanations for the owner of a small shop.

You are given facts as JSON. Every number you may use is already in those facts, formatted the way it must appear.

Rules:
- Use numbers only by copying them exactly from the facts, character for character (for example "$1,240", "31%", "4.5"). Never compute, round, convert, combine or estimate a number, and never introduce one that is not in the facts.
- Write every number as digits. Do not spell numbers as words, and do not use words such as "half", "double" or "twice".
- Do not mention SKU identifiers; the line is shown next to the SKU.
- One sentence per SKU, at most 35 words. Say what to do and why, in terms a shop owner cares about: cash, running out, margin at stake.
- The weekly summary is two or three sentences.
- If a fact is "n/a", leave it out.

This restriction matters because the owner acts on these numbers: a sentence with a number the system did not compute would be a fabricated figure."""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"sku": {"type": "string"}, "text": {"type": "string"}},
                "required": ["sku", "text"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "lines"],
    "additionalProperties": False,
}


def _canon(token: str) -> str | None:
    try:
        d = Decimal(token.replace(",", ""))
    except InvalidOperation:
        return None
    return format(d.normalize(), "f")


def extract_numbers(text: str, skus: list[str] | None = None) -> list[str]:
    """Every number in `text`, canonicalised. SKU identifiers are removed first."""
    for s in sorted(skus or [], key=len, reverse=True):
        text = text.replace(s, " ")
    text = _SKU_LIKE.sub(" ", text)
    out = []
    for m in _NUMBER.finditer(text):
        c = _canon(m.group(0).rstrip(","))
        if c is not None:
            out.append(c)
    return out


def allowed_numbers(*fact_dicts: dict) -> set[str]:
    """Numbers the LLM may use: everything that appears in the supplied display strings."""
    allowed: set[str] = set()
    for f in fact_dicts:
        for v in f.get("fmt", {}).values():
            allowed.update(extract_numbers(str(v)))
    return allowed


def verify_text(text: str, allowed: set[str], skus: list[str] | None = None) -> tuple[bool, list[str]]:
    """True if every number in `text` is an allowed number and no number is spelled out."""
    bad = [n for n in extract_numbers(text, skus) if n not in allowed]
    bad += [m.group(0) for m in _NUMBER_WORDS.finditer(text)]
    return (not bad, bad)


def llm_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


def _payload(facts: dict, batch: list[dict]) -> str:
    slim = {
        "week": {"status": facts["week"]["status"], **facts["week"]["fmt"]},
        "skus": [{"sku": f["sku"], "decision": f["decision"], "perishable": f["perishable"], **f["fmt"]} for f in batch],
    }
    return json.dumps(slim, indent=1)


def _call(client, cfg, facts: dict, batch: list[dict]) -> dict | None:
    """One request. Returns parsed JSON, or None if the response cannot be used."""
    kwargs = dict(
        model=str(cfg.llm.model),
        max_tokens=int(cfg.llm.max_tokens),
        system=SYSTEM_PROMPT,
        output_config={"effort": str(cfg.llm.effort), "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
        messages=[{"role": "user", "content": "Facts:\n" + _payload(facts, batch) + "\n\nWrite the weekly summary and one line per SKU."}],
    )
    if bool(cfg.llm.refusal_fallback):
        response = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kwargs)
    else:
        response = client.messages.create(**kwargs)
    if response.stop_reason in ("refusal", "max_tokens"):
        return None
    text = next((b.text for b in response.content if b.type == "text"), None)
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _cache_file(cfg, facts: dict) -> Path:
    from cspa.config import resolve_path

    key = hashlib.sha256(json.dumps([PROMPT_VERSION, str(cfg.llm.model), str(cfg.llm.effort), facts], sort_keys=True, default=str).encode()).hexdigest()[:24]
    return resolve_path(cfg.llm.cache_dir) / f"{key}.json"


def explain(facts: dict, cfg, client=None, use_cache: bool = True) -> dict:
    """Explanations for a week. Always returns a complete result.

    Returns {'summary', 'lines': {sku: text}, 'source': {sku: 'llm'|'template'},
             'summary_source', 'rejected': [{sku, text, bad_numbers}], 'error'}.
    """
    out = explain_with_templates(facts)
    out.update(rejected=[], error=None)
    if not facts["skus"] and not facts["week"]:
        return out
    if client is None:
        if not bool(cfg.llm.enabled) or not llm_available():
            return out
        cache = _cache_file(cfg, facts)
        if use_cache and cache.exists():
            return json.loads(cache.read_text())
        try:
            import anthropic

            client = anthropic.Anthropic()
        except Exception as exc:  # SDK missing or misconfigured: templates still work
            out["error"] = f"{type(exc).__name__}: {exc}"
            return out
    else:
        cache = None

    week = facts["week"]
    week_allowed = allowed_numbers(week)
    by_sku = {f["sku"]: f for f in facts["skus"]}
    all_skus = list(by_sku)
    size = int(cfg.llm.max_skus_per_call)
    batches = [facts["skus"][i : i + size] for i in range(0, len(facts["skus"]), size)] or [[]]
    try:
        for bi, batch in enumerate(batches):
            data = _call(client, cfg, facts, batch)
            if data is None:
                continue
            if bi == 0 and isinstance(data.get("summary"), str):
                ok, bad = verify_text(data["summary"], week_allowed, all_skus)
                if ok and data["summary"].strip():
                    out["summary"], out["summary_source"] = data["summary"].strip(), "llm"
                else:
                    out["rejected"].append({"sku": None, "text": data["summary"], "bad_numbers": bad})
            for item in data.get("lines", []):
                sku, text = item.get("sku"), item.get("text")
                if sku not in by_sku or not isinstance(text, str) or not text.strip():
                    continue
                ok, bad = verify_text(text, allowed_numbers(by_sku[sku], week), all_skus)
                if ok:
                    out["lines"][sku], out["source"][sku] = text.strip(), "llm"
                else:
                    out["rejected"].append({"sku": sku, "text": text, "bad_numbers": bad})
    except Exception as exc:  # network, auth, rate limit, bad request: keep whatever passed, template for the rest
        out["error"] = f"{type(exc).__name__}: {exc}"
    if cache is not None and out["error"] is None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out, indent=1))
    return out
