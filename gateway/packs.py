"""Capability packs: on-demand tool + runbook loading.

Off by default. When a pack config is supplied (MCP_PACKS / MCP_PACKS_FILE),
the gateway stops offering every tool on every call: it offers finish() and
load_capability(), puts a manifest of packs in the system prompt, and the
model loads only the packs a task needs. Each pack is a set of tool-name
matchers (glob patterns against whatever the MCP endpoint lists) and,
optionally, a runbook that is injected when the pack is loaded.

The mechanism is generic; the packs (which tools, which runbook, the
"load me when" line) are operator config, exactly like the endpoint behind
MCP_URL. Empty config = legacy behaviour, every tool offered on every call.
"""
from __future__ import annotations

import fnmatch
import json
import logging
import os

from . import config

log = logging.getLogger("gateway.packs")


class Pack:
    def __init__(self, name: str, match: list[str], when: str, runbook: str):
        self.name = name
        self.match = match          # glob patterns, matched against tool names
        self.when = when            # one-line "load me when…" for the manifest
        self.runbook = runbook      # injected as the load result

    def matches(self, tool_name: str) -> bool:
        return any(fnmatch.fnmatchcase(tool_name, p) for p in self.match)


_packs: "dict[str, Pack] | None" = None


def _runbook(spec: dict) -> str:
    """Resolve a pack's runbook: inline `runbook`, or `runbook_file` read from
    MCP_RUNBOOKS_DIR. A missing file is a warning, not a crash — the pack still
    loads its tools, just without guidance."""
    inline = str(spec.get("runbook") or "").strip()
    fname = str(spec.get("runbook_file") or "").strip()
    if not fname:
        return inline
    path = os.path.join(config.MCP_RUNBOOKS_DIR, fname) if config.MCP_RUNBOOKS_DIR else fname
    try:
        with open(path, encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError as e:
        log.warning("packs: runbook %s unreadable (%s); pack keeps its tools", path, e)
        return inline


def _load() -> "dict[str, Pack]":
    raw = config.MCP_PACKS
    if not raw and config.MCP_PACKS_FILE:
        try:
            with open(config.MCP_PACKS_FILE, encoding="utf-8") as fh:
                raw = fh.read()
        except OSError as e:
            log.warning("packs: MCP_PACKS_FILE %s unreadable (%s)", config.MCP_PACKS_FILE, e)
            return {}
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except ValueError as e:
        log.error("packs: MCP_PACKS is not valid JSON (%s); packs disabled", e)
        return {}
    out: "dict[str, Pack]" = {}
    for name, spec in (data or {}).items():
        if not isinstance(spec, dict):
            continue
        match = spec.get("match") or spec.get("prefixes") or []
        if isinstance(match, str):
            match = [match]
        out[name] = Pack(name, list(match), str(spec.get("when") or ""), _runbook(spec))
    log.info("packs: loaded %d capabilit%s: %s", len(out),
             "y" if len(out) == 1 else "ies", ", ".join(out) or "(none)")
    return out


def reload() -> None:
    """Drop the cache so the next access re-reads config (used by tests)."""
    global _packs
    _packs = None


def all_packs() -> "dict[str, Pack]":
    global _packs
    if _packs is None:
        _packs = _load()
    return _packs


def enabled() -> bool:
    return bool(all_packs())


def names() -> list[str]:
    return list(all_packs())


def get(name: str) -> "Pack | None":
    return all_packs().get(name)


def tools_for(name: str, all_tools: list[dict]) -> list[dict]:
    p = get(name)
    if not p:
        return []
    return [t for t in all_tools if p.matches(str(t.get("name") or ""))]


def manifest() -> str:
    """The capabilities list injected into the system prompt in deferred mode."""
    if not enabled():
        return ""
    lines = [
        "You are not carrying every tool at once. Capabilities load on demand — call "
        "load_capability(\"<name>\") to add a capability's tools and its runbook, and "
        "load only what the task in front of you needs. For a question that needs none "
        "of these (general knowledge, a chat), just answer; do not load anything.",
        "",
        "Capabilities:",
    ]
    for p in all_packs().values():
        lines.append(f"- {p.name}: {p.when}")
    lines += [
        "",
        "You can only act through capabilities you have loaded. A task may reveal it "
        "needs another; load it then. If a load reports it found no tools, say so — do "
        "not carry on as if the capability were simply unavailable.",
    ]
    return "\n".join(lines)
