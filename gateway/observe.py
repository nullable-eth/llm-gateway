"""Read-only observability tools: Prometheus and Alertmanager.

Without these the model reached for `kubectl exec ... curl prometheus`, which
is refused (and should be). Metrics, alert history and silences are the first
thing an SRE looks at, so they are first-class, read-only tools here: a query
can never change anything, so none of them is an action.
"""
import json
from datetime import datetime, timedelta, timezone

import httpx

from . import config

NAMES = {"prometheus_query", "prometheus_query_range", "list_alerts", "list_silences"}

TOOLS = [
    {"type": "function", "function": {"name": "prometheus_query",
        "description": "Run an instant PromQL query against the cluster's Prometheus (read-only). Use for current state: up, kube_pod_status_ready, node_*, container_*, ALERTS, etc.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "PromQL expression"},
            "time": {"type": "string", "description": "optional RFC3339 or unix time; default now"}},
            "required": ["query"]}}},
    {"type": "function", "function": {"name": "prometheus_query_range",
        "description": "Run a PromQL range query (read-only) to see how something changed over time, e.g. when a pod started restarting. Returns a compact series summary.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "minutes": {"type": "integer", "description": "how far back from now; default 60, max 10080"},
            "step": {"type": "string", "description": "resolution, e.g. 30s, 1m, 5m; default chosen from minutes"}},
            "required": ["query"]}}},
    {"type": "function", "function": {"name": "list_alerts",
        "description": "List alerts currently known to Alertmanager (read-only), including whether each is silenced or inhibited.",
        "parameters": {"type": "object", "properties": {
            "filter": {"type": "string", "description": "optional label matcher, e.g. namespace=\"media\""}},
            "required": []}}},
    {"type": "function", "function": {"name": "list_silences",
        "description": "List active and pending Alertmanager silences (read-only): who created them, why, and until when.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
]


def _labels(metric: dict) -> str:
    name = metric.get("__name__", "")
    rest = ",".join(f'{k}="{v}"' for k, v in sorted(metric.items()) if k != "__name__")
    return f"{name}{{{rest}}}"


def _cap(text: str) -> str:
    lim = config.TOOL_OUTPUT_MAX
    return text if len(text) <= lim else text[:lim] + f"\n… truncated ({len(text)} chars)"


async def _prom(path: str, params: dict) -> dict | str:
    if not config.PROMETHEUS_URL:
        return "prometheus is not configured (PROMETHEUS_URL unset)"
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.get(f"{config.PROMETHEUS_URL}{path}", params=params)
        body = r.json()
    except Exception as exc:
        return f"prometheus query failed: {type(exc).__name__}: {exc}"
    if body.get("status") != "success":
        return f"prometheus error: {body.get('errorType')}: {body.get('error')}"
    return body.get("data") or {}


async def prometheus_query(query: str, time: str = "") -> str:
    if not query.strip():
        return "REFUSED: query is required"
    data = await _prom("/api/v1/query", {"query": query, **({"time": time} if time else {})})
    if isinstance(data, str):
        return data
    kind, result = data.get("resultType"), data.get("result")
    if kind == "vector":
        if not result:
            return "(empty result)"
        lines = [f"{_labels(s.get('metric', {}))} {s.get('value', [None, '?'])[1]}" for s in result]
        return _cap(f"{len(result)} series\n" + "\n".join(lines))
    if kind in ("scalar", "string"):
        return str(result[1] if isinstance(result, list) else result)
    return _cap(json.dumps(data)[:config.TOOL_OUTPUT_MAX])


async def prometheus_query_range(query: str, minutes: int = 60, step: str = "") -> str:
    if not query.strip():
        return "REFUSED: query is required"
    minutes = max(1, min(int(minutes or 60), 10080))
    step = step or f"{max(15, minutes * 60 // 120)}s"
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=minutes)
    data = await _prom("/api/v1/query_range", {"query": query, "start": start.timestamp(),
                                               "end": end.timestamp(), "step": step})
    if isinstance(data, str):
        return data
    result = data.get("result") or []
    if not result:
        return "(empty result)"
    out = [f"{len(result)} series, last {minutes} min, step {step}"]
    for s in result:
        vals = s.get("values") or []
        nums = []
        for _, v in vals:
            try:
                nums.append(float(v))
            except ValueError:
                pass
        first_t = datetime.fromtimestamp(vals[0][0], timezone.utc).strftime("%H:%M:%SZ") if vals else "-"
        last_t = datetime.fromtimestamp(vals[-1][0], timezone.utc).strftime("%H:%M:%SZ") if vals else "-"
        summary = (f"first={vals[0][1]}@{first_t} last={vals[-1][1]}@{last_t} "
                   f"min={min(nums):g} max={max(nums):g} points={len(vals)}") if nums else "no numeric points"
        # Where the value changed: the useful part of a timeline, cheaply.
        changes, prev = [], None
        for t, v in vals:
            if v != prev:
                changes.append(f"{datetime.fromtimestamp(t, timezone.utc).strftime('%H:%M')}={v}")
                prev = v
        out.append(f"{_labels(s.get('metric', {}))}: {summary}; changes: {' '.join(changes[:30])}"
                   + (" …" if len(changes) > 30 else ""))
    return _cap("\n".join(out))


async def _am(path: str, params: dict | None = None):
    if not config.ALERTMANAGER_URL:
        return "alertmanager is not configured (ALERTMANAGER_URL unset)"
    try:
        async with httpx.AsyncClient(timeout=20) as c:
            r = await c.get(f"{config.ALERTMANAGER_URL}{path}", params=params or {})
        return r.json()
    except Exception as exc:
        return f"alertmanager query failed: {type(exc).__name__}: {exc}"


async def list_alerts(filter: str = "") -> str:
    data = await _am("/api/v2/alerts", {"filter": filter} if filter else None)
    if isinstance(data, str):
        return data
    if not data:
        return "(no alerts)"
    lines = []
    for a in data:
        lb = dict(a.get("labels") or {})
        st = a.get("status") or {}
        state = st.get("state", "?")
        if st.get("silencedBy"):
            state += " (silenced)"
        if st.get("inhibitedBy"):
            state += " (inhibited)"
        name = lb.pop("alertname", "?")
        summary = (a.get("annotations") or {}).get("summary", "")
        lines.append(f"{name} [{state}] since {a.get('startsAt', '?')} {lb} — {summary}")
    return _cap(f"{len(lines)} alerts\n" + "\n".join(lines))


async def list_silences() -> str:
    data = await _am("/api/v2/silences")
    if isinstance(data, str):
        return data
    live = [s for s in data if (s.get("status") or {}).get("state") in ("active", "pending")]
    if not live:
        return "(no active silences)"
    lines = [f"{s.get('id')} [{s['status']['state']}] until {s.get('endsAt')} by {s.get('createdBy')}: "
             f"{[(m['name'], m['value']) for m in s.get('matchers', [])]} — {s.get('comment')}"
             for s in live]
    return _cap("\n".join(lines))


async def dispatch(name: str, args: dict) -> str:
    if name == "prometheus_query":
        return await prometheus_query(str(args.get("query") or ""), str(args.get("time") or ""))
    if name == "prometheus_query_range":
        return await prometheus_query_range(str(args.get("query") or ""),
                                            int(args.get("minutes") or 60), str(args.get("step") or ""))
    if name == "list_alerts":
        return await list_alerts(str(args.get("filter") or ""))
    if name == "list_silences":
        return await list_silences()
    return f"REFUSED: unknown tool '{name}'"
