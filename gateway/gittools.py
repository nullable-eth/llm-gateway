"""Git tools: read the repos, and propose changes as pull requests.

This is how the agent makes a change that STICKS. kubectl writes restore
service now and are reverted by Flux later; anything meant to last is a git
change, and until this existed the agent could only describe one. Now it can
open the PR, the operator reviews and merges it from a phone, and Flux
reconciles it.

The boundary is the pull request, and it is enforced here rather than trusted
to the prompt:

  * only the repos in GIT_REPOS, under GIT_OWNER;
  * commits only ever land on a branch named GIT_BRANCH_PREFIX + something,
    created from the base branch or an existing agent branch — never on the
    base branch itself, never on a human branch;
  * the only write is "commit to that branch, then open (or update) a PR";
    nothing merges, closes, approves, deletes or touches settings;
  * *.sops.yaml and .github/workflows are refused (the agent cannot encrypt,
    and CI is a privilege boundary), and every YAML file must still parse.

The token is a fine-grained PAT limited to those repos with Contents and
Pull requests read/write. It never appears in any tool output.
"""
import base64
import json
import logging
import re
import time

import httpx
import yaml

from . import config

log = logging.getLogger("gateway")

REFUSED_PATH = re.compile(r"(^|/)\.github/workflows/|\.sops\.ya?ml$")
FOOTER = ("\n\n---\nOpened by the Whitehorse agent (llm-gateway `git_open_pr`). "
          "Nothing merges without a human.")


class GitError(Exception):
    pass


def _repo(name: str) -> str:
    name = (name or "").strip().split("/")[-1]
    allowed = {r.lower(): r for r in config.GIT_REPOS}
    if name.lower() not in allowed:
        raise GitError(f"repo '{name}' is not one the agent may use; "
                       f"allowed: {', '.join(config.GIT_REPOS)}")
    return f"{config.GIT_OWNER}/{allowed[name.lower()]}"


def _client() -> httpx.AsyncClient:
    token = config.git_token()
    if not token:
        raise GitError("git is not configured (no GIT_TOKEN); the operator has to "
                       "add the agent's GitHub token first")
    return httpx.AsyncClient(
        base_url=config.GIT_API, timeout=30,
        headers={"Authorization": f"Bearer {token}",
                 "Accept": "application/vnd.github+json",
                 "X-GitHub-Api-Version": "2022-11-28",
                 "User-Agent": "whitehorse-agent"})


async def _ok(r: httpx.Response, what: str) -> dict:
    if r.status_code >= 300:
        try:
            msg = r.json().get("message", r.text)
        except ValueError:
            msg = r.text
        raise GitError(f"{what}: HTTP {r.status_code}: {str(msg)[:300]}")
    return r.json() if r.content else {}


async def _default_branch(c, repo: str) -> str:
    return (await _ok(await c.get(f"/repos/{repo}"), "repo lookup"))["default_branch"]


async def _head(c, repo: str, branch: str) -> str | None:
    r = await c.get(f"/repos/{repo}/git/ref/heads/{branch}")
    if r.status_code == 404:
        return None
    return (await _ok(r, f"ref {branch}"))["object"]["sha"]


async def _file(c, repo: str, path: str, ref: str) -> str | None:
    r = await c.get(f"/repos/{repo}/contents/{path.lstrip('/')}", params={"ref": ref})
    if r.status_code == 404:
        return None
    d = await _ok(r, f"read {path}")
    if isinstance(d, list):
        raise GitError(f"{path} is a directory; use git_list_files")
    if d.get("encoding") == "base64":
        return base64.b64decode(d["content"]).decode("utf-8", "replace")
    # >1MB files come back without content
    raw = await c.get(f"/repos/{repo}/contents/{path.lstrip('/')}", params={"ref": ref},
                      headers={"Accept": "application/vnd.github.raw+json"})
    if raw.status_code >= 300:
        raise GitError(f"read {path}: HTTP {raw.status_code}")
    return raw.text


def _cap(text: str) -> str:
    return text if len(text) <= config.TOOL_OUTPUT_MAX else (
        text[:config.TOOL_OUTPUT_MAX] + "\n…(truncated; narrow the request)")


# ------------------------------------------------------------------ reads
async def list_files(repo: str, path: str = "", ref: str = "") -> str:
    try:
        full = _repo(repo)
        async with _client() as c:
            ref = ref or await _default_branch(c, full)
            tree = await _ok(await c.get(f"/repos/{full}/git/trees/{ref}",
                                         params={"recursive": "1"}), "tree")
        prefix = path.strip("/")
        paths = [e["path"] for e in tree.get("tree", []) if e.get("type") == "blob"
                 and (not prefix or e["path"] == prefix or e["path"].startswith(prefix + "/"))]
        if not paths:
            return f"no files under '{prefix or '/'}' in {full}@{ref}"
        head = f"{len(paths)} file(s) under '{prefix or '/'}' in {full}@{ref}:\n"
        return _cap(head + "\n".join(paths))
    except GitError as e:
        return f"REFUSED: {e}"


async def read_file(repo: str, path: str, ref: str = "", start_line: int = 1,
                    max_lines: int = 400) -> str:
    try:
        full = _repo(repo)
        async with _client() as c:
            ref = ref or await _default_branch(c, full)
            text = await _file(c, full, path, ref)
        if text is None:
            return f"{path} does not exist in {full}@{ref}"
        lines = text.split("\n")
        start = max(1, int(start_line or 1))
        end = min(len(lines), start - 1 + max(1, int(max_lines or 400)))
        body = "\n".join(lines[start - 1:end])
        more = (f" — {len(lines) - end} more line(s); call again with start_line={end + 1}"
                if end < len(lines) else "")
        # The header is outside the file text. Copy `old` for git_open_pr from
        # the text below it verbatim, never including this line.
        out = f"# {full}@{ref}:{path} lines {start}-{end} of {len(lines)}{more}\n{body}"
        if len(out) > config.TOOL_OUTPUT_MAX:
            return _cap(out) + f"\n(ask for fewer lines, e.g. max_lines={max(20, (end - start) // 2)})"
        return out
    except GitError as e:
        return f"REFUSED: {e}"


async def search(repo: str, query: str) -> str:
    try:
        full = _repo(repo)
        async with _client() as c:
            d = await _ok(await c.get("/search/code",
                                      params={"q": f"{query} repo:{full}", "per_page": 30}),
                          "search")
        items = d.get("items") or []
        if not items:
            return (f"no matches for {query!r} in {full} (GitHub's index can lag a "
                    f"new push; git_list_files + git_read_file always work)")
        return _cap("\n".join(f"{i['path']}" for i in items))
    except GitError as e:
        return f"REFUSED: {e}"


async def list_prs(repo: str) -> str:
    try:
        full = _repo(repo)
        async with _client() as c:
            prs = await _ok(await c.get(f"/repos/{full}/pulls",
                                        params={"state": "open", "per_page": 50}), "pulls")
        mine = [p for p in prs if p["head"]["ref"].startswith(config.GIT_BRANCH_PREFIX)]
        if not mine:
            return f"no open agent PRs in {full}"
        return "\n".join(f"#{p['number']} {p['title']} (branch {p['head']['ref']}) {p['html_url']}"
                         for p in mine)
    except GitError as e:
        return f"REFUSED: {e}"


# ----------------------------------------------------------------- writes
def _slug(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40].strip("-")
    return f"{config.GIT_BRANCH_PREFIX}{s or 'change'}-{time.strftime('%Y%m%d-%H%M%S')}"


def _check_yaml(path: str, text: str) -> None:
    if not path.endswith((".yaml", ".yml")):
        return
    try:
        list(yaml.safe_load_all(text))
    except yaml.YAMLError as e:
        raise GitError(f"{path} would no longer be valid YAML: {str(e)[:300]}")


async def open_pr(repo: str, title: str, body: str, changes: list,
                  branch: str = "", base: str = "") -> str:
    """Commit `changes` to an agent branch and open (or update) its PR.

    Each change is one of:
      {"path": p, "old": exact_text, "new": replacement}   edit in place
      {"path": p, "content": full_text}                    create or replace
      {"path": p, "delete": true}                          remove
    """
    try:
        full = _repo(repo)
        title = (title or "").strip()
        if len(title) < 8:
            raise GitError("a PR needs a real title")
        if not changes or not isinstance(changes, list):
            raise GitError("no changes given")
        if branch and not branch.startswith(config.GIT_BRANCH_PREFIX):
            raise GitError(f"the agent may only commit to branches starting with "
                           f"'{config.GIT_BRANCH_PREFIX}'")
        async with _client() as c:
            default = await _default_branch(c, full)
            base = base or default
            if branch in (base, default):
                raise GitError("never commits to the base branch; a PR is the boundary")
            existing = await _head(c, full, branch) if branch else None
            if branch and existing is None:
                raise GitError(f"branch {branch} does not exist; omit branch to start a new one")
            parent = existing or await _head(c, full, base)
            if parent is None:
                raise GitError(f"base branch {base} not found")
            read_ref = branch if existing else base

            tree, touched = [], []
            for ch in changes:
                path = str((ch or {}).get("path") or "").strip().lstrip("/")
                if not path or ".." in path.split("/"):
                    raise GitError(f"bad path {path!r}")
                if REFUSED_PATH.search(path):
                    raise GitError(f"{path}: SOPS files and CI workflows are not the "
                                   f"agent's to edit; describe the change in the PR body instead")
                if ch.get("delete"):
                    tree.append({"path": path, "mode": "100644", "type": "blob", "sha": None})
                    touched.append(f"delete {path}")
                    continue
                if "content" in ch:
                    text = str(ch["content"])
                else:
                    old, new = ch.get("old"), ch.get("new")
                    if not isinstance(old, str) or not old or not isinstance(new, str):
                        raise GitError(f"{path}: give either content, or old+new")
                    cur = await _file(c, full, path, read_ref)
                    if cur is None:
                        raise GitError(f"{path} does not exist on {read_ref}; use content to create it")
                    n = cur.count(old)
                    if n != 1:
                        raise GitError(f"{path}: `old` must match exactly once on {read_ref}, "
                                       f"it matched {n} times — re-read the file and copy the "
                                       f"text verbatim, including indentation")
                    text = cur.replace(old, new)
                _check_yaml(path, text)
                tree.append({"path": path, "mode": "100644", "type": "blob", "content": text})
                touched.append(path)

            base_tree = (await _ok(await c.get(f"/repos/{full}/git/commits/{parent}"),
                                   "parent commit"))["tree"]["sha"]
            new_tree = await _ok(await c.post(f"/repos/{full}/git/trees",
                                              json={"base_tree": base_tree, "tree": tree}), "tree")
            commit = await _ok(await c.post(f"/repos/{full}/git/commits", json={
                "message": f"{title}\n\n{(body or '').strip()}\n\nProposed by the Whitehorse agent."
                           .strip(),
                "tree": new_tree["sha"], "parents": [parent]}), "commit")

            if existing:
                await _ok(await c.patch(f"/repos/{full}/git/refs/heads/{branch}",
                                        json={"sha": commit["sha"], "force": False}), "update branch")
                prs = await _ok(await c.get(f"/repos/{full}/pulls", params={
                    "state": "open", "head": f"{config.GIT_OWNER}:{branch}"}), "find PR")
                if prs:
                    return (f"pushed {commit['sha'][:7]} to {branch}, updating PR "
                            f"#{prs[0]['number']}: {prs[0]['html_url']}\nchanged: {', '.join(touched)}")
            else:
                branch = _slug(title)
                await _ok(await c.post(f"/repos/{full}/git/refs",
                                       json={"ref": f"refs/heads/{branch}", "sha": commit["sha"]}),
                          "create branch")
            pr = await _ok(await c.post(f"/repos/{full}/pulls", json={
                "title": title, "head": branch, "base": base,
                "body": (body or "").strip() + FOOTER}), "open PR")
        return (f"opened PR #{pr['number']} on {full}: {pr['html_url']}\n"
                f"branch {branch}, commit {commit['sha'][:7]}; changed: {', '.join(touched)}\n"
                f"To revise it, call git_open_pr again with branch=\"{branch}\".")
    except GitError as e:
        return f"REFUSED: {e}"
    except httpx.HTTPError as e:
        return f"git failed: {e}"


TOOLS = [
    {"type": "function", "function": {"name": "git_list_files",
        "description": "List files in one of the operator's repos (Whitehorse = the cluster's GitOps repo, plus cluster-agent, agentmemory, llm-gateway). Use a path prefix to narrow, e.g. 'kubernetes/apps/media'.",
        "parameters": {"type": "object", "properties": {
            "repo": {"type": "string"}, "path": {"type": "string"},
            "ref": {"type": "string", "description": "branch or sha; default branch if omitted"}},
            "required": ["repo"]}}},
    {"type": "function", "function": {"name": "git_read_file",
        "description": "Read a file from one of the repos. Always read a file before proposing an edit to it, and copy `old` text from this output verbatim.",
        "parameters": {"type": "object", "properties": {
            "repo": {"type": "string"}, "path": {"type": "string"}, "ref": {"type": "string"},
            "start_line": {"type": "integer"}, "max_lines": {"type": "integer"}},
            "required": ["repo", "path"]}}},
    {"type": "function", "function": {"name": "git_search",
        "description": "Search a repo's code for text (GitHub code search). Returns matching file paths.",
        "parameters": {"type": "object", "properties": {
            "repo": {"type": "string"}, "query": {"type": "string"}},
            "required": ["repo", "query"]}}},
    {"type": "function", "function": {"name": "git_list_prs",
        "description": "List the agent's own open pull requests in a repo.",
        "parameters": {"type": "object", "properties": {"repo": {"type": "string"}},
            "required": ["repo"]}}},
    {"type": "function", "function": {"name": "git_open_pr",
        "description": "Make a lasting change: commit edits to a new agent branch and open a pull request for the operator to review and merge (Flux then applies it). This is the right way to change the cluster permanently; kubectl changes are reverted by Flux. Prefer small edits: {path, old, new} where old is copied exactly (whitespace included) from git_read_file and matches once. Use {path, content} only for new files. To revise your own open PR, pass its branch.",
        "parameters": {"type": "object", "properties": {
            "repo": {"type": "string"},
            "title": {"type": "string", "description": "conventional-commit style, e.g. 'fix(media): raise sonarr memory request'"},
            "body": {"type": "string", "description": "why, what you observed (evidence), and how to verify after merge"},
            "changes": {"type": "array", "items": {"type": "object", "properties": {
                "path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"},
                "content": {"type": "string"}, "delete": {"type": "boolean"}},
                "required": ["path"]}},
            "branch": {"type": "string", "description": "only to add a commit to one of your existing agent/ branches"},
            "base": {"type": "string", "description": "target branch; default branch if omitted"}},
            "required": ["repo", "title", "body", "changes"]}}},
]

NAMES = {t["function"]["name"] for t in TOOLS}


async def dispatch(name: str, args: dict) -> str:
    if name == "git_list_files":
        return await list_files(args.get("repo", ""), args.get("path", ""), args.get("ref", ""))
    if name == "git_read_file":
        return await read_file(args.get("repo", ""), args.get("path", ""), args.get("ref", ""),
                               int(args.get("start_line", 1) or 1),
                               int(args.get("max_lines", 400) or 400))
    if name == "git_search":
        return await search(args.get("repo", ""), args.get("query", ""))
    if name == "git_list_prs":
        return await list_prs(args.get("repo", ""))
    if name == "git_open_pr":
        return await open_pr(args.get("repo", ""), args.get("title", ""), args.get("body", ""),
                             args.get("changes") or [], args.get("branch", ""),
                             args.get("base", ""))
    return f"REFUSED: unknown tool '{name}'"
