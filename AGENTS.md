# q-AI — Agent Context

q-AI is a unified security testing tool for agentic AI. It is a Python package
(`q-uestionable-ai` on PyPI) with seven modules, a CLI (`qai`), and a local web UI.

## Critical Context

This is a **security research tool**. Some offensive-looking patterns are intentional:
payload generation, injection techniques, malicious MCP server behavior, and related
research logic are part of the product. Do not "fix" those patterns unless explicitly instructed.

## Security Constraints

These are product invariants, not preferences.

- **Web UI binding:** The server binds to `127.0.0.1` only. Never bind to `0.0.0.0` or external interfaces.
- **API key storage:** API keys go in the OS keyring only. Never store keys in config files, environment variables, or source code.

## Operating Rules

- Work in **plan/approve mode**
- Before making code changes, file edits, or git actions:
  1. read the relevant files and task materials
  2. state the implementation plan clearly
  3. wait for explicit approval
- Verify before claiming. Do not describe repo behavior or implementation state from memory.
- Do not treat other AI reviewer feedback as a work order. Assess it point by point.

## Hard Boundaries

- Do not create PRs. Push the branch and stop.
- Do not add dependencies without explicit approval.
- Do not install extra CLI tools (`gh`, `hub`, etc.).
- Do not write transient plan/spec/session files into the repo.
  Use the OS temp directory if a scratch file is required.
- Never create files in: `plans/*`, `design_docs/*`, `*/.plan`, `docs/superpowers/*`
- Only write files to standard repo locations (`src/`, `tests/`, `docs/`) or paths named in the task brief.

## Do Not Modify Unless Explicitly Instructed

- `pyproject.toml` version, dependencies, or build config
- `.github/workflows/`
- `DEVELOPMENT_WORKFLOW.md`
- module/repo structure or module names
- intentional offensive security research behavior

## Tech Stack

- Python >= 3.11
- Package manager: `uv`
- Web: FastAPI + Jinja2 + HTMX + DaisyUI
- Database: SQLite
- CLI: Typer
- Lint/format: ruff (line length 100)
- Type check: mypy
- Tests: pytest + pytest-asyncio + pytest-timeout
- Cross-platform: Windows, macOS, Linux

## Code Quality Rules

These rules exist to prevent recurring bugs. Follow them exactly.

- **Function size:** Max 50 lines of logic (excluding docstring). Decompose if longer.
- **Guard clauses:** Return/raise early on error conditions. Do not nest the happy path.
- **No magic values:** Define constants at module level or use enums.
- **Parameterized SQL:** Always use `?` placeholders. Never interpolate values into SQL strings.
- **Empty collection guard:** Check if a collection is empty before using it in SQL `IN (...)` or iteration.
- **Context managers:** Use `with` for file handles, DB connections, and anything requiring cleanup.
- **Don't suppress errors silently:** If catching an exception to continue, log it or comment why.
- **Max 3 levels of indentation** inside a function body. Prefer composition over nesting.
- **External input defense:** When parsing external/untrusted data (JSON from files, tool output, API responses):
  - After `json.loads`, check `isinstance(result, dict)` before calling `.get()`
  - Coerce numeric fields with try/except before arithmetic
  - Wrap per-record processing in try/except that appends warnings instead of crashing
  - Never pass a potentially empty string to `json.loads` without guarding
  - Warn and skip malformed records — never crash the import

## Coding Standards

- Google-style docstrings on all public functions and classes
- Type hints required on all public function signatures
- Use `pathlib.Path` for all file paths
- Use subprocess list form; avoid `shell=True` unless explicitly unavoidable
- Async: MCP SDK is async-native. Use `async/await` for MCP interactions.

## Repo Layout

```text
src/q_ai/
├── core/
├── server/
├── orchestrator/
├── mcp/
├── audit/
├── proxy/
├── inject/
├── chain/
├── ipi/
├── cxp/
├── rxp/
└── imports/

tests/
```

## Core Commands

```bash
uv sync --group dev
uv run pytest
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/q_ai/
qai --help
```

## Git Workflow

- Feature branches required for code changes (`feature/*`, `fix/*`)
- Before editing code, check branch with `git branch --show-current`
- If on `main`, create/switch to a feature or fix branch first
- End of session: commit, stash, or discard; never leave uncommitted changes

## Validation

Run before every commit:

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy src/q_ai/ && uv run pre-commit run --all-files
```

Smoke test the CLI after changes:

```bash
qai --help
```

Test scope:
- Follow the task brief exactly if it specifies tests
- Otherwise default to scoped tests for changed code

## Failure / Timeout Policy

- If verification fails and you cannot resolve it in 2 attempts, stop spinning and report what failed
- If a test run exceeds 30 seconds, stop and identify the stuck test
- Do not increase timeouts and wait longer — diagnose instead
- Kill orphaned Python/Node processes before rerunning tests if needed

## Legal / Ethical

- Only test systems you own, control, or are explicitly authorized to test
- Follow responsible disclosure
- Frame the tooling as defensive security testing tooling

---

## Platform Notes

### Windows CMD

CMD corrupts `git commit -m "message with spaces"`. When running in Windows CMD, use this workaround:

```cmd
echo "feat: description here" > .commitmsg && git commit -F .commitmsg && del .commitmsg
```

This does not apply to PowerShell, bash, or other shells.

---

## Cursor Cloud specific instructions

Environment is provisioned by the startup update script (`uv sync --group dev --python 3.11`);
`uv` lives in `~/.local/bin` (already on PATH via `.bashrc`). Standard commands are in the
`## Core Commands` section above — use those. Notes below are non-obvious gotchas only.

- **Pin the venv to Python 3.11.** The repo has no committed lockfile and `mypy` is configured
  with `python_version = "3.11"`. If the venv is built on Python 3.12+, `mypy` fails inside the
  `numpy` stubs (`Type statement is only supported in Python 3.12 and greater`). Always create the
  venv with `--python 3.11` (matches the CI lint job). Rebuild with `uv sync --group dev --python 3.11`
  if `mypy` reports that stub error.
- **Web UI:** `uv run qai ui --port <PORT> --no-browser` starts FastAPI bound to `127.0.0.1` only.
  It auto-selects a free port if `--port` is omitted and writes the chosen port to `~/.qai/port`.
  All state (targets, runs, findings) persists to the SQLite DB at `~/.qai/qai.db`.
- **Known brittle tests:** A handful of CLI tests (in `tests/core/test_cli_prompt.py`,
  `tests/inject/test_cli.py`, `tests/ipi/test_cli.py`) are sensitive to `typer`/`click` and `rich`
  versions. Because there is no lockfile, `uv sync` resolves the newest releases, which can drift
  from what these tests assert (e.g. `typer._click.exceptions.Exit` vs `click.exceptions.Exit`,
  and `rich` truncating table column text). These failures are dependency drift, not code bugs.
