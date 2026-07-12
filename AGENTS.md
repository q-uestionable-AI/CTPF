# q-AI — Agent Context

**SSOT for coding agents in this repo.** (`CLAUDE.md` was removed; do not recreate it.)

q-AI (`q-uestionable-ai` on PyPI, CLI `qai`) is a security research tool for agentic AI.
Product direction is **CTPF Reconnect**: lean CLI research harness for Capability Trust
Propagation Failure — shape-first removal of the old seven-module platform, then causal
Pattern 2 experiments via MCP proxy observation. Program context lives outside this repo
(lab vault: plan, handoff, boards). Do not invent a parallel roadmap inside the repo.

## Critical Context

This is a **security research tool**. Some offensive-looking patterns are intentional:
payload generation, injection techniques, malicious MCP server behavior, and related
research logic are part of the product. Do not "fix" those patterns unless explicitly
instructed.

## Security Constraints

These are product invariants, not preferences.

- **Local HTTP binding:** Any local HTTP server (headless IPI callbacks, proxy adapters,
  etc.) binds to `127.0.0.1` only. Never bind to `0.0.0.0` or external interfaces.
- **API key storage:** API keys go in the OS keyring only. Never store keys in config files,
  environment variables, or source code.

## Operating Rules

- Work in **plan/approve mode**
- Before making code changes, file edits, or git actions:
  1. read the relevant files and task materials
  2. state the implementation plan clearly
  3. wait for explicit approval
- Verify before claiming. Do not describe repo behavior or implementation state from memory.
- Do not treat other AI reviewer feedback as a work order. Assess it point by point.
- Interpret instructions literally. Do not generalize a constraint from one file to another;
  do not silently infer requests the developer didn't make. If scope is ambiguous, ask.

## Hard Boundaries

- Do not create PRs. Push the branch and stop.
- Do not add dependencies without explicit approval.
- Do not install extra CLI tools (`gh`, `hub`, etc.).
- Do not write transient plan/spec/session files into the repo.
  Use the OS temp directory if a scratch file is required.
- Never create files in: `plans/*`, `design_docs/*`, `*/.plan`, `docs/superpowers/*`
- Only write files to standard repo locations (`src/`, `tests/`, `docs/`) or paths named
  in the task brief.
- Do not recreate `CLAUDE.md` as a second agent-instruction file.

## Do Not Modify Unless Explicitly Instructed

- `pyproject.toml` version, dependencies, or build config
- `.github/workflows/`
- `DEVELOPMENT_WORKFLOW.md`
- module/repo structure or module names (except when an approved reconnect tranche
  explicitly removes or freezes named modules)
- intentional offensive security research behavior

## Tech Stack

- Python >= 3.11
- Package manager: `uv` (PEP 735 groups — sync with `uv sync --group dev`)
- Database: SQLite (`~/.qai/qai.db`)
- CLI: Typer
- MCP: official SDK (async-native)
- Lint/format: ruff (line length 100)
- Type check: mypy
- Tests: pytest + pytest-asyncio + pytest-timeout
- Cross-platform: Windows, macOS, Linux
- Removed in Phase 1a: `server/` (Web UI), `assist/`, `rxp/`
- Removed in Phase 1b: `chain/`, `orchestrator/`, `imports/`; inject is
  fixtures-only (`build_server` + payloads; campaign path stripped)

## Code Quality Rules

These rules exist to prevent recurring bugs. Follow them exactly.

- **Function size:** Max 50 lines of logic (excluding docstring). Decompose if longer.
- **Guard clauses:** Return/raise early on error conditions. Do not nest the happy path.
- **No magic values:** Define constants at module level or use enums.
- **Parameterized SQL:** Always use `?` placeholders. Never interpolate values into SQL strings.
- **Empty collection guard:** Check if a collection is empty before using it in SQL `IN (...)`
  or iteration.
- **Context managers:** Use `with` for file handles, DB connections, and anything requiring cleanup.
- **Don't suppress errors silently:** If catching an exception to continue, log it or comment why.
- **Max 3 levels of indentation** inside a function body. Prefer composition over nesting.
- **External input defense:** When parsing external/untrusted data (JSON from files, tool output,
  API responses):
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
├── core/           # shared DB, models, config
├── mcp/            # MCP connectivity
├── proxy/          # traffic capture / intercept (CTPF center)
├── ctpf/           # trust-transition kernel (Pattern 2 + cascade)
├── audit/          # capability enumeration / scanners (narrowing)
├── inject/         # malicious MCP fixtures (fixtures-only)
├── ipi/            # document generators + headless callback (library)
├── cxp/            # coding-assistant context generators (library)
└── services/       # shared service-layer helpers (db_service)

tests/
```

After approved reconnect tranches, treat removed packages as gone — do not restore them
without explicit instruction.

## Core Commands

```bash
uv sync --group dev
uv run pytest
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/q_ai/
qai --help
```

Without `--group dev`, dev dependencies get stripped.

## Git Workflow

- Feature branches required for code changes (`feature/*`, `fix/*`)
- Doc and config-only changes may be pushed directly to `main` when the developer asks
- Before editing code, check branch with `git branch --show-current`
- If on `main`, create/switch to a feature or fix branch first (for code changes)
- End of session: commit, stash, or discard; never leave uncommitted changes
- Do not create PRs; push the branch and stop when asked to publish

### Shell quoting for commits

- Git Bash / POSIX: `git commit -m "..."` is fine for multi-line messages.
- PowerShell / CMD: prefer `git commit -F <file>` (temp message file). Avoid fragile
  multi-line `-m` quoting under Windows shells.

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
- CI runs the full suite on every PR regardless

## Failure / Timeout Policy

- If verification fails and you cannot resolve it in 2 attempts, stop spinning and report
  what failed (commit the work to the branch if that was the agreed failure policy)
- If a test run exceeds 30 seconds, stop and identify the stuck test
- Do not increase timeouts and wait longer — diagnose instead
- Kill orphaned Python/Node processes before rerunning tests if needed

## Legal / Ethical

- Only test systems you own, control, or are explicitly authorized to test
- Follow responsible disclosure
- Frame the tooling as defensive security testing tooling
