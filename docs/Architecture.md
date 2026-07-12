# q-AI — Architecture

## Purpose

This document records durable architectural invariants for q-AI after the CTPF
Reconnect Phase 1 shape cut. It describes system shape, boundaries, and
operating model — not route inventories or file-by-file ownership.

## System Shape

q-AI is a local Python **CTPF research harness** with one public operator entry
point: the `qai` CLI.

**Centered substrate:**

| Area | Responsibility |
| --- | --- |
| `proxy` | Capture, intercept, live modify, export MCP traffic (Textual TUI) |
| `mcp` | MCP transports and sessions (stdio, SSE, streamable HTTP) |
| `core` | Shared models, SQLite persistence, config, credentials, providers |

**Library modules** (not public CLI pillars; used as fixtures / research code):

| Area | Responsibility |
| --- | --- |
| `audit` | Capability enumeration / scanners; SARIF export |
| `ipi` | Document generators + headless callback listener |
| `cxp` | Coding-assistant context-file generators |
| `inject` | Malicious MCP fixture servers (`build_server` + payload templates) |

**Removed in Phase 1** (do not restore without explicit instruction): Web UI
(`server/`), `assist/`, `rxp/`, `chain/`, `orchestrator/`, `imports/`, and the
inject campaign/scoring path.

Transitional public CLI verbs: `proxy`, `targets`, `runs`, `findings`, `config`,
`db`, `--version`. New verbs such as `inspect` / `evidence` are deferred until
a CTPF experiment defines a real interface.

## Shared Backbone

Persistence is a local SQLite database (`~/.qai/qai.db`). The common schema
centers on `targets`, `runs`, `findings`, `evidence`, and `settings`, with
additional tables retained for historical module data (proxy sessions, IPI hits,
legacy chain/inject/RXP tables may still exist for old DBs even when writers are
gone).

`services/db_service.py` provides shared database helpers used by the
transitional CLI. Other former “service layer” UI/workflow helpers were removed
with the Web UI and orchestrator.

## Boundaries and Responsibilities

### Core Boundary

`core` owns durable cross-cutting contracts: database access, schema migration,
shared data models, configuration, credential lookup (OS keyring), and
provider/model abstractions.

### MCP Boundary

`mcp` centralizes client connections to MCP servers. Proxy adapters and other
callers build on that shared async transport boundary rather than embedding
transport setup ad hoc.

### Proxy Boundary

`proxy` is the CTPF observation center: bidirectional message capture, optional
intercept with forward / modify / drop, session persistence, and export. Live
mutation of server→client tool results is a first-class research path; full
agent counterfactual replay is not assumed.

### Operator Surface

`cli.py` is the only public product surface. Library Typer apps (for example
`python -m q_ai.ipi`, `python -m q_ai.inject`) may remain for fixture workflows
but are not root `qai` pillars.

## LLM Boundary

Provider-facing code stays behind shared core interfaces (`core.llm`,
`core.providers`, LiteLLM-backed implementation). Modules that need models
consume that boundary rather than embedding provider SDKs directly.

## Security and Trust Invariants

- Any local HTTP listener (IPI headless callback, proxy listen adapters) binds
  to `127.0.0.1` only — never `0.0.0.0` or external interfaces for product
  surfaces.
- Non-secret settings and credentials use separate paths: ordinary settings in
  config/database stores; API keys only in the OS keyring.
- The IPI callback listener may optionally expose itself via a tunnel adapter
  (`--tunnel cloudflare`) for testing remote/cloud AI targets. When tunneled,
  the listener trusts only the `CF-Connecting-IP` header for source-IP
  resolution and ignores `X-Forwarded-For`. Public-exposure hardening
  (body-size limits, per-IP rate limiting, conservative timeouts) applies in
  tunnel mode; local-only listener behavior is unchanged.
- At most one tunneled IPI callback listener should exist on a host at a time.
  The active-callback state file (`~/.qai/active-callback`) is the single-writer
  coordination point for CLI-launched listeners.

## What This Document Intentionally Omits

This document does not freeze Mintlify page trees, CTPF experiment schemas, or
fixture catalogs. Those evolve with Pattern 2+ work and should be verified in
source or the lab vault plan when needed.
