# CTPF Research Harness — Architecture

## Purpose

This document records durable architectural invariants for CTPF Research Harness after the CTPF
Reconnect Phase 1 shape cut. It describes system shape, boundaries, and
operating model — not route inventories or file-by-file ownership.

## System Shape

CTPF Research Harness is a local Python harness with one preferred public operator entry point:
the `ctpf` CLI. The former `qai` executable remains a compatibility alias.

**Centered substrate:**

| Area | Responsibility |
| --- | --- |
| `proxy` | Capture, intercept, live modify, export MCP traffic (Textual TUI) |
| `mcp` | MCP transports and sessions (stdio, SSE, streamable HTTP) |
| `core` | Shared models, SQLite persistence, config, credentials, LLM protocol |
| `ctpf` | Trust-transition kernel, trace parsing, evidence bundles, cascade fixture |
| experiment adapters | Cascade director, OpenAI-compatible driver, Claude Code CLI runtime |

**Library modules** (not public CLI pillars):

| Area | Responsibility |
| --- | --- |
| `audit` | Capability enumeration / scanners; SARIF export |

**Removed in Phase 1** (do not restore without explicit instruction): Web UI
(`server/`), `assist/`, `rxp/`, `chain/`, `orchestrator/`, `imports/`, `ipi/`,
`cxp/`, and `inject/`.

Public CLI verbs: `proxy`, `experiment`, `targets`, `runs`, `findings`, `config`,
`db`, `--version`. New verbs such as `inspect`, `evidence`, or `fixture` are deferred
until a demonstrated experiment defines a real interface.

## Shared Backbone

Persistence is a local SQLite database (`~/.qai/qai.db`). The common schema
centers on `targets`, `runs`, `findings`, `evidence`, and `settings`, with
additional tables retained for historical module data. IPI, CXP, Inject, Chain, and
RXP tables may still exist for old databases even though their writers are gone.

`services/db_service.py` provides shared database helpers used by the
transitional CLI. Other former “service layer” UI/workflow helpers were removed
with the Web UI and orchestrator.

## Boundaries and Responsibilities

### Core Boundary

`core` owns durable cross-cutting contracts: database access, schema migration,
shared data models, configuration, credential lookup (OS keyring), and the
provider-agnostic protocol used by driven inference.

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

`cli.py` is the only public product surface. Demonstrated fixture modules are
implementation equipment behind the narrow experiment command, not independent CLIs.

## LLM Boundary

Provider-facing code stays behind `core.llm` and the LiteLLM-backed implementation.
The demonstrated remote target is OpenAI-compatible; the independent Claude Code adapter
uses runtime-managed authentication rather than an embedded provider SDK.

## Security and Trust Invariants

- Any CTPF-owned local HTTP listener or proxy adapter binds to `127.0.0.1` only.
- Non-secret settings and credentials use separate paths: ordinary settings in
  config/database stores; API keys only in the OS keyring.
- API keys are read only from the OS keyring. The Claude Code target uses the
  runtime's secure login and receives a minimal non-secret environment.
- The packaged cascade fixture is the only runtime experiment fixture. Historical
  calibration and proxy fixtures remain under `tests/fixtures/`; no generic fixture
  hierarchy or `ctpf fixture` command exists.

## What This Document Intentionally Omits

This document does not freeze Mintlify page trees, CTPF experiment schemas, or
fixture catalogs. Those evolve with Pattern 2+ work and should be verified in
source or the lab vault plan when needed.
