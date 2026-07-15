# CTPF Research Harness — Architecture

## Purpose

CTPF is a research project for testing Capability Trust Propagation Failure. The Python package is
its evidence-first reference harness: it coordinates controlled conditions and preserves the
experimental record. It is not the thesis itself and is not a generic scanner, proxy product,
red-team platform, or workflow orchestrator.

This document describes the current source architecture and its durable responsibility boundaries.
It does not turn proposed studies or integrations into implemented behavior.

## Responsibility layers

| Layer | Responsibility | Location |
| --- | --- | --- |
| Research thesis | Define provenance, integrity, authorization scope, intended audience, persistence, and capability-promotion questions | Research program and study records outside the repository |
| Stable harness | Condition isolation, protocol capture, trust-transition records, causal comparison, effect verification, and evidence integrity | Reusable `ctpf` package |
| Study instrument | Scenario task, prompt, fixture, intervention, authority semantics, and effect oracle expectations | Narrow scenario modules and tests |
| External system | Supply a model, agent runtime, benchmark task, or raw research artifact when a named study requires it | Registered target or separately approved study integration |
| Interpretation | Decide what an observation establishes and authorize publication language | Human researcher |

External labels or conclusions never become CTPF conclusions automatically. Raw external artifacts
must be preserved before normalization.

## Package shape

| Area | Responsibility |
| --- | --- |
| `experiment.py` | CLI and coordinator for complete isolated condition sequences, manifests, comparisons, and bundles |
| `kernel` | Trust-transition schemas, observations, trace parsing, scenario-specific comparison, effect oracles, and evidence bundles |
| `proxy` | MCP observation/intervention infrastructure: capture, optional forward/modify/drop, persistence, replay, and export |
| `mcp` | Async MCP connections, transports, discovery, and protocol models |
| `driven_inference.py` | Demonstrated OpenAI-compatible model/tool loop and pinned inference transcript capture |
| `external_runtime.py` | Demonstrated external agent-runtime seam, currently Claude Code for the cascade workflow |
| `core` | SQLite state, shared models, configuration, OS-keyring credentials, and provider-neutral LLM contracts |
| `services` | Small shared database helpers |
| `audit` | Frozen secondary enumeration/scanner library with reporting and SARIF support; not a root CLI or methods contribution |

The public operator entry point is the `ctpf` CLI. Its research command is `ctpf experiment`; proxy,
targets, stored records, configuration, and database commands support that work.

## Experimental flow

1. The scenario declares its prompt, tools, mutation, expected trust transition, and effect.
2. The coordinator allocates a series and condition-scoped run identifiers.
3. Baseline, manipulated, and hardened conditions execute sequentially with isolated state.
4. The proxy preserves protocol messages and the original payload for every modified response.
5. The selected operator invokes either the manual research seam, driven inference, or a demonstrated
   external runtime.
6. Scenario parsing records invocation, response, persistence, later consumption, and external effect
   as separate observations.
7. The kernel compares conditions conservatively. Missing causal continuity or effect evidence fails
   closed to `INCONCLUSIVE`.
8. Run manifests preserve progress and failed attempts; bundles declare and hash copied artifacts.

A tool invocation is not an external effect. A modified response is not a confirmed propagation.
Mechanical classification and scientific interpretation remain separate.

## Packaged scenarios

### Cascade memo

`ctpf experiment run cascade-memo` uses two sessions per condition. Session A can persist a memo;
Session B reads the resulting artifact without a fresh response mutation. The scorer can require
exact write → artifact → read continuity before accepting a later matching action and run-scoped
sink effect.

Demonstrated seams are manual Cursor operation, OpenAI-compatible driven inference, a sequential
two-model matrix, and Claude Code as an external agent runtime. Those workflows produced confirmed,
inconclusive, mixed-control, and confounded observations; execution support is not scientific
generality.

Cascade memo is the experiment included in the public `v0.14.0` package.

### Pattern 2

`ctpf experiment run pattern2` uses one session per condition. It changes a status result, records
whether `apply_change` is invoked, and independently verifies the matching run-scoped sink effect.

Pattern 2 is packaged on unreleased source `main` and has one demonstrated OpenAI-compatible driven
acceptance series. No Pattern 2 Claude Code result or cross-runtime claim is currently demonstrated.
It must not be documented as available from PyPI `v0.14.0`.

Both scenarios use shared condition, target, transcript, trace, effect, comparison, and bundle
machinery. Their prompts, tools, mutations, trust semantics, parsers, and oracle rules remain
scenario-specific. That is engineering reuse, not scientific generality and not a generic fixture
hierarchy.

## Target and runtime boundaries

Persisted targets identify the execution seam:

- `inference` with `driver=openai-compatible`, an exact model, generation pins, and an OS-keyring
  credential name;
- `agent-runtime` with the demonstrated `claude-code-cli` driver and runtime-managed authentication;
- MCP server targets used by observation/proxy workflows.

Driven inference uses LiteLLM behind the provider-neutral `core.llm` contract. API keys are fetched
from the OS keyring and are never stored in target metadata, config files, environment variables, or
evidence. The Claude Code adapter supplies a minimal non-secret environment and relies on the
runtime's secure login.

There is no adapter registry, plugin system, recipe language, DAG, or general external-artifact
ingestion contract. A future study may add one narrow external seam only after demonstrating that
the external tool cannot already provide the required CTPF evidence contract.

## State and evidence boundaries

Local operational state uses SQLite at `~/.ctpf/ctpf.db`. The schema centers on targets, runs,
findings, evidence, settings, and proxy sessions; upgraded databases may retain historical tables
whose product writers were removed.

Controlled research traces, transcripts, mutations, effects, manifests, bundles, analysis notes, and
publication material belong in an operator-selected external research directory or the lab vault,
never in the Git checkout. The repository contains reusable source, tests, narrow synthetic fixtures,
schemas, and tool documentation.

## Security and trust invariants

- Every CTPF-owned local HTTP listener or proxy adapter binds to `127.0.0.1` only.
- API keys are stored and read only through the OS keyring.
- Modified protocol responses preserve their originals.
- Condition state and external effects are run-scoped.
- Evidence that is missing, malformed, contaminated, or causally incomplete is not reconstructed.
- Research output directories must be outside Git checkouts.
- Fixtures remain narrow and colocated with demonstrated scenarios; no generic fixture command or
  hierarchy exists.

## Removed and frozen surfaces

Web UI (`server/`), Assist, RXP, Chain, Orchestrator, Imports, IPI, CXP, and Inject packages were
removed and must not be restored without an approved demonstrated need. Historical database tables
may remain for migration and deletion compatibility.

Audit remains in-tree but frozen and secondary. It does not define CTPF's identity and should not
expand in competition with dedicated external scanners.

## Release boundary

The installed package version is defined by the release tag, not by the current contents of `main`.
As of `v0.14.0`, the public package contains cascade memo but not Pattern 2. Source documentation must
identify unreleased behavior explicitly until a separately approved release changes that boundary.

## Deliberate omissions

This document does not freeze public documentation navigation, publication claims, external-tool
integrations, or a future study design. Those require their own evidence and approval boundaries.
