# CTPF Research Harness

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![CI](https://github.com/q-uestionable-AI/CTPF/actions/workflows/ci.yml/badge.svg)](https://github.com/q-uestionable-AI/CTPF/actions/workflows/ci.yml)
[![CodeQL](https://github.com/q-uestionable-AI/CTPF/actions/workflows/codeql.yml/badge.svg)](https://github.com/q-uestionable-AI/CTPF/actions/workflows/codeql.yml)
[![PyPI](https://img.shields.io/pypi/v/ctpf.svg)](https://pypi.org/project/ctpf/)
[![Python 3.11–3.13](https://img.shields.io/badge/python-3.11--3.13-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Docs](https://img.shields.io/badge/docs-ctpf.q--uestionable.ai-8b5cf6)](https://ctpf.q-uestionable.ai)

CTPF is a research project investigating **Capability Trust Propagation Failure**: whether data
crossing tools, artifacts, sessions, or capabilities acquires authority that its provenance,
integrity, authorization scope, or intended audience does not justify.

The `ctpf` Python package and CLI implement the project's research harness. They coordinate
controlled conditions, preserve original and modified protocol evidence, check run-scoped external
effects, compare results under defined rules, and produce evidence bundles with file hashes. The
package does not provide general-purpose vulnerability scanning, red-team automation, or workflow
orchestration.

The governed control interface is designed for operation by an AI research agent. The CLI remains
directly usable by a human for validation. A human authorizes the campaign targets, exact remote
RunSpecs, resource and effect limits, spending, and risk. The agent can then execute the listed
runs without separate approval for each run. Humans retain scientific adjudication and publication
decisions. Mechanical scores are not research conclusions.

## Research method

A CTPF experiment:

1. declares the user-approved scope, trust boundary, prohibited capability, and observable effect;
2. runs isolated baseline, manipulated, and when justified hardened conditions under fixed
   configuration values;
3. preserves the original response before a narrow intervention;
4. records invocation, result, persistence, later consumption, and external effect separately;
5. requires exact causal continuity and a matching run-scoped effect where the scenario calls for
   them; and
6. classifies missing, malformed, contaminated, or contradictory evidence as `INCONCLUSIVE`.

`CONFIRMED`, `NOT_OBSERVED`, and `INCONCLUSIVE` apply only to the named experiment and pinned
conditions. They do not establish a population rate, general model vulnerability or resistance,
production impact, or the validity of CTPF as a universal class.

## Current empirical scope

| Scenario | Question | Availability and recorded use |
| --- | --- | --- |
| `cascade-memo` | Can changed authority persist into an artifact, cross a session boundary, and precede a matching action and effect? | Included in PyPI `v0.14.0`. It has been used in manual, OpenAI-compatible inference, matrix, and external agent-runtime runs. Recorded outcomes vary by configuration and include `CONFIRMED`, `INCONCLUSIVE`, and confounded runs. |
| `pattern2` | Can one changed status response precede a matching privileged action and run-scoped effect? | Implemented on unreleased `main` and used in one pinned OpenAI-compatible acceptance series. It is not included in `v0.14.0`. |
| `pattern3-scope` | Can an approved read scope precede a write outside that scope and a matching run-scoped effect? | Implemented on unreleased `main`. Its deterministic preflight passed. The first governed inference attempt stopped during fixture startup before a model request, so it produced no empirical outcome. It is not included in `v0.14.0`. |

These are narrow experimental scenarios. The `cascade-memo` and `pattern2` prompts delegate
conditional action authority to returned data. Observations from those scenarios therefore cover
the configured response-to-effect mechanism; they do not establish independently emergent
authority promotion. The `pattern3-scope` prompt instead authorizes only `read_record`, prohibits
`write_record`, and directs the model to request fresh approval for a change. It has not produced an
empirical inference outcome.

## Commands

| Command | Role |
| --- | --- |
| `ctpf experiment` | Run packaged controlled experiments |
| `ctpf experiment control` | Machine lifecycle for an untrusted AI caller (source `main` only) |
| `ctpf experiment govern` | TTY-only human campaign-policy, key, and exceptional per-run authority (source `main` only) |
| `ctpf targets` | Register inference and agent-runtime target profiles |
| `ctpf proxy` | Observe or intervene in MCP traffic and preserve protocol evidence |
| `ctpf runs` / `ctpf findings` | Inspect retained operational records |
| `ctpf config` / `ctpf db` | Manage non-secret settings, OS-keyring credentials, and local state |

`control` and `govern` are present on unreleased source `main`. They are not part of public
`v0.14.0`. Proxy replay and arbitrary target mutation remain human-operated supporting surfaces, not
the autonomous entry path.

Audit enumeration and SARIF export remain in-tree as a frozen secondary library without a public
CLI. Fixtures are scenario-specific. The removed IPI, CXP, Inject, Chain, RXP, Assist, Imports,
Orchestrator, and Web UI packages are not present as product modules.

## Run the released package

Install `v0.14.0` and inspect its experiment commands:

```bash
pip install ctpf==0.14.0
ctpf experiment run --help
ctpf experiment run cascade-memo --help
```

To run `cascade-memo` against an OpenAI-compatible target, first store the endpoint credential in
the OS keyring. The target record contains the credential name, not the credential value:

```bash
ctpf config set-credential local-research
ctpf targets add "Local Model" http://127.0.0.1:1234/v1 --type inference --meta driver=openai-compatible --meta model=MODEL_ID --meta credential=local-research --meta max_tokens=512 --meta temperature=0 --meta seed=0 --meta reasoning_effort=none
```

The command prints an eight-character target ID prefix. Run the isolated baseline, manipulated, and
hardened conditions into an absolute research directory outside a Git checkout:

```bash
ctpf experiment run cascade-memo --target TARGET_ID_PREFIX --output-root ABSOLUTE_EXTERNAL_DIRECTORY
```

The output directory contains condition-scoped traces, inference transcripts where available,
mutation records with originals, effect artifacts, run manifests, mechanical comparisons, and a
hashed evidence bundle. The configured model/runtime is external; the packaged cascade tools and
effects are synthetic and run-scoped.

## Run from source

```bash
git clone https://github.com/q-uestionable-AI/CTPF.git
cd CTPF
uv sync --group dev
uv run ctpf experiment run --help
```

Source `main` currently includes `cascade-memo`, `pattern2`, `pattern3-scope`, and the governed
`control` / `govern` interfaces. Do not treat source-only behavior as part of `v0.14.0` or as a
commitment to another package release.

### Agent control interface on source `main`

These commands inspect capabilities and validate a RunSpec without starting an experiment:

```bash
uv run ctpf experiment control capabilities
uv run ctpf experiment control validate < runspec.json
```

A human creates one signed campaign policy through `ctpf experiment govern` on an interactive TTY.
For bounded remote work, that policy lists the exact RunSpec digests the agent may execute; each
listed run starts idempotently, and the policy does not authorize a changed or additional RunSpec.
Per-run approval is also available. Deploy the agent inside an external OS/runtime sandbox; the
harness does not provide containment for unrestricted shell access. Details:
[Agent-Operable Lifecycle](https://ctpf.q-uestionable.ai/experiments/agent-operable).

Documentation: [ctpf.q-uestionable.ai](https://ctpf.q-uestionable.ai).

---

## Legal

All tools are intended for authorized security testing only. Only test systems you own,
control, or have explicit permission to test. Use responsible disclosure for vulnerabilities
discovered during testing.

## License

[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)

## AI Disclosure

This project uses AI-assisted development. CI runs automated tests, linting, type checking, and
security scans.
