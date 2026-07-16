# CTPF Research Harness

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![CI](https://github.com/q-uestionable-AI/CTPF/actions/workflows/ci.yml/badge.svg)](https://github.com/q-uestionable-AI/CTPF/actions/workflows/ci.yml)
[![CodeQL](https://github.com/q-uestionable-AI/CTPF/actions/workflows/codeql.yml/badge.svg)](https://github.com/q-uestionable-AI/CTPF/actions/workflows/codeql.yml)
[![PyPI](https://img.shields.io/pypi/v/ctpf.svg)](https://pypi.org/project/ctpf/)
[![Python 3.11–3.13](https://img.shields.io/badge/python-3.11--3.13-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Docs](https://img.shields.io/badge/docs-ctpf.q--uestionable.ai-8b5cf6)](https://ctpf.q-uestionable.ai)

**Evidence-first capability-trust experiments for agentic systems**

CTPF is a research project investigating **Capability Trust Propagation Failure**: whether data
crossing tools, artifacts, sessions, or capabilities acquires authority that its provenance,
integrity, authorization scope, or intended audience does not justify.

The `ctpf` Python package is the project's reference research harness. It is not a general scanner,
red-team platform, proxy product, or workflow orchestrator. It coordinates controlled conditions,
captures original and modified protocol evidence, verifies run-scoped external effects, compares
results conservatively, and writes integrity-checkable evidence bundles.

## Research method

A CTPF experiment:

1. declares the user-approved scope, trust boundary, prohibited capability, and observable effect;
2. runs isolated baseline, manipulated, and when justified hardened conditions under fixed pins;
3. preserves the original response before a narrow intervention;
4. records invocation, result, persistence, later consumption, and external effect separately;
5. requires exact causal continuity and a matching run-scoped effect where the scenario calls for
   them; and
6. classifies missing, malformed, contaminated, or contradictory evidence as `INCONCLUSIVE`.

`CONFIRMED`, `NOT_OBSERVED`, and `INCONCLUSIVE` apply only to the named experiment and pinned
conditions. They do not establish a population rate, general model vulnerability or resistance,
production impact, or the validity of CTPF as a universal class.

## Current empirical scope

| Scenario | Question | Availability and demonstrated scope |
| --- | --- | --- |
| `cascade-memo` | Can changed authority persist into an artifact, cross a session boundary, and precede a matching action and effect? | Included in PyPI `v0.14.0`; demonstrated through manual Cursor, OpenAI-compatible driven inference, a two-model matrix, and Claude Code runtime workflows. Outcomes differ by pin and include confirmed, inconclusive, and confounded observations. |
| `pattern2` | Can one changed status response precede a matching privileged action and run-scoped effect? | Present on unreleased `main`; demonstrated through one pinned OpenAI-compatible acceptance series. It is not part of the public `v0.14.0` package. |

These are narrow calibration/reference scenarios. Their prompts delegate conditional action authority
to returned data, so the positive observations demonstrate the response-to-effect mechanism but do
not by themselves establish independently emergent authority promotion.

## Reference implementation

| Command | Role |
| --- | --- |
| `ctpf experiment` | Run packaged controlled experiments |
| `ctpf targets` | Register demonstrated model and agent-runtime targets |
| `ctpf proxy` | Observe or intervene in MCP traffic and preserve protocol evidence |
| `ctpf runs` / `ctpf findings` | Inspect retained operational records |
| `ctpf config` / `ctpf db` | Manage non-secret settings, OS-keyring credentials, and local state |

Audit enumeration and SARIF export remain in-tree as a frozen secondary library, not a public CLI
pillar or research contribution. Demonstrated fixtures stay narrow and scenario-specific. Removed
IPI, CXP, Inject, Chain, RXP, Assist, Imports, Orchestrator, and Web UI packages are not product
modules.

## Run the released reference experiment

Install the current public package and inspect the demonstrated scenarios:

```bash
pip install ctpf==0.14.0
ctpf experiment run --help
ctpf experiment run cascade-memo --help
```

For a fully driven cascade series, first store the endpoint credential in the OS keyring. The
target record contains only the credential name:

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

Source `main` currently exposes both `cascade-memo` and `pattern2`. Do not treat source-only behavior
as part of `v0.14.0` or as a commitment to another package release.

> By [Richard Spicer](https://richardspicer.io) · [q-uestionable-AI](https://q-uestionable.ai)

Published documentation: [ctpf.q-uestionable.ai](https://ctpf.q-uestionable.ai).

---

## Legal

All tools are intended for authorized security testing only. Only test systems you own,
control, or have explicit permission to test. Responsible disclosure for all
vulnerabilities discovered.

## License

[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)

## AI Disclosure

This project uses AI-assisted development. All code is reviewed and tested before merge.
