# CTPF Research Harness

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![CI](https://github.com/q-uestionable-AI/CTPF/actions/workflows/ci.yml/badge.svg)](https://github.com/q-uestionable-AI/CTPF/actions/workflows/ci.yml)
[![CodeQL](https://github.com/q-uestionable-AI/CTPF/actions/workflows/codeql.yml/badge.svg)](https://github.com/q-uestionable-AI/CTPF/actions/workflows/codeql.yml)
[![Python 3.11–3.13](https://img.shields.io/badge/python-3.11--3.13-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Docs](https://img.shields.io/badge/docs-ctpf.q--uestionable.ai-8b5cf6)](https://ctpf.q-uestionable.ai)

**Trust-boundary testing for agentic systems**

CTPF Research Harness investigates **Capability Trust Propagation Failure (CTPF)**: whether low-trust data
(for example a tool result) is silently promoted into higher-authority actions when
provenance, integrity, authorization scope, or intended audience are not preserved.

The product shape is a small local CLI: capture MCP traffic, mutate it under control via
proxy, and keep core target, run, and finding records in SQLite. Controlled experiments write
traces, effect artifacts, and hashed evidence bundles to a required operator-selected output
directory outside the Git checkout. Individual experiments **confirm** or **fail to observe**
promotion under pinned conditions — they do not “falsify CTPF” as a class.

### Public CLI

| Command | Role |
|---------|------|
| `ctpf proxy` | Intercept, inspect, modify, and export MCP traffic (Textual TUI) |
| `ctpf experiment` | Run controlled CTPF experiments |
| `ctpf targets` | Register MCP targets |
| `ctpf runs` / `ctpf findings` | Inspect stored runs and findings |
| `ctpf config` / `ctpf db` | Settings and local database maintenance |
| `ctpf --version` | Package version |

The former `qai` command remains a compatibility alias for `ctpf` during the identity transition.

### Libraries (not root CLI pillars)

IPI document generators + headless callback, inject malicious MCP fixture servers, CXP
context generators, and audit enumeration/SARIF export remain in-tree as libraries for
research fixtures. They are not equal product modules on the public CLI.
Audit-library findings can include OWASP MCP, OWASP Agentic, MITRE ATLAS, and CWE
identifiers; audit remains a library capability rather than a root CLI surface.

> By [Richard Spicer](https://richardspicer.io) · [q-uestionable-AI](https://q-uestionable.ai)

---

## Quick Start

Install from PyPI:

```bash
pip install ctpf
```

Starting with v0.12.0, the former `q-uestionable-ai` distribution is a compatibility package
that installs the same-version `ctpf` distribution.

```bash
ctpf proxy --help
ctpf targets add "My Server" http://localhost:3000/sse
```

Or run from source:

```bash
git clone https://github.com/q-uestionable-AI/CTPF.git
cd CTPF
uv sync --group dev
uv run ctpf proxy --help
uv run ctpf targets add "My Server" http://localhost:3000/sse
```

---

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
