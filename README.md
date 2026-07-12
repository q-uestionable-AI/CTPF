# {q-AI}

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![CI](https://github.com/q-uestionable-AI/qai/actions/workflows/ci.yml/badge.svg)](https://github.com/q-uestionable-AI/qai/actions/workflows/ci.yml)
[![CodeQL](https://github.com/q-uestionable-AI/qai/actions/workflows/codeql.yml/badge.svg)](https://github.com/q-uestionable-AI/qai/actions/workflows/codeql.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Docs](https://img.shields.io/badge/docs-q--uestionable.ai-8b5cf6)](https://docs.q-uestionable.ai)

**CTPF research harness — MCP observation, controlled fixtures, and evidence**

q-AI investigates **Capability Trust Propagation Failure (CTPF)**: whether low-trust data
(for example a tool result) is silently promoted into higher-authority actions when
provenance, integrity, authorization scope, or intended audience are not preserved.

The product shape is a small local CLI: capture MCP traffic, mutate it under control via
proxy, and keep runs/findings/evidence in SQLite. Individual experiments **confirm** or
**fail to observe** promotion under pinned conditions — they do not “falsify CTPF” as a class.

### Public CLI (transitional)

| Command | Role |
|---------|------|
| `qai proxy` | Intercept, inspect, modify, and export MCP traffic (Textual TUI) |
| `qai targets` | Register MCP targets |
| `qai runs` / `qai findings` | Inspect stored runs and findings |
| `qai config` / `qai db` | Settings and local database maintenance |
| `qai --version` | Package version |

### Libraries (not root CLI pillars)

IPI document generators + headless callback, inject malicious MCP fixture servers, CXP
context generators, and audit enumeration/SARIF export remain in-tree as libraries for
research fixtures. They are not equal product modules on the public CLI.

> By [Richard Spicer](https://richardspicer.io) · [{q-AI}](https://q-uestionable.ai)

---

## Quick Start

```bash
pip install q-uestionable-ai
# or from source:
git clone https://github.com/q-uestionable-AI/qai.git
cd qai
uv sync --group dev
```

```bash
qai proxy --help
qai targets add "My Server" http://localhost:3000/sse
```

---

## Framework mappings (library audit)

When using the audit library, findings can map to:

| Framework | Coverage |
|-----------|----------|
| [OWASP MCP Top 10](https://owasp.org/www-project-mcp-top-10/) | All 10 categories |
| [OWASP Agentic Top 10](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/) | All 10 categories |
| [MITRE ATLAS](https://atlas.mitre.org/) | Technique-level mapping per finding category |
| [CWE](https://cwe.mitre.org/) | Weakness-level mapping per finding category |

---

Architecture notes: [docs/Architecture.md](docs/Architecture.md).
Published docs: [docs.q-uestionable.ai](https://docs.q-uestionable.ai) (Mintlify may still describe removed surfaces until a follow-up prune).

---

## Legal

All tools are intended for authorized security testing only. Only test systems you own,
control, or have explicit permission to test. Responsible disclosure for all
vulnerabilities discovered.

## License

[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)

## AI Disclosure

This project uses AI-assisted development. All code is reviewed and tested before merge.
