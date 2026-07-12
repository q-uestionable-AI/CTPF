# {q-AI}

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![CI](https://github.com/q-uestionable-AI/qai/actions/workflows/ci.yml/badge.svg)](https://github.com/q-uestionable-AI/qai/actions/workflows/ci.yml)
[![CodeQL](https://github.com/q-uestionable-AI/qai/actions/workflows/codeql.yml/badge.svg)](https://github.com/q-uestionable-AI/qai/actions/workflows/codeql.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Docs](https://img.shields.io/badge/docs-q--uestionable.ai-8b5cf6)](https://docs.q-uestionable.ai)

**CTPF research harness — MCP observation, controlled fixtures, and evidence**

A research program testing whether MCP and agentic AI vulnerabilities are exploitable end-to-end, with execution-level proof.

Model-layer tests like Garak, PyRIT, or BIPIA can tell you "this model follows injected instructions." An audit scan can tell you "this MCP server has a vulnerable tool." A qai finding connects the two: the model weakness was exploitable end-to-end through a real agentic system. Authenticated callbacks confirm execution, not just compliance.

Capabilities (transitional CLI surface):

- Intercept MCP traffic (`qai proxy`)
- Register and manage targets, runs, and findings
- Library modules remain for IPI document generation, inject fixtures, audit scanning, and related research paths (not all are root-CLI commands)

All findings stored in a SQLite database.

> By [Richard Spicer](https://richardspicer.io) · [{q-AI}](https://q-uestionable.ai)

---

## Quick Start

Intercept MCP traffic:

```bash
qai proxy --help
```

Register a target:

```bash
qai targets add "My Server" http://localhost:3000/sse
```

---

## Framework Coverage

Audit findings map to four security taxonomies:

| Framework | Coverage |
|-----------|----------|
| [OWASP MCP Top 10](https://owasp.org/www-project-mcp-top-10/) | All 10 categories |
| [OWASP Agentic Top 10](https://genai.owasp.org/resource/owasp-top-10-for-agentic-applications-for-2026/) | All 10 categories |
| [MITRE ATLAS](https://atlas.mitre.org/) | Technique-level mapping per finding category |
| [CWE](https://cwe.mitre.org/) | Weakness-level mapping per finding category |

---

## Install

```bash
pip install q-uestionable-ai
```

Or from source:

```bash
git clone https://github.com/q-uestionable-AI/qai.git
cd qai
uv sync --group dev
```

---

Full documentation at [docs.q-uestionable.ai](https://docs.q-uestionable.ai)

---

## Legal

All tools are intended for authorized security testing only. Only test systems you own, control, or have explicit permission to test. Responsible disclosure for all vulnerabilities discovered.

## License

[Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)

## AI Disclosure

This project uses AI-assisted development. All code is reviewed and tested before merge.
