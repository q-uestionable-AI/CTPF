# {q-AI} — Roadmap

> **Superseded for product direction (2026-07-11):** CTPF Reconnect governs sequencing.
> Live plan: lab vault `qai/Plans/CTPF Reconnect Plan.md`. In-repo shape: README +
> `docs/Architecture.md`. Historical phases below describe the pre-strip platform and
> should not steer new work.

## Problem Statement

AI agents are being deployed with broad access to tools, data, and systems — but the security
tooling hasn't kept pace. MCP adoption is accelerating (Claude, ChatGPT, Cursor, Gemini, VS Code)
while most organisations have no visibility into what their MCP servers expose or how agents
behave when tools are manipulated.

Existing tools (Garak, PyRIT) focus on LLM output analysis — they test what a model says, not
what an agent does. {q-AI} tests agent infrastructure: MCP servers, tool trust boundaries,
content ingestion pipelines, coding assistant context files, and retrieval systems. The distinction
matters because agent-level attacks produce real-world effects — code execution, data exfiltration,
lateral movement — that output-level analysis doesn't catch.

### Competitive Landscape

| Tool | Approach | {q-AI} Differentiator |
|------|----------|----------------------|
| Garak (NVIDIA) | LLM output vulnerability scanning | Tests agent infrastructure, not model outputs |
| PyRIT (Microsoft) | Multi-turn red teaming | Maps to OWASP MCP Top 10, tests MCP protocol |
| Invariant MCP-Scan | CLI scanner for tool description attacks | Full OWASP MCP Top 10 + multi-module kill chain |
| ScanMCP.com | Cloud-based scanner | Open source, local execution, SARIF for CI/CD |
| Equixly CLI | Commercial MCP testing | Free, community-extensible |
| MCP Guardian (EQTY Lab) | Runtime security proxy | Complementary — guardian is runtime defence, {q-AI} is pre-deployment testing |

---

## Phased Delivery

### Phase 1–4: Foundation ✅

All seven modules ported and functional. Shared SQLite database with parent/child run lineage.
Local web UI (FastAPI + HTMX). CLI with module subcommands. 1278 tests.

### Phase 5: Workflow Orchestration ✅

"Assess an MCP Server" workflow: audit → proxy (background) + inject, end-to-end through the
browser. Provider-agnostic injection (litellm, any provider/model). Keyring-based credential
storage. Live WebSocket updates. Module adapters for all seven modules.

### Phase 5b: AI Assistant (v0.7.0) ✅

Built-in AI assistant for navigating capabilities and interpreting scan results. RAG over product
documentation and user-provided knowledge. Provider-agnostic (Ollama, Anthropic, OpenAI, etc.).
Three-tier trust boundary model: trusted product docs, semi-trusted user knowledge, untrusted
scan-derived content. CLI (interactive, single-shot, piped input) and web UI (full-page chat,
contextual run results panel). Shipped in PRs #81–#83.

### Phase 6: Public Launch

**Goal:** The repo is public, the package is on PyPI, and old repos are transitioned.

**Sequence:**
1. Final repo review — README rewrite, SECURITY.md, responsible use policy, CI green
2. Flip `q-uestionable-AI/qai` to public
3. Publish `q-uestionable-ai` to PyPI (first release)
4. Deprecation releases for `counteragent` and `countersignal` — CLI warning + README banner
5. Transfer `q-uestionable-AI/counteragent` → `richardspicer/counteragent`, same for countersignal
6. Unified Mintlify docs at `docs.q-uestionable.ai`
7. q-uestionable.ai site update
8. Announcement post on richardspicer.io

**Done when:** `pip install q-uestionable-ai` works, repo is public, old repos redirect.

### Phase 7: Workflow Completion (reduced scope)

Two workflows promoted to v1.0 completion — those that serve hero-module research and the
publication pipeline. Three others deferred or removed.

| Workflow | Modules | Notes |
|----------|---------|-------|
| Test Document Ingestion | ipi, rxp | Generate payloads, validate retrieval before deployment. Serves Campaign 1 publication. |
| Test a Coding Assistant | cxp | Build poisoned repo, guided manual steps, record results. Serves CXP CVE lane. |

**Deferred to post-1.0:**

- **Trace an Attack Path** and **Measure Blast Radius** (both `chain`-based) — pending
  `chain` hero-tier decision from Campaign 3.
- **Manage Research** — removed from the workflow list. Cross-module research view is
  already functional from Phase 3 without a workflow wrapper.

**Done when:** Both remaining workflows launchable from the browser with correct module orchestration.

### Phase 8: Scenario Format v0

**Goal:** Ship a portable unit of attack knowledge. Scenarios are `ipi`-shaped YAML with
explicit extension points, reverse-engineered from real Campaign 1 output rather than
designed in advance. Nuclei has templates, Garak has probes, Metasploit has modules — qai
needs the equivalent.

**Deliverables:**

- Scenario format v0 specification (RFC)
- Reference implementation — loader, runner, exporter
- `qai scenario run <path|url>`, `qai scenario validate`, `qai scenario export`
- `ipi` producing and consuming scenarios end-to-end against live campaign payloads
- At least one reference scenario shipped with the platform

Extension to `cxp` and `audit` happens only after `ipi` integration is proven in real use.

**Done when:** A scenario from another researcher loads and runs against an operator's target
with a single command and produces reproducible indicators.

### Phase 9: Publication Pipeline (Agent Security Index)

**Goal:** Publish the Agent Security Index as a numbered publication series. Each campaign's
existing Publishable Outputs constitute a Publication. Publication 1 ships during Campaign 1
Phase 6 Evidence Assembly.

**Deliverables:**

- Methodology doc — written during Campaign 1 Phase 6, reviewed by an external reader capable
  of pushing back on selection bias and disclosure timing
- Publication 1 — Campaign 1 findings, evidence bundle, and reproduction pack that an external
  researcher can rerun
- Publication surface on mlsecopslab.io with stable URL scheme
- Announcement post on richardspicer.io at Publication 1
- Annual cadence commitment by default; more frequent only after the Campaign 1 pipeline proves
  light enough to sustain it

**Done when:** Publication 1 shipped with validated reproduction pack. Subsequent campaign
publications inherit the pipeline without reinventing it.

### Phase 10: Research Validation and v1.0

**Goal:** Interface stability commitment backed by real research evidence.

See [v1.0 Exit Criteria](#v10-exit-criteria) below.

---

## Planned Directions (Post-v1.0)

### Public API & MCP Server

Expose qai's capabilities programmatically. Enables CI/CD security gates, SIEM/ticketing integration, scripted research campaigns, lab automation, and AI agent access.

**Sequence:** Service layer extraction → API v1 (read) → API v1 (write + auth) → Headless mode → MCP server v1 → Dogfooding.

The API is a versioned `/api/v1/` JSON surface backed by a shared service layer that all interfaces (UI, CLI, API, MCP) call. The MCP server is a thin adapter over the API — it does not invent new semantics. stdio transport first. Read-only by default with opt-in scan capability behind full guardrail set (target allowlist, wait_for_user with canonical approval summary, parameter normalization, audit logging).

**Plan:** `Plans/api-mcp-server-plan.md`
**RFC:** `Plans/api-mcp-server-rfc.md`
**Concept:** `Plans/api-mcp-server-concept.md`

## Deferred

### Desktop Application Packaging

Packaging qai as a double-clickable desktop app (pywebview + PyInstaller) across Windows, macOS,
and Linux. Deferred in favor of the research-platform direction: scenario format and publication
pipeline take priority over packaging breadth. Re-evaluate post-1.0 based on adoption signals.

**Plan:** `Plans/desktop-packaging-plan.md`
**RFC:** `Plans/desktop-packaging-rfc.md`

---

## v1.0 Exit Criteria

SemVer 1.0 is a public commitment to interface stability. After 1.0, breaking changes to CLI
commands, report formats, data models, or YAML schemas require a major version bump.

### Capability Readiness

| Module | Gate |
|--------|------|
| audit | ≥3 distinct MCP server targets scanned, findings documented in Research Log |
| proxy | ≥1 end-to-end intercept session captured and replayed against a live MCP server |
| inject | ≥1 campaign run against a real LLM endpoint with scoring results documented |
| chain | ≥1 chain executed end-to-end against real targets with step evidence documented |
| ipi | ≥1 payload document triggering a callback from a target platform |
| cxp | ≥1 hit confirmed against a real coding assistant |
| rxp | ≥1 validated retrieval rank measurement against a real embedding model |

### Research Validation

- ≥1 publishable finding from inject, chain, ipi, or cxp campaigns (finding ID in Findings/)
- OWASP MCP Top 10 mapping validated against real scan results
- Publication 1 of the Agent Security Index shipped with validated reproduction pack
- Methodology doc reviewed by an external reader capable of pushing back on selection bias and disclosure timing

### Scenario Format

- Scenario format v0 spec shipped
- At least one reference scenario per confirmed hero module
- Scenario format spec frozen at 1.0; breaking changes require major version bump

### Interface Stability

**CLI commands** — frozen at 1.0 for all implemented subcommands. Flags and required arguments stable.

**Data models** — `Run`, `Target`, `Finding`, `Evidence`, `Severity`, `RunStatus` frozen.
Module-specific models frozen when their v1.0 gate is met.

**Report formats** — HTML, SARIF, JSON scan reports; inject YAML payload schema; chain YAML
template schema. Breaking changes require major version bump after 1.0.

**Protocols** — `ProviderClient` protocol interface frozen. `BaseScanner` ABC frozen.
`TransportAdapter` protocol frozen.

### Explicitly Post-1.0

- ~~Proxy client-facing HTTP adapters (proxy as standalone network service)~~ shipped in PR #94
- Chain `blast-radius` and `detect` command implementations (stubs ship at v1.0)
- Trace an Attack Path and Measure Blast Radius workflows — pending `chain` hero-tier decision from Campaign 3
- Multi-model comparison in RXP
- Detection rule export automation

---

## What Success Looks Like

- A CVE or bounty finding with a published finding doc and Sigma/Wazuh detection rules
- Detection rules that other defenders actually deploy
- A tool that security researchers use for real MCP assessments
- Conference CFP accepted (DEF CON AI Village, USENIX, etc.)

---

## Framework Mapping

| Framework | Usage |
|-----------|-------|
| OWASP MCP Top 10 | Primary vulnerability taxonomy for all scanner categories (all 10 categories mapped, verified 2026-03-18) |
| OWASP Top 10 for Agentic AI | Agentic attack classification — all 10 ASI categories mapped |
| MITRE ATLAS | Adversarial ML technique mapping — all 10 categories mapped, verified against ATLAS.yaml v5.4.0 |
| CWE | Weakness enumeration for SARIF consumers and security tooling — all 10 categories mapped |

Framework mappings are maintained in `src/q_ai/core/data/frameworks.yaml` (static;
the former `qai update-frameworks` updater was removed in the CTPF Reconnect strip).

---

## Out of Scope (for now)

- **Cloud-hosted version.** {q-AI} is a local lab tool. No SaaS, no multi-tenant, no hosted scanning.
- **GUI installer.** ~~Target audience is security researchers comfortable with Python tooling.~~ Desktop application packaging was a planned direction but is now deferred in favor of scenario format and publication pipeline work — see `Plans/desktop-packaging-plan.md`.
- **Real-time collaboration.** Single-operator tool. Research sharing happens via exported findings and published reports.
- **LLM output testing.** That's Garak's problem. {q-AI} tests infrastructure and agent behaviour, not model outputs.

---

## Reference Links

- OWASP MCP Top 10: https://owasp.org/www-project-mcp-top-10/
- OWASP Top 10 for Agentic AI: https://genai.owasp.org/
- MCP Specification: https://modelcontextprotocol.io/
- MITRE ATLAS: https://atlas.mitre.org/
- Garak: https://github.com/NVIDIA/garak
- PyRIT: https://github.com/Azure/PyRIT
