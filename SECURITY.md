# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in CTPF Research Harness, please report it responsibly.

**Preferred:** Use [GitHub Private Vulnerability Reporting](https://github.com/q-uestionable-AI/CTPF/security/advisories/new) — click "Report a vulnerability" in the Security tab. This keeps coordination on-platform and follows the [OpenSSF Vulnerability Disclosure Guide](https://github.com/ossf/oss-vulnerability-guide).

**Alternative:** Email **security@q-uestionable.ai** with a description of the vulnerability, steps to reproduce, and potential impact assessment.

1. **Do not** open a public GitHub issue for security vulnerabilities
2. Allow up to 72 hours for initial response
3. We will coordinate disclosure timeline with you

## Scope

CTPF Research Harness is a security testing tool. Vulnerabilities in the tool itself (not in targets being tested) are in scope:

- Command injection in CLI argument handling
- Credential leakage in reports, logs, or local artifacts
- API key exposure via the keyring integration or config files
- Dependency vulnerabilities with exploitable paths
- Unsafe deserialization of findings, MCP messages, experiment evidence, or fixture payloads
- Local proxy-listener issues including SSRF or path traversal in file outputs
- Bypass of the agent-operable control/govern boundary on source `main` (forged campaign policy or
  approval, unlisted or post-authorization RunSpec execution, budget or cancellation races,
  evidence path escape)

Out of scope: vulnerabilities in third-party MCP servers or AI systems discovered *by* the tool — those should be reported to the relevant vendor.

## Autonomous-caller threat model

Source `main` treats the AI caller as untrusted. Protected assets are the signed local policy and
approval records, finite campaign RunSpec allowlists, OS-keyring secrets, approved output roots,
exact target and scenario fingerprints, resource budgets, and research evidence integrity.

Enforcement assumptions:

- Control contracts reject unknown fields; there is no arbitrary URL, shell, or proxy command on the
  autonomous surface.
- Policy and approval use HMAC-SHA-256 with a keyring-held local secret. Bounded-remote standing
  policy authorizes only its exact RunSpec digests. A process that can read the key can forge local
  authority; protect the host accordingly.
- Budgets and deadlines are reserved before effect boundaries. Cancellation is durable; interrupted
  leases do not auto-resume.
- Evidence verification establishes internal consistency only, not independent authenticity or
  scientific validity.
- Local HTTP listeners bind to `127.0.0.1` only. Network and data-egress classes are constrained by
  signed policy and target identity.

Unsafe deployment: running an unconstrained full-shell agent beside CTPF without an external
OS/runtime sandbox. That residual is a deployment requirement, not a harness claim of containment.

## Evidence at Rest

CTPF Research Harness stores operational records locally in plaintext under `~/.ctpf/`. The SQLite database at `~/.ctpf/ctpf.db` includes proxy session data and other finding evidence; upgraded databases may also retain historical records from removed modules. Controlled experiment traces and evidence bundles are written to the operator-selected directory outside the Git checkout.

Access control relies on filesystem permissions. On POSIX, `~/.ctpf/` is created with mode `0o700`; any pre-existing wider mode is narrowed on the next harness startup. On Windows the default user-profile ACLs apply — the harness does not set additional ACL restrictions there.

Experiment traces and bundles can contain model output, tool arguments, protocol messages, and
operator-supplied target metadata. Review an evidence bundle before sharing or publishing it even
when its built-in secret-pattern scan reports no hits.

To purge or rotate evidence: `ctpf db reset` wipes the database entirely; `ctpf runs delete <run_id>` removes a specific run and its findings; `ctpf db backup` takes a timestamped snapshot before destructive operations.

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.14.x  | Yes       |
| 0.13.x and earlier | No |

Unreleased `main` is development source, not a separately supported `0.14.0` distribution. Security
fixes target the latest supported release line and current development branch.
