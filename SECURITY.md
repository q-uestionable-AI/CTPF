# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in {q-AI}, please report it responsibly.

**Preferred:** Use [GitHub Private Vulnerability Reporting](https://github.com/q-uestionable-AI/qai/security/advisories/new) — click "Report a vulnerability" in the Security tab. This keeps coordination on-platform and follows the [OpenSSF Vulnerability Disclosure Guide](https://github.com/ossf/oss-vulnerability-guide).

**Alternative:** Email **security@q-uestionable.ai** with a description of the vulnerability, steps to reproduce, and potential impact assessment.

1. **Do not** open a public GitHub issue for security vulnerabilities
2. Allow up to 72 hours for initial response
3. We will coordinate disclosure timeline with you

## Scope

{q-AI} is a security testing tool. Vulnerabilities in the tool itself (not in targets being tested) are in scope:

- Command injection in CLI argument handling
- Credential leakage in reports, logs, or local artifacts
- API key exposure via the keyring integration or config files
- Dependency vulnerabilities with exploitable paths
- Unsafe deserialization of findings, MCP messages, or fixture payloads
- Local listener issues (IPI headless callback, proxy HTTP listen adapters) including SSRF or path traversal in file outputs

Out of scope: vulnerabilities in third-party MCP servers or AI systems discovered *by* the tool — those should be reported to the relevant vendor.

## Evidence at Rest

{q-AI} stores run artifacts locally in plaintext under `~/.qai/`. The SQLite database at `~/.qai/qai.db` includes IPI callback hit bodies, headers, source IPs, user agents, proxy session data, and other finding evidence. Payload documents, exports, and backups live alongside it under the same directory.

Access control relies on filesystem permissions. On POSIX, `~/.qai/` is created with mode `0o700`; any pre-existing wider mode is narrowed on the next qai startup. On Windows the default user-profile ACLs apply — qai does not set additional ACL restrictions there.

To purge or rotate evidence: `qai db reset` wipes the database entirely; `qai runs delete <run_id>` removes a specific run and its findings; `qai db backup` takes a timestamped snapshot before destructive operations.

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |
