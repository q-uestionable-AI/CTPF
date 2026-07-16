"""Authenticated local policy and per-run authorization grants."""

from __future__ import annotations

import base64
import datetime
import hashlib
import hmac
import secrets
import uuid

from ctpf.automation.canonical import canonical_bytes, sha256_digest
from ctpf.automation.contracts import (
    AuthorizationGrant,
    ContractError,
    DecisionKind,
    GrantSource,
    PolicyDecision,
    PolicyDocument,
    RunSpec,
)
from ctpf.core.config import delete_local_secret, get_local_secret, set_local_secret

_APPROVAL_SECRET_NAME = "automation-approval-key-v1"  # noqa: S105 - keyring account name
_KEY_BYTES = 32
_POLICY_DOMAIN = b"ctpf-policy-v1\x00"
_GRANT_DOMAIN = b"ctpf-authorization-grant-v1\x00"


class ApprovalError(RuntimeError):
    """Raised when authenticated authorization material is invalid."""


def initialize_approval_key() -> str:
    """Create the local signing key if absent and return its public key ID."""
    encoded = get_local_secret(_APPROVAL_SECRET_NAME)
    if encoded is None:
        key = secrets.token_bytes(_KEY_BYTES)
        set_local_secret(_APPROVAL_SECRET_NAME, _encode_key(key))
    else:
        key = _decode_key(encoded)
    return _key_id(key)


def delete_approval_key() -> None:
    """Delete the local signing key, invalidating all signed control records."""
    delete_local_secret(_APPROVAL_SECRET_NAME)


def approval_key_id() -> str:
    """Return the current local signing-key identifier without exposing the key."""
    return _key_id(_load_key())


def sign_policy(policy: PolicyDocument) -> tuple[str, str]:
    """Return the HMAC signature and key ID for one canonical policy body."""
    _validate_policy_contract(policy)
    key = _load_key()
    return _sign(_POLICY_DOMAIN, policy.to_payload(), key), _key_id(key)


def authenticate_policy(
    policy: PolicyDocument,
    signature: str,
    key_id: str,
    *,
    now: datetime.datetime | None = None,
) -> str:
    """Authenticate a policy body and active validity interval.

    Args:
        policy: Strict parsed policy body.
        signature: Stored lowercase HMAC-SHA-256 signature.
        key_id: Stored SHA-256 identifier of the signing key.
        now: Optional aware evaluation time.

    Returns:
        Canonical policy digest.

    Raises:
        ApprovalError: If authentication or temporal validity fails.
    """
    _validate_policy_contract(policy)
    key = _load_key()
    if not hmac.compare_digest(key_id, _key_id(key)):
        raise ApprovalError("policy signing-key identifier does not match")
    _verify_signature(_POLICY_DOMAIN, policy.to_payload(), signature, key, "policy")
    current = _aware_utc(now)
    created = _parse_timestamp(policy.created_at, "policy created_at")
    expires = _parse_timestamp(policy.expires_at, "policy expires_at")
    if created >= expires or current < created or current >= expires:
        raise ApprovalError("policy is outside its active validity interval")
    return sha256_digest(policy.to_payload())


def issue_authorization_grant(  # noqa: PLR0913 - exact authorization bindings
    spec: RunSpec,
    policy: PolicyDocument,
    decision: PolicyDecision,
    source: GrantSource,
    *,
    policy_signature: str,
    policy_key_id: str,
    lifetime_seconds: int,
    issued_at: datetime.datetime | None = None,
) -> tuple[AuthorizationGrant, str]:
    """Issue and sign one short-lived grant bound to an exact RunSpec.

    Policy signature and status must be authenticated before this function is
    called. Human-per-run grants must only be issued by the future governance
    path after an interactive approval.
    """
    _validate_spec_contract(spec)
    _validate_policy_contract(policy)
    _validate_decision_source(decision, source)
    issued = _aware_utc(issued_at)
    authenticate_policy(policy, policy_signature, policy_key_id, now=issued)
    policy_digest = sha256_digest(policy.to_payload())
    spec_digest = sha256_digest(spec.to_payload())
    if (
        decision.spec_digest != spec_digest
        or decision.policy_digest != policy_digest
        or spec.policy_id != policy.policy_id
    ):
        raise ApprovalError("policy decision does not match the RunSpec and policy")
    if decision.reason_code != "policy_match" or not decision.minimum_reservations.is_within(
        spec.limits
    ):
        raise ApprovalError("policy decision does not authorize the requested resources")
    authorized_tiers = (
        policy.standing_tiers if source == GrantSource.STANDING_POLICY else policy.per_run_tiers
    )
    if spec.requested_tier not in authorized_tiers:
        raise ApprovalError("grant source cannot authorize the requested tier")
    if not 1 <= lifetime_seconds <= policy.limits.approval_lifetime_seconds:
        raise ApprovalError("grant lifetime exceeds the policy approval lifetime")
    key = _load_key()
    expires = issued + datetime.timedelta(seconds=lifetime_seconds)
    policy_expires = _parse_timestamp(policy.expires_at, "policy expires_at")
    policy_created = _parse_timestamp(policy.created_at, "policy created_at")
    if issued < policy_created or issued >= policy_expires:
        raise ApprovalError("grant issuance time is outside the policy validity interval")
    if expires > policy_expires:
        raise ApprovalError("grant would outlive its authorizing policy")
    grant = AuthorizationGrant(
        grant_id=uuid.uuid4().hex,
        source=source,
        spec_digest=spec_digest,
        policy_id=policy.policy_id,
        policy_digest=policy_digest,
        scenario_fingerprint=spec.experiment.scenario_fingerprint,
        targets=spec.experiment.targets,
        authorized_tier=spec.requested_tier,
        limits=spec.limits,
        issued_at=_format_timestamp(issued),
        expires_at=_format_timestamp(expires),
        nonce=secrets.token_hex(32),
        key_id=_key_id(key),
    )
    return grant, _sign(_GRANT_DOMAIN, grant.to_payload(), key)


def authenticate_authorization_grant(
    grant: AuthorizationGrant,
    signature: str,
    spec: RunSpec,
    policy: PolicyDocument,
    *,
    now: datetime.datetime | None = None,
) -> None:
    """Authenticate a grant and its exact RunSpec and policy bindings."""
    _validate_grant_contract(grant)
    _validate_spec_contract(spec)
    _validate_policy_contract(policy)
    key = _load_key()
    if not hmac.compare_digest(grant.key_id, _key_id(key)):
        raise ApprovalError("grant signing-key identifier does not match")
    _verify_signature(_GRANT_DOMAIN, grant.to_payload(), signature, key, "grant")
    _validate_grant_times(grant, policy, _aware_utc(now))
    expected = (
        sha256_digest(spec.to_payload()),
        spec.policy_id,
        sha256_digest(policy.to_payload()),
        spec.experiment.scenario_fingerprint,
        spec.experiment.targets,
        spec.requested_tier,
        spec.limits,
    )
    actual = (
        grant.spec_digest,
        grant.policy_id,
        grant.policy_digest,
        grant.scenario_fingerprint,
        grant.targets,
        grant.authorized_tier,
        grant.limits,
    )
    if actual != expected:
        raise ApprovalError("grant does not match the exact RunSpec and policy")
    authorized_tiers = (
        policy.standing_tiers
        if grant.source == GrantSource.STANDING_POLICY
        else policy.per_run_tiers
    )
    if grant.authorized_tier not in authorized_tiers:
        raise ApprovalError("grant source cannot authorize its bound tier")


def _validate_decision_source(decision: PolicyDecision, source: GrantSource) -> None:
    expected = {
        DecisionKind.ALLOWED_STANDING_POLICY: GrantSource.STANDING_POLICY,
        DecisionKind.APPROVAL_REQUIRED: GrantSource.HUMAN_PER_RUN,
    }.get(decision.kind)
    if expected is None:
        raise ApprovalError("a denied decision cannot authorize a grant")
    if source != expected:
        raise ApprovalError("grant source does not match the policy decision")


def _validate_grant_times(
    grant: AuthorizationGrant,
    policy: PolicyDocument,
    now: datetime.datetime,
) -> None:
    issued = _parse_timestamp(grant.issued_at, "grant issued_at")
    expires = _parse_timestamp(grant.expires_at, "grant expires_at")
    policy_created = _parse_timestamp(policy.created_at, "policy created_at")
    policy_expires = _parse_timestamp(policy.expires_at, "policy expires_at")
    lifetime = (expires - issued).total_seconds()
    if issued < policy_created or issued >= expires or expires > policy_expires:
        raise ApprovalError("grant validity interval is invalid")
    if lifetime > policy.limits.approval_lifetime_seconds:
        raise ApprovalError("grant lifetime exceeds the policy approval lifetime")
    if now < policy_created or now < issued or now >= expires or now >= policy_expires:
        raise ApprovalError("grant is outside its active validity interval")


def _validate_policy_contract(policy: PolicyDocument) -> None:
    try:
        parsed = PolicyDocument.from_payload(policy.to_payload())
    except (AttributeError, ContractError, TypeError, ValueError) as exc:
        raise ApprovalError(f"policy contract is invalid: {exc}") from exc
    if parsed != policy:
        raise ApprovalError("policy contract is not normalized")


def _validate_spec_contract(spec: RunSpec) -> None:
    try:
        parsed = RunSpec.from_payload(spec.to_payload())
    except (AttributeError, ContractError, TypeError, ValueError) as exc:
        raise ApprovalError(f"RunSpec contract is invalid: {exc}") from exc
    if parsed != spec:
        raise ApprovalError("RunSpec contract is not normalized")


def _validate_grant_contract(grant: AuthorizationGrant) -> None:
    try:
        parsed = AuthorizationGrant.from_payload(grant.to_payload())
    except (AttributeError, ContractError, TypeError, ValueError) as exc:
        raise ApprovalError(f"authorization grant contract is invalid: {exc}") from exc
    if parsed != grant:
        raise ApprovalError("authorization grant contract is not normalized")


def _load_key() -> bytes:
    encoded = get_local_secret(_APPROVAL_SECRET_NAME)
    if encoded is None:
        raise ApprovalError("automation approval key is not initialized")
    return _decode_key(encoded)


def _encode_key(key: bytes) -> str:
    return base64.urlsafe_b64encode(key).decode("ascii")


def _decode_key(encoded: str) -> bytes:
    try:
        key = base64.b64decode(encoded, altchars=b"-_", validate=True)
    except (ValueError, TypeError) as exc:
        raise ApprovalError("stored automation approval key is malformed") from exc
    if len(key) != _KEY_BYTES:
        raise ApprovalError("stored automation approval key has an invalid length")
    return key


def _key_id(key: bytes) -> str:
    return hashlib.sha256(key).hexdigest()


def _sign(domain: bytes, payload: dict[str, object], key: bytes) -> str:
    return hmac.new(key, domain + canonical_bytes(payload), hashlib.sha256).hexdigest()


def _verify_signature(
    domain: bytes,
    payload: dict[str, object],
    signature: str,
    key: bytes,
    label: str,
) -> None:
    expected = _sign(domain, payload, key)
    if not hmac.compare_digest(signature, expected):
        raise ApprovalError(f"{label} signature is invalid")


def _aware_utc(value: datetime.datetime | None) -> datetime.datetime:
    current = value or datetime.datetime.now(datetime.UTC)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ApprovalError("authorization time must be timezone-aware")
    return current.astimezone(datetime.UTC).replace(microsecond=0)


def _parse_timestamp(value: str, label: str) -> datetime.datetime:
    try:
        return datetime.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.UTC)
    except ValueError as exc:
        raise ApprovalError(f"{label} is invalid") from exc


def _format_timestamp(value: datetime.datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")
