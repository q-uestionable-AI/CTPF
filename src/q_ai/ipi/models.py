"""IPI-specific data models for q_ai.

This module defines the IPI-specific enums used for indirect prompt injection
testing: document formats, hiding techniques, payload styles, and payload types.

Shared models (Campaign, Hit, HitConfidence) live here alongside the enums
since q_ai consolidates all IPI models in a single file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Self


class Format(StrEnum):
    """Supported document formats for payload generation.

    Each format supports different hiding techniques appropriate
    to its structure and typical processing pipelines.

    Attributes:
        PDF: Portable Document Format - Phase 1 & 2 techniques.
        IMAGE: PNG/JPG images - VLM attack surface (visible, subtle, EXIF).
        MARKDOWN: Markdown documents - HTML comments, zero-width chars.
        HTML: HTML documents - comments, hidden divs, CSS tricks.
        DOCX: Word documents - hidden text, comments, metadata.
        ICS: Calendar files - description, VALARM, X-properties.
        EML: Email files - headers, hidden HTML, attachments.

    Example:
        >>> from q_ai.ipi.models import Format
        >>> fmt = Format.PDF
        >>> fmt.value
        'pdf'
    """

    PDF = "pdf"
    IMAGE = "image"
    MARKDOWN = "markdown"
    HTML = "html"
    DOCX = "docx"
    ICS = "ics"
    EML = "eml"


class Technique(StrEnum):
    """Payload hiding techniques organized by format and phase.

    Control Condition:
        NONE: Payload rendered as normal visible text. Used as a control
            condition for measuring hiding technique uplift. Applies to
            every format.

    PDF Techniques (Phase 1 - Basic):
        WHITE_INK: White text on white background - invisible but extractable.
        OFF_CANVAS: Text positioned outside visible page boundaries.
        METADATA: Payload stored in PDF metadata fields (Title, Author, etc.).

    PDF Techniques (Phase 2 - Advanced):
        TINY_TEXT: 0.5pt font - below human visual threshold but parseable.
        WHITE_RECT: Text covered by opaque white rectangle overlay.
        FORM_FIELD: Hidden AcroForm field with payload as value.
        ANNOTATION: PDF annotation/comment layer containing payload.
        JAVASCRIPT: Document-level JavaScript action with embedded payload.
        EMBEDDED_FILE: Hidden file attachment stream within PDF.
        INCREMENTAL: Payload in PDF incremental update/custom metadata section.

    Image Techniques (Phase 3 - VLM Attack Surface):
        VISIBLE_TEXT: Human-readable text overlay on image.
        SUBTLE_TEXT: Low contrast, small font, or edge-placed text.
        EXIF_METADATA: Payload in EXIF metadata fields.

    Markdown Techniques (Phase 3):
        HTML_COMMENT: Payload in HTML comment tags (<!-- -->).
        LINK_REFERENCE: Payload in link reference definition.
        ZERO_WIDTH: Payload encoded using zero-width Unicode characters.
        HIDDEN_BLOCK: Payload in hidden HTML block (div with display:none).

    HTML Techniques (Phase 3):
        SCRIPT_COMMENT: Payload in JavaScript comment inside script tag.
        CSS_OFFSCREEN: Payload in element positioned off-screen with CSS.
        DATA_ATTRIBUTE: Payload in HTML data-* attribute.
        META_TAG: Payload in HTML meta tag content.

    DOCX Techniques (Phase 3):
        DOCX_HIDDEN_TEXT: Payload in text with hidden font attribute.
        DOCX_TINY_TEXT: Payload in 0.5pt font (below visual threshold).
        DOCX_WHITE_TEXT: White text on white background.
        DOCX_COMMENT: Payload in Word comment/annotation.
        DOCX_METADATA: Payload in document core properties.
        DOCX_HEADER_FOOTER: Payload in document header or footer.

    ICS Techniques (Phase 3 - Calendar Invite Attack Surface):
        ICS_DESCRIPTION: Payload in event DESCRIPTION property.
        ICS_LOCATION: Payload in event LOCATION property.
        ICS_VALARM: Payload in VALARM reminder DESCRIPTION.
        ICS_X_PROPERTY: Payload in custom X- extension property.

    EML Techniques (Phase 3 - Email Attack Surface):
        EML_X_HEADER: Payload in custom X- email header.
        EML_HTML_HIDDEN: Payload in hidden HTML div (display:none).
        EML_ATTACHMENT: Payload in text file attachment.

    Example:
        >>> from q_ai.ipi.models import Technique
        >>> technique = Technique.WHITE_INK
        >>> technique.value
        'white_ink'
    """

    # Control condition (no hiding)
    NONE = "none"

    # PDF Phase 1 techniques
    WHITE_INK = "white_ink"
    OFF_CANVAS = "off_canvas"
    METADATA = "metadata"

    # PDF Phase 2 techniques
    TINY_TEXT = "tiny_text"
    WHITE_RECT = "white_rect"
    FORM_FIELD = "form_field"
    ANNOTATION = "annotation"
    JAVASCRIPT = "javascript"
    EMBEDDED_FILE = "embedded_file"
    INCREMENTAL = "incremental"

    # Image Phase 3 techniques (VLM attack surface)
    VISIBLE_TEXT = "visible_text"
    SUBTLE_TEXT = "subtle_text"
    EXIF_METADATA = "exif_metadata"

    # Markdown Phase 3 techniques
    HTML_COMMENT = "html_comment"
    LINK_REFERENCE = "link_reference"
    ZERO_WIDTH = "zero_width"
    HIDDEN_BLOCK = "hidden_block"

    # HTML Phase 3 techniques
    SCRIPT_COMMENT = "script_comment"
    CSS_OFFSCREEN = "css_offscreen"
    DATA_ATTRIBUTE = "data_attribute"
    META_TAG = "meta_tag"

    # DOCX Phase 3 techniques
    DOCX_HIDDEN_TEXT = "docx_hidden_text"
    DOCX_TINY_TEXT = "docx_tiny_text"
    DOCX_WHITE_TEXT = "docx_white_text"
    DOCX_COMMENT = "docx_comment"
    DOCX_METADATA = "docx_metadata"
    DOCX_HEADER_FOOTER = "docx_header_footer"

    # ICS Phase 3 techniques (calendar invite attack surface)
    ICS_DESCRIPTION = "ics_description"
    ICS_LOCATION = "ics_location"
    ICS_VALARM = "ics_valarm"
    ICS_X_PROPERTY = "ics_x_property"

    # EML Phase 3 techniques (email attack surface)
    EML_X_HEADER = "eml_x_header"
    EML_HTML_HIDDEN = "eml_html_hidden"
    EML_ATTACHMENT = "eml_attachment"


class PayloadStyle(StrEnum):
    """Payload content styles for social engineering effectiveness.

    Styles range from obvious (for baseline testing) to subtle (for realistic
    attack simulation). More subtle styles blend the injection instruction
    into plausible document content.

    Attributes:
        OBVIOUS: Direct injection markers - easily detectable baseline.
        CITATION: Disguised as document reference or citation.
        REVIEWER: Appears as note to document reviewer/editor.
        HELPFUL: Framed as helpful supplementary resource.
        ACADEMIC: Academic or research cross-reference format.
        COMPLIANCE: Looks like compliance or verification requirement.
        DATASOURCE: Appears as data source attribution.

    Example:
        >>> from q_ai.ipi.models import PayloadStyle
        >>> style = PayloadStyle.CITATION
        >>> style.value
        'citation'
    """

    OBVIOUS = "obvious"
    CITATION = "citation"
    REVIEWER = "reviewer"
    HELPFUL = "helpful"
    ACADEMIC = "academic"
    COMPLIANCE = "compliance"
    DATASOURCE = "datasource"


class DocumentTemplate(StrEnum):
    """Document context templates aligning payload framing to source probes.

    Each template (except GENERIC) frames the injection within a realistic
    document context borrowed from published probe suites (Garak, BIPIA).
    GENERIC preserves legacy behavior with no document-context framing.

    Garak-aligned members:
        WHOIS: WHOIS domain record assessment (LatentWhois).
        TRANSLATION_EN_FR: English→French translation task.
        TRANSLATION_EN_ZH: English→Chinese translation task.
        LEGAL_SNIPPET: Legal text summarization.
        REPORT: Business/financial report summarization.
        RESUME: Resume/CV candidate assessment.

    BIPIA-aligned members:
        EMAIL: Email QA task.
        WEB: Web article QA task.
        TABLE: Markdown table QA task.
        CODE: Code error QA task.
        NEWS: News article summarization.

    Special:
        GENERIC: No document-context framing (default).

    Example:
        >>> from q_ai.ipi.models import DocumentTemplate
        >>> DocumentTemplate.WHOIS.value
        'whois'
    """

    GENERIC = "generic"

    WHOIS = "whois"
    TRANSLATION_EN_FR = "translation_en_fr"
    TRANSLATION_EN_ZH = "translation_en_zh"
    LEGAL_SNIPPET = "legal_snippet"
    REPORT = "report"
    RESUME = "resume"

    EMAIL = "email"
    WEB = "web"
    TABLE = "table"
    CODE = "code"
    NEWS = "news"


class PayloadType(StrEnum):
    """Payload objectives defining the attack goal.

    Types represent different risk levels and attack objectives,
    from proof-of-execution callbacks to more dangerous actions.

    Attributes:
        CALLBACK: Simple HTTP callback - proof of execution (default, safe).
        EXFIL_SUMMARY: Attempts to exfiltrate document summary via callback.
        EXFIL_CONTEXT: Attempts to exfiltrate conversation context.
        SSRF_INTERNAL: Server-side request forgery to internal endpoints.
        INSTRUCTION_OVERRIDE: Attempts to override system instructions.
        TOOL_ABUSE: Attempts to misuse agent tools/capabilities.
        PERSISTENCE: Attempts to persist instructions across sessions.

    Note:
        Non-callback payload types require the --dangerous CLI flag and are
        intended for authorized security testing only.

    Example:
        >>> from q_ai.ipi.models import PayloadType
        >>> ptype = PayloadType.CALLBACK
        >>> ptype.value
        'callback'
    """

    CALLBACK = "callback"
    EXFIL_SUMMARY = "exfil_summary"
    EXFIL_CONTEXT = "exfil_context"
    SSRF_INTERNAL = "ssrf_internal"
    INSTRUCTION_OVERRIDE = "instruction_override"
    TOOL_ABUSE = "tool_abuse"
    PERSISTENCE = "persistence"


class CitationFrame(StrEnum):
    """Callback-line framing choice for CITATION-style CALLBACK payloads.

    Controls which CITATION text ``q_ai.ipi.sweep_service.render_sweep_prompt``
    emits when the sweep case is ``style=CITATION`` and
    ``payload_type=CALLBACK``. Any other style/type combination ignores this
    value — see Campaign 1 Phase 4 Step 3 and
    ``Archive/RFC/RFC-IPI-Template-Callback-Rationale.md``.

    Attributes:
        PLAIN: Pre-4.5 hardcoded CITATION sentence. Ignores the hosting
            template's ``callback_role``; the same sentence renders across
            every template, producing a report-context baseline suitable as
            a control condition.
        TEMPLATE_AWARE: Post-4.5 composed CITATION sentence that interpolates
            the hosting template's ``callback_role`` noun phrase. Default;
            matches the behavior shipped since PR #121.

    Example:
        >>> from q_ai.ipi.models import CitationFrame
        >>> CitationFrame.TEMPLATE_AWARE.value
        'template-aware'
    """

    PLAIN = "plain"
    TEMPLATE_AWARE = "template-aware"


class HitConfidence(StrEnum):
    """Confidence level for callback hit authenticity.

    Used to distinguish genuine agent callbacks from scanner noise,
    based on token validation, User-Agent analysis, and request shape.

    Attributes:
        HIGH: Valid campaign token present — strong proof of execution.
        MEDIUM: No/invalid token, but programmatic User-Agent (python-requests, etc.).
        LOW: No/invalid token, browser or scanner User-Agent.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class TemplateSpec:
    """Metadata and content structure for a document context template.

    A ``TemplateSpec`` pairs a ``DocumentTemplate`` enum value with the
    source provenance and the text skeleton used when rendering a payload
    document. Phase 4.2 populates specs with stub content that Phase 4.3
    replaces with fully source-aligned text.

    Attributes:
        id: The ``DocumentTemplate`` this spec describes.
        name: Human-readable template name (e.g., "WHOIS Record Assessment").
        description: Short CLI help description.
        source_tool: Originating project — "garak", "bipia", or "generic".
        source_reference: Probe class or dataset identifier
            (e.g., "LatentWhois").
        source_commit: Git commit hash of the source repo pinned for
            alignment verification. Empty string for GENERIC.
        top_instruction: Task framing text placed before the document body
            (may be empty string).
        context_template: Document body template containing a ``{payload}``
            marker at the injection point.
        formats: ``Format`` values compatible with this template.
        default_style: Suggested ``PayloadStyle`` for this template.
        callback_role: Noun phrase identifying the role this URL plays
            within the template's document context (e.g., 'the registrar
            enrichment feed' for a WHOIS record). Used as the ``{source}``
            substitution in style-based CALLBACK payloads. Empty string
            disables template-aware rationale (falls through to style text
            as written).
    """

    id: DocumentTemplate
    name: str
    description: str
    source_tool: str
    source_reference: str
    source_commit: str
    top_instruction: str
    context_template: str
    formats: tuple[Format, ...]
    default_style: PayloadStyle
    callback_role: str = ""


@dataclass
class Campaign:
    """A generated payload campaign tracking document and callback info.

    Attributes:
        id: DB primary key (uuid4().hex).
        uuid: Campaign UUID (used in callback URL).
        token: Per-campaign auth secret.
        filename: Generated document filename.
        format: Document format (e.g., "pdf", "image", "markdown").
        technique: Hiding technique used (e.g., "white_ink", "metadata").
        callback_url: Full URL that will be triggered if payload executes.
        output_path: Full path to generated file. None for legacy campaigns.
        payload_style: Social engineering style (e.g., "obvious", "citation").
        payload_type: Attack objective type (e.g., "callback", "exfil_summary").
        run_id: FK to runs table.
        template_id: Document context template alias (e.g., 'whois', 'generic').
            None for legacy campaigns created before the template system.
        created_at: UTC timestamp when campaign was created.
    """

    id: str
    uuid: str
    token: str
    filename: str
    format: str
    technique: str
    callback_url: str
    output_path: str | None = None
    payload_style: str = "obvious"
    payload_type: str = "callback"
    run_id: str | None = None
    template_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict.

        Returns:
            Dictionary with all fields; datetimes are ISO-format strings.
        """
        return {
            "id": self.id,
            "uuid": self.uuid,
            "token": self.token,
            "filename": self.filename,
            "format": self.format,
            "technique": self.technique,
            "callback_url": self.callback_url,
            "output_path": self.output_path,
            "payload_style": self.payload_style,
            "payload_type": self.payload_type,
            "run_id": self.run_id,
            "template_id": self.template_id,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Self:
        """Construct from a sqlite3.Row dict.

        Args:
            row: Dictionary from sqlite3.Row with ISO datetime strings.

        Returns:
            Campaign instance with proper Python types.
        """
        created_at_raw = row.get("created_at")
        return cls(
            id=row["id"],
            uuid=row["uuid"],
            token=row["token"],
            filename=row["filename"],
            format=row["format"],
            technique=row["technique"],
            callback_url=row["callback_url"],
            output_path=row.get("output_path"),
            payload_style=row.get("payload_style", "obvious"),
            payload_type=row.get("payload_type", "callback"),
            run_id=row.get("run_id"),
            template_id=row.get("template_id"),
            created_at=datetime.fromisoformat(created_at_raw)
            if isinstance(created_at_raw, str)
            else datetime.now(UTC),
        )


@dataclass
class Hit:
    """A callback hit received from an AI agent executing a payload.

    Records details of incoming HTTP requests to the callback server,
    providing proof-of-execution evidence.

    Attributes:
        id: DB primary key (uuid4().hex).
        uuid: Campaign UUID this hit belongs to.
        source_ip: IP address of the requesting client.
        user_agent: HTTP User-Agent header value.
        headers: JSON-serialized headers dict.
        confidence: Hit confidence level based on token validity and request analysis.
        timestamp: UTC timestamp when hit was received.
        body: Captured request data (query params for GET, body for POST).
        token_valid: Whether the campaign authentication token was present and valid.
        via_tunnel: True when ``source_ip`` was resolved from a trusted forwarded
            header (tunnel mode). False when it came from the TCP peer (direct).
    """

    id: str
    uuid: str
    source_ip: str
    user_agent: str
    headers: str
    confidence: HitConfidence
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    body: str | None = None
    token_valid: bool = False
    via_tunnel: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict.

        Returns:
            Dictionary with all fields; datetimes are ISO-format strings.
        """
        return {
            "id": self.id,
            "uuid": self.uuid,
            "source_ip": self.source_ip,
            "user_agent": self.user_agent,
            "headers": self.headers,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
            "body": self.body,
            "token_valid": self.token_valid,
            "via_tunnel": self.via_tunnel,
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Self:
        """Construct from a sqlite3.Row dict.

        Args:
            row: Dictionary from sqlite3.Row with ISO datetime strings and
                integer booleans.

        Returns:
            Hit instance with proper Python types.
        """
        timestamp_raw = row.get("timestamp")
        return cls(
            id=row["id"],
            uuid=row["uuid"],
            source_ip=row["source_ip"],
            user_agent=row["user_agent"],
            headers=row["headers"],
            confidence=HitConfidence(row["confidence"]),
            timestamp=datetime.fromisoformat(timestamp_raw)
            if isinstance(timestamp_raw, str)
            else datetime.now(UTC),
            body=row.get("body"),
            token_valid=bool(row.get("token_valid", 0)),
            via_tunnel=bool(row.get("via_tunnel", 0)),
        )


__all__ = [
    "Campaign",
    "CitationFrame",
    "DocumentTemplate",
    "Format",
    "Hit",
    "HitConfidence",
    "PayloadStyle",
    "PayloadType",
    "Technique",
    "TemplateSpec",
]
