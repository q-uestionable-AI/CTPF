"""Data models for inject MCP fixture payloads.

Defines technique categories and reusable payload templates used to build
malicious MCP servers for controlled security research.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class InjectionTechnique(StrEnum):
    """Categories of tool-based injection techniques.

    Attributes:
        DESCRIPTION_POISONING: Hostile instructions embedded in tool
            descriptions.
        OUTPUT_INJECTION: Hostile content returned in tool responses.
        CROSS_TOOL_ESCALATION: Multi-tool chaining or shadow-tool patterns.
    """

    DESCRIPTION_POISONING = "description_poisoning"
    OUTPUT_INJECTION = "output_injection"
    CROSS_TOOL_ESCALATION = "cross_tool_escalation"


@dataclass
class PayloadTemplate:
    """A reusable injection payload definition for fixture MCP servers.

    Attributes:
        name: Unique identifier for this payload.
        technique: The injection technique category.
        description: Human-readable description of what this payload tests.
        owasp_ids: Relevant OWASP MCP Top 10 / Agentic AI Top 10 IDs.
        target_agents: Agent types this payload is designed for
            (empty = universal).
        relevant_categories: Audit finding categories this payload relates
            to (empty = universal).
        tool_name: MCP tool name to register.
        tool_description: The poisoned tool description text.
        tool_params: Tool input schema mapping param name to
            ``{type, description}``.
        tool_response: Response template string with ``{param_name}``
            substitution.
        test_query: Optional sample user message for manual fixture use.
    """

    name: str
    technique: InjectionTechnique
    description: str
    owasp_ids: list[str] = field(default_factory=list)
    target_agents: list[str] = field(default_factory=list)
    relevant_categories: list[str] = field(default_factory=list)
    tool_name: str = ""
    tool_description: str = ""
    tool_params: dict[str, dict[str, str]] = field(default_factory=dict)
    tool_response: str = ""
    test_query: str = ""
