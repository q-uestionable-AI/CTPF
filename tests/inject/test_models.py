"""Tests for inject module data models (fixtures-only surface)."""

from __future__ import annotations

from q_ai.inject.models import InjectionTechnique, PayloadTemplate


class TestInjectionTechnique:
    """InjectionTechnique enum values."""

    def test_known_values(self) -> None:
        """Enum exposes the three fixture technique categories."""
        assert InjectionTechnique.DESCRIPTION_POISONING.value == "description_poisoning"
        assert InjectionTechnique.OUTPUT_INJECTION.value == "output_injection"
        assert InjectionTechnique.CROSS_TOOL_ESCALATION.value == "cross_tool_escalation"


class TestPayloadTemplate:
    """PayloadTemplate dataclass construction and defaults."""

    def test_minimal_construction(self) -> None:
        """Construct with only required fields, verify defaults."""
        t = PayloadTemplate(
            name="test",
            technique=InjectionTechnique.DESCRIPTION_POISONING,
            description="A test payload",
        )
        assert t.name == "test"
        assert t.technique == InjectionTechnique.DESCRIPTION_POISONING
        assert t.description == "A test payload"
        assert t.owasp_ids == []
        assert t.target_agents == []
        assert t.relevant_categories == []
        assert t.tool_name == ""
        assert t.tool_description == ""
        assert t.tool_params == {}
        assert t.tool_response == ""
        assert t.test_query == ""

    def test_full_construction(self) -> None:
        """Construct with all fields including tool definition."""
        t = PayloadTemplate(
            name="exfil",
            technique=InjectionTechnique.DESCRIPTION_POISONING,
            description="Exfil test",
            owasp_ids=["MCP03", "MCP06"],
            target_agents=["claude"],
            tool_name="get_weather",
            tool_description="Get weather <IMPORTANT>exfil</IMPORTANT>",
            tool_params={"city": {"type": "string", "description": "City name"}},
            tool_response="Weather for {city}: 72°F",
            test_query="What's the weather in London?",
        )
        assert t.tool_name == "get_weather"
        assert t.tool_description.startswith("Get weather")
        assert "city" in t.tool_params
        assert "{city}" in t.tool_response
        assert t.test_query == "What's the weather in London?"

    def test_test_query_default_empty(self) -> None:
        """test_query defaults to empty string."""
        t = PayloadTemplate(
            name="test",
            technique=InjectionTechnique.OUTPUT_INJECTION,
            description="Test",
        )
        assert t.test_query == ""

    def test_relevant_categories_default_empty(self) -> None:
        """relevant_categories defaults to empty list."""
        t = PayloadTemplate(
            name="test",
            technique=InjectionTechnique.DESCRIPTION_POISONING,
            description="test",
        )
        assert t.relevant_categories == []

    def test_relevant_categories_set(self) -> None:
        """relevant_categories can be set explicitly."""
        t = PayloadTemplate(
            name="test",
            technique=InjectionTechnique.DESCRIPTION_POISONING,
            description="test",
            relevant_categories=["tool_poisoning", "prompt_injection"],
        )
        assert t.relevant_categories == ["tool_poisoning", "prompt_injection"]
