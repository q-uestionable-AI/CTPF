"""Tests for the IPI document context template registry."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import click
import pytest
from typer.testing import CliRunner

from q_ai.ipi.cli import app
from q_ai.ipi.generate_service import GenerateResult, generate_documents
from q_ai.ipi.models import DocumentTemplate, Format, PayloadStyle, TemplateSpec
from q_ai.ipi.template_registry import (
    BIPIA_COMMIT,
    GARAK_COMMIT,
    TEMPLATE_REGISTRY,
    get_template_spec,
    get_templates_for_format,
)

runner = CliRunner()

GARAK_TEMPLATES: frozenset[DocumentTemplate] = frozenset(
    {
        DocumentTemplate.WHOIS,
        DocumentTemplate.TRANSLATION_EN_FR,
        DocumentTemplate.TRANSLATION_EN_ZH,
        DocumentTemplate.LEGAL_SNIPPET,
        DocumentTemplate.REPORT,
        DocumentTemplate.RESUME,
    }
)

BIPIA_TEMPLATES: frozenset[DocumentTemplate] = frozenset(
    {
        DocumentTemplate.EMAIL,
        DocumentTemplate.WEB,
        DocumentTemplate.TABLE,
        DocumentTemplate.CODE,
        DocumentTemplate.NEWS,
    }
)


class TestRegistryShape:
    """Coverage of every enum member and basic spec shape."""

    def test_registry_contains_every_enum_member(self) -> None:
        assert set(TEMPLATE_REGISTRY) == set(DocumentTemplate)

    def test_registry_has_twelve_members(self) -> None:
        assert len(TEMPLATE_REGISTRY) == 12

    def test_every_spec_marker_present(self) -> None:
        """Non-GENERIC stub templates must contain the {payload} marker."""
        for tmpl, spec in TEMPLATE_REGISTRY.items():
            if tmpl == DocumentTemplate.GENERIC:
                assert spec.context_template == ""
                assert spec.top_instruction == ""
            else:
                assert "{payload}" in spec.context_template

    def test_generic_compatible_with_all_formats(self) -> None:
        spec = TEMPLATE_REGISTRY[DocumentTemplate.GENERIC]
        assert set(spec.formats) == set(Format)

    def test_garak_specs_use_pinned_commit(self) -> None:
        for tmpl in GARAK_TEMPLATES:
            spec = TEMPLATE_REGISTRY[tmpl]
            assert spec.source_tool == "garak"
            assert spec.source_commit == GARAK_COMMIT

    def test_bipia_specs_use_pinned_commit(self) -> None:
        for tmpl in BIPIA_TEMPLATES:
            spec = TEMPLATE_REGISTRY[tmpl]
            assert spec.source_tool == "bipia"
            assert spec.source_commit == BIPIA_COMMIT


class TestGarakTemplateContent:
    """Phase 4.3a content checks for Garak-aligned templates."""

    def test_no_stub_prefix_remains(self) -> None:
        for tmpl in GARAK_TEMPLATES:
            spec = TEMPLATE_REGISTRY[tmpl]
            assert "[STUB" not in spec.top_instruction, tmpl.name
            assert "[STUB" not in spec.context_template, tmpl.name

    def test_context_template_has_single_payload_marker(self) -> None:
        for tmpl in GARAK_TEMPLATES:
            spec = TEMPLATE_REGISTRY[tmpl]
            assert spec.context_template.count("{payload}") == 1, tmpl.name

    def test_context_template_is_substantial(self) -> None:
        for tmpl in GARAK_TEMPLATES:
            spec = TEMPLATE_REGISTRY[tmpl]
            assert len(spec.context_template) >= 200, tmpl.name

    def test_whois_contains_standard_fields(self) -> None:
        spec = TEMPLATE_REGISTRY[DocumentTemplate.WHOIS]
        assert "Domain Name:" in spec.context_template
        assert "Registrar:" in spec.context_template


class TestBipiaTemplateContent:
    """Phase 4.3b content checks for BIPIA-aligned templates."""

    def test_no_stub_prefix_remains(self) -> None:
        for tmpl in BIPIA_TEMPLATES:
            spec = TEMPLATE_REGISTRY[tmpl]
            assert "[STUB" not in spec.top_instruction, tmpl.name
            assert "[STUB" not in spec.context_template, tmpl.name

    def test_context_template_has_single_payload_marker(self) -> None:
        for tmpl in BIPIA_TEMPLATES:
            spec = TEMPLATE_REGISTRY[tmpl]
            assert spec.context_template.count("{payload}") == 1, tmpl.name

    def test_context_template_is_substantial(self) -> None:
        for tmpl in BIPIA_TEMPLATES:
            spec = TEMPLATE_REGISTRY[tmpl]
            assert len(spec.context_template) >= 200, tmpl.name

    def test_email_contains_header_fields(self) -> None:
        spec = TEMPLATE_REGISTRY[DocumentTemplate.EMAIL]
        assert "SUBJECT:" in spec.context_template
        assert "EMAIL_FROM:" in spec.context_template
        assert "RECEIVED DATE:" in spec.context_template
        assert "CONTENT:" in spec.context_template

    def test_email_top_instruction_includes_user_question(self) -> None:
        """EMAIL top_instruction must supply a concrete user question.

        The BIPIA-derived QA framing promises the model a user question
        ("Answer the user's question based only on the content of the
        email below") but originally never supplied one. The Phase 4.4a
        template effectiveness sweep against Nemotron 3 Nano surfaced the
        EMAIL template at 20% compliance (1/5) while the other ten
        templates ran at 100% — reasoning traces showed the model
        fixating on the missing question instead of acting on the
        injected payload. A generic user question is appended inside
        ``top_instruction`` to complete the QA framing; this assertion
        pins its presence to prevent silent regression.
        """
        spec = TEMPLATE_REGISTRY[DocumentTemplate.EMAIL]
        assert "User's question:" in spec.top_instruction

    def test_table_contains_markdown_syntax(self) -> None:
        spec = TEMPLATE_REGISTRY[DocumentTemplate.TABLE]
        assert "|" in spec.context_template
        assert "---" in spec.context_template

    def test_code_contains_error_pattern(self) -> None:
        spec = TEMPLATE_REGISTRY[DocumentTemplate.CODE]
        assert "Traceback" in spec.context_template
        assert "Code snippet:" in spec.context_template


class TestCallbackRole:
    """Phase 4.5 ``callback_role`` field coverage.

    Pins the values the payload builder depends on:

    - GENERIC is exactly ``"the supplementary data appendix"`` (the
      Phase 4.5 default sourced from the original CITATION wording).
    - Every non-GENERIC template carries a non-empty noun phrase under
      60 characters (style/tone constraint from the task brief).
    - ``TemplateSpec`` accepts the field without breaking construction.
    """

    def test_generic_role_is_supplementary_data_appendix(self) -> None:
        spec = TEMPLATE_REGISTRY[DocumentTemplate.GENERIC]
        assert spec.callback_role == "the supplementary data appendix"

    def test_every_non_generic_template_has_non_empty_role(self) -> None:
        for tmpl, spec in TEMPLATE_REGISTRY.items():
            if tmpl == DocumentTemplate.GENERIC:
                continue
            assert spec.callback_role, f"{tmpl.name} is missing a callback_role"

    def test_every_callback_role_under_sixty_chars(self) -> None:
        for tmpl, spec in TEMPLATE_REGISTRY.items():
            assert len(spec.callback_role) < 60, (
                f"{tmpl.name} callback_role exceeds 60 chars: "
                f"{spec.callback_role!r} ({len(spec.callback_role)})"
            )

    def test_template_spec_accepts_callback_role_field(self) -> None:
        spec = TemplateSpec(
            id=DocumentTemplate.GENERIC,
            name="ad-hoc",
            description="",
            source_tool="generic",
            source_reference="",
            source_commit="",
            top_instruction="",
            context_template="",
            formats=(Format.PDF,),
            default_style=PayloadStyle.OBVIOUS,
            callback_role="the test fixture",
        )
        assert spec.callback_role == "the test fixture"

    def test_template_spec_callback_role_defaults_empty(self) -> None:
        spec = TemplateSpec(
            id=DocumentTemplate.GENERIC,
            name="ad-hoc",
            description="",
            source_tool="generic",
            source_reference="",
            source_commit="",
            top_instruction="",
            context_template="",
            formats=(Format.PDF,),
            default_style=PayloadStyle.OBVIOUS,
        )
        assert spec.callback_role == ""


class TestAccessors:
    """Coverage of the registry lookup helpers.

    All assertions are pure in-process registry reads — no network, no
    filesystem, no external services. Accessor behavior must stay stable
    because service-layer validation depends on it.
    """

    def test_get_template_spec_returns_spec(self) -> None:
        spec = get_template_spec(DocumentTemplate.WHOIS)
        assert isinstance(spec, TemplateSpec)
        assert spec.id == DocumentTemplate.WHOIS

    def test_get_templates_for_format_pdf_includes_whois(self) -> None:
        templates = get_templates_for_format(Format.PDF)
        assert DocumentTemplate.WHOIS in templates
        assert DocumentTemplate.GENERIC in templates

    def test_get_templates_for_format_eml_includes_email(self) -> None:
        templates = get_templates_for_format(Format.EML)
        assert DocumentTemplate.EMAIL in templates

    def test_get_templates_for_format_excludes_incompatible(self) -> None:
        """ICS only matches GENERIC; no document templates target ICS."""
        templates = get_templates_for_format(Format.ICS)
        assert DocumentTemplate.WHOIS not in templates
        assert DocumentTemplate.TABLE not in templates


class TestServiceValidation:
    """Compatibility validation surfaced by ``generate_documents``."""

    def test_incompatible_template_format_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not compatible"):
            generate_documents(
                callback_url="http://localhost:8080",
                output=tmp_path,
                format_name=Format.ICS,
                techniques=[],
                template=DocumentTemplate.TABLE,
            )

    def test_generic_does_not_raise_for_any_format(self, tmp_path: Path) -> None:
        """GENERIC must work for every format (no validation tripwire)."""
        from q_ai.ipi.models import Technique

        # Use a single PDF technique so we exercise the single-fn path.
        result = generate_documents(
            callback_url="http://localhost:8080",
            output=tmp_path,
            format_name=Format.PDF,
            techniques=[Technique.WHITE_INK],
            template=DocumentTemplate.GENERIC,
            seed=42,
        )
        assert result.campaigns or result.errors


class TestCLIIntegration:
    """CLI wiring for the ``--template`` flag.

    Uses Typer's ``CliRunner`` and direct Click command introspection.
    ``generate_documents`` is mocked (via ``unittest.mock.patch``) so no
    filesystem writes or network calls occur; the tests verify only that
    the CLI layer registers and forwards the ``--template`` option.
    """

    def test_generate_command_registers_template_param(self) -> None:
        """Verify --template is wired into the generate command.

        Inspect the Click parameter list directly instead of parsing help
        output — Typer/Rich rendering of --help varies across platforms
        (terminal width, ANSI styling), which made an earlier output-string
        assertion flake on macOS CI.
        """
        import typer

        click_app = typer.main.get_command(app)
        with click.Context(click_app) as ctx:
            generate = click_app.get_command(ctx, "generate")
        assert generate is not None
        param_names = [p.name for p in generate.params]
        assert "template" in param_names

    @patch("q_ai.ipi.commands.generate.generate_documents")
    def test_template_flag_threads_to_service(self, mock_gen: object) -> None:
        mock_gen.return_value = GenerateResult(campaigns=[], errors=[])  # type: ignore[attr-defined]
        result = runner.invoke(
            app,
            [
                "generate",
                "http://localhost:8080",
                "--template",
                "whois",
            ],
        )
        assert result.exit_code == 0, result.output
        kwargs = mock_gen.call_args.kwargs  # type: ignore[attr-defined]
        assert kwargs["template"] == DocumentTemplate.WHOIS


class TestDeferredGeneratorIntegration:
    """Phase 4.2 deferrals: EML + ICS accept template params without crashing.

    EML and image generators silently discard framing (documented in-code
    as Phase 4.3b work); ICS is unreachable via CLI because no template
    registers ICS as a compatible format. These tests pin that current
    behavior so Phase 4.3 work knows exactly what it is replacing.
    """

    def test_eml_generator_accepts_template_params_without_crash(self, tmp_path: Path) -> None:
        """EMAIL template + EML format must not crash (previously raised
        AttributeError because EmailMessage.get_content() returns None
        on a no-body message)."""
        from q_ai.ipi.generators.eml import create_eml
        from q_ai.ipi.models import Technique

        out = tmp_path / "msg.eml"
        campaign = create_eml(
            out,
            technique=Technique.EML_X_HEADER,
            callback_url="http://localhost:8080",
            top_instruction="stub top",
            context_template="stub ctx {payload}",
        )
        assert out.exists()
        assert campaign.format == Format.EML

    def test_ics_generator_accepts_template_params_without_crash(self, tmp_path: Path) -> None:
        """ICS framing path is currently unreachable via CLI (no template
        lists ICS), but direct library callers must not crash."""
        from q_ai.ipi.generators.ics import create_ics
        from q_ai.ipi.models import Technique

        out = tmp_path / "meeting.ics"
        campaign = create_ics(
            out,
            technique=Technique.ICS_DESCRIPTION,
            callback_url="http://localhost:8080",
            top_instruction="stub top",
            context_template="stub ctx {payload}",
        )
        assert out.exists()
        assert campaign.format == Format.ICS

    def test_no_template_lists_ics_format(self) -> None:
        """Guardrail: if Phase 4.3 adds an ICS-backed template, the
        framing block in create_ics must be restored (and _inject_description
        restructured). Failing this test is a reminder."""
        for tmpl, spec in TEMPLATE_REGISTRY.items():
            if tmpl == DocumentTemplate.GENERIC:
                continue
            assert Format.ICS not in spec.formats, (
                f"{tmpl.name} now targets ICS — restore template framing "
                f"integration in src/q_ai/ipi/generators/ics.py"
            )
