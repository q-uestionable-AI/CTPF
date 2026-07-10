"""``qai ipi generate`` — create document(s) with hidden payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from q_ai.core.cli.prompt import build_teaching_tip, is_tty, prompt_or_fail
from q_ai.ipi.callback_state import read_valid_state
from q_ai.ipi.commands._shared import (
    PHASE1_TECHNIQUES,
    PHASE2_TECHNIQUES,
    _parse_citation_frame,
    app,
    console,
    validate_format,
)
from q_ai.ipi.generate_service import GenerateResult, generate_documents
from q_ai.ipi.generators import ENCODING_CHOICES, get_techniques_for_format
from q_ai.ipi.models import (
    CitationFrame,
    DocumentTemplate,
    Format,
    PayloadStyle,
    PayloadType,
    Technique,
)
from q_ai.ipi.sweep_selection import (
    NoFindings,
    SelectedTemplate,
    StaleRefusal,
    TieRefusal,
    select_template_for_target,
)

_COMMANDLINE_SOURCE_NAME = "COMMANDLINE"


def parse_techniques(technique_str: str) -> list[Technique]:
    """Parse technique specification string into list of Technique enums.

    Supports preset names (all, phase1, phase2), individual technique names,
    or comma-separated lists of technique names.

    Args:
        technique_str: Technique specification. Options:
            - "all": All techniques from both phases
            - "phase1": WHITE_INK, OFF_CANVAS, METADATA
            - "phase2": TINY_TEXT, WHITE_RECT, FORM_FIELD, ANNOTATION,
                       JAVASCRIPT, EMBEDDED_FILE, INCREMENTAL
            - Single technique name (e.g., "white_ink")
            - Comma-separated names (e.g., "white_ink,metadata")

    Returns:
        List of Technique enum values.

    Raises:
        ValueError: If any technique name is invalid.
    """
    technique_str = technique_str.lower().strip()

    if technique_str == "all":
        # Technique.NONE is a control condition (visible payload), not a
        # hiding technique. Exclude it from batch presets; callers who want
        # the control must request it explicitly with --technique none.
        return [t for t in Technique if t is not Technique.NONE]
    if technique_str == "phase1":
        return [Technique(t) for t in PHASE1_TECHNIQUES]
    if technique_str == "phase2":
        return [Technique(t) for t in PHASE2_TECHNIQUES]

    technique_names = [t.strip() for t in technique_str.split(",")]
    techniques = []

    for name in technique_names:
        try:
            techniques.append(Technique(name))
        except ValueError:
            raise ValueError(f"Invalid technique: {name}") from None

    return techniques


def _parse_payload_style(payload_style: str) -> PayloadStyle:
    """Parse and validate a payload style string.

    Args:
        payload_style: Raw payload style name from CLI input.

    Returns:
        Validated PayloadStyle enum value.

    Raises:
        typer.Exit: If the payload style is invalid.
    """
    try:
        return PayloadStyle(payload_style)
    except ValueError:
        console.print(f"[red]X Invalid payload style: {payload_style}[/red]")
        console.print(f"  Valid options: {', '.join(p.value for p in PayloadStyle)}")
        raise typer.Exit(1) from None


def _parse_payload_type(payload_type: str) -> PayloadType:
    """Parse and validate a payload type string.

    Args:
        payload_type: Raw payload type name from CLI input.

    Returns:
        Validated PayloadType enum value.

    Raises:
        typer.Exit: If the payload type is invalid.
    """
    try:
        return PayloadType(payload_type)
    except ValueError:
        console.print(f"[red]X Invalid payload type: {payload_type}[/red]")
        console.print(f"  Valid options: {', '.join(p.value for p in PayloadType)}")
        raise typer.Exit(1) from None


def _parse_encoding(encoding: str) -> str:
    """Validate a payload URL encoding choice.

    Args:
        encoding: Raw encoding name from CLI input.

    Returns:
        The validated encoding string (one of ``ENCODING_CHOICES``).

    Raises:
        typer.Exit: If the encoding is not recognized.
    """
    if encoding not in ENCODING_CHOICES:
        console.print(f"[red]X Invalid encoding: {encoding}[/red]")
        console.print(f"  Valid options: {', '.join(ENCODING_CHOICES)}")
        raise typer.Exit(1)
    return encoding


def _enforce_dangerous_gate(payload_type_enum: PayloadType, dangerous: bool) -> None:
    """Enforce the --dangerous safety gate for non-callback payload types.

    Args:
        payload_type_enum: The resolved payload type.
        dangerous: Whether the --dangerous flag was provided.

    Raises:
        typer.Exit: If a dangerous payload type is used without --dangerous.
    """
    if payload_type_enum == PayloadType.CALLBACK:
        return

    if not dangerous:
        console.print(
            f"[red]X Payload type '{payload_type_enum.value}' requires --dangerous flag[/red]"
        )
        console.print("  Non-callback payloads can cause real harm to target systems.")
        console.print("  Use [bold]--dangerous[/bold] to confirm authorized testing.")
        raise typer.Exit(1)

    console.print()
    console.print("[bold red]" + "=" * 60 + "[/bold red]")
    console.print("[bold red]  WARNING: DANGEROUS PAYLOAD TYPE ENABLED[/bold red]")
    console.print(f"[bold red]  Type: {payload_type_enum.value}[/bold red]")
    console.print("[bold red]  For authorized security testing only.[/bold red]")
    console.print("[bold red]" + "=" * 60 + "[/bold red]")
    console.print()


def _resolve_techniques(technique: str, format_name: Format) -> list[Technique]:
    """Parse technique string and filter to those valid for the given format.

    Args:
        technique: Technique specification string (preset or comma-separated names).
        format_name: Target format for filtering available techniques.

    Returns:
        List of validated Technique enums applicable to the format.

    Raises:
        typer.Exit: If parsing fails or no valid techniques remain.
    """
    try:
        techniques = parse_techniques(technique)
    except ValueError as e:
        console.print(f"[red]X {e}[/red]")
        console.print("  Valid presets: all, phase1, phase2")
        console.print(f"  Valid techniques: {', '.join(t.value for t in Technique)}")
        raise typer.Exit(1) from None

    format_techniques = get_techniques_for_format(format_name, include_none=True)
    valid_techniques = [t for t in techniques if t in format_techniques]

    if not valid_techniques:
        console.print(f"[red]X No valid techniques for format '{format_name.value}'[/red]")
        console.print(f"  Available techniques: {', '.join(t.value for t in format_techniques)}")
        raise typer.Exit(1)

    if len(valid_techniques) < len(techniques):
        skipped = [t for t in techniques if t not in format_techniques]
        skipped_names = ", ".join(t.value for t in skipped)
        console.print(
            f"[yellow]! Skipping techniques not available"
            f" for {format_name.value}: {skipped_names}[/yellow]"
        )

    return valid_techniques


def _resolve_template_for_target(
    ctx: typer.Context,
    target: str | None,
    template: DocumentTemplate,
) -> DocumentTemplate:
    """Resolve the generate ``--template`` when ``--target`` is supplied.

    Auto-selects from sweep findings only when ``target`` is set AND
    ``--template`` was not explicitly passed on the command line. Explicit
    ``--template`` always wins, regardless of ``target`` or findings
    state. This invariant is enforced at the CLI entry point (not at any
    caller site) so future entry points that share this command's
    arguments inherit the same contract.

    Args:
        ctx: Typer/Click context used to distinguish an explicit
            ``--template`` from its default value.
        target: Target ID passed via ``--target``, or None.
        template: The template value already resolved from CLI parsing
            (default :attr:`DocumentTemplate.GENERIC` when not supplied).

    Returns:
        The template to use for generation.

    Raises:
        typer.Exit: On any auto-select refusal (tie, stale-refuse,
            no-findings). Exits with code 1 after printing a specific
            error message.
    """
    if target is None:
        return template

    source = ctx.get_parameter_source("template")
    if source is not None and source.name == _COMMANDLINE_SOURCE_NAME:
        return template

    result = select_template_for_target(target)

    if isinstance(result, SelectedTemplate):
        _emit_auto_select_prefix(result)
        return result.template

    if isinstance(result, TieRefusal):
        _print_tie_error(target, result)
    elif isinstance(result, StaleRefusal):
        _print_stale_refuse_error(target, result)
    elif isinstance(result, NoFindings):
        _print_no_findings_error(target)
    raise typer.Exit(1)


def _emit_auto_select_prefix(selection: SelectedTemplate) -> None:
    """Print the one-line auto-select prefix to stdout.

    Args:
        selection: Successful selection result.
    """
    day_word = "day" if selection.age_days == 1 else "days"
    percent = round(selection.compliance_rate * 100)
    iso = selection.completed_at.isoformat()
    line = (
        f"Auto-selected template: {selection.template.value}"
        f" (sweep run {iso}, {selection.age_days} {day_word} ago,"
        f" {percent}% compliance)"
    )
    if selection.stale_warn:
        line += " — consider re-running sweep"
    console.print(line)


def _print_no_findings_error(target: str) -> None:
    """Print the no-findings refusal to the console.

    Args:
        target: Target ID that yielded no sweep runs.
    """
    console.print(f"[red]X No sweep findings for target {target!r}.[/red]")
    console.print(f"  Run `qai ipi sweep --target {target}` first, or pass --template explicitly.")


def _print_tie_error(target: str, refusal: TieRefusal) -> None:
    """Print the tie refusal listing all candidates inside the 10pp band.

    Args:
        target: Target ID associated with the run.
        refusal: The :class:`TieRefusal` describing the near-tie.
    """
    console.print(
        f"[red]X Sweep findings for target {target!r} are tied within"
        " 10pp — cannot auto-select a template:[/red]"
    )
    for template, rate in refusal.candidates:
        percent = round(rate * 100)
        console.print(f"  - {template.value}: {percent}% compliance")
    console.print("  Pass --template explicitly to choose one.")


def _print_stale_refuse_error(target: str, refusal: StaleRefusal) -> None:
    """Print the stale-refuse error with run timestamp and age.

    Args:
        target: Target ID associated with the run.
        refusal: The :class:`StaleRefusal` with timestamp and age.
    """
    iso = refusal.completed_at.isoformat()
    console.print(
        f"[red]X Most recent sweep for target {target!r} completed"
        f" {iso} ({refusal.age_days} days ago) — exceeds the 30-day"
        " freshness limit.[/red]"
    )
    console.print("  Run a fresh sweep, or pass --template explicitly.")


def _display_generate_results(
    result: GenerateResult,
    format_name: Format,
    style: PayloadStyle,
    payload_type_enum: PayloadType,
    callback_url: str,
) -> None:
    """Display generation results to the console.

    Args:
        result: GenerateResult with campaigns and errors lists.
        format_name: Format used for generation.
        style: Payload style used.
        payload_type_enum: Payload type used.
        callback_url: Callback URL for display.
    """
    if len(result.campaigns) > 1:
        console.print(
            f"\n[bold green]OK Generated {len(result.campaigns)} "
            f"{format_name.value.upper()} files "
            f"({style.value} payload, {payload_type_enum.value} type):[/bold green]"
        )
        for c in result.campaigns:
            console.print(f"  - {c.filename} ({c.technique}) -> UUID: [cyan]{c.uuid}[/cyan]")
    elif result.campaigns:
        c = result.campaigns[0]
        console.print(f"\n[bold green]OK Generated:[/bold green] {c.filename}")
        console.print(f"  Format: {format_name.value}")
        console.print(f"  Technique: {c.technique}")
        console.print(f"  Payload Style: {style.value}")
        console.print(f"  Payload Type: {payload_type_enum.value}")
        console.print(f"  UUID: [cyan]{c.uuid}[/cyan]")

    for err in result.errors:
        console.print(f"  [yellow]! {err}[/yellow]")

    console.print(f"\n[dim]Callback URL: {callback_url}/c/<uuid>[/dim]")


@app.command(
    epilog=(
        "Examples:\n"
        "  qai ipi generate http://localhost:8080\n"
        "  qai ipi generate http://localhost:8080 --format image --technique all\n"
        "  qai ipi generate --callback http://localhost:8080 --technique phase1\n"
        "  qai ipi generate  (interactive — prompts for callback URL)"
    ),
)
def generate(  # noqa: PLR0913 — CLI entry point, one parameter per Typer option
    ctx: typer.Context,
    callback: Annotated[
        str | None,
        typer.Argument(
            help="Callback server URL (prompted interactively if omitted).",
        ),
    ] = None,
    callback_option: Annotated[
        str | None,
        typer.Option("--callback", "-c", help="Callback server URL (alternative to positional)."),
    ] = None,
    output: Annotated[
        Path,
        typer.Option(
            "--output",
            "-o",
            help="Output path (file or directory, default: ~/.qai/payloads/)",
        ),
    ] = Path.home() / ".qai" / "payloads",  # noqa: B008 — Typer default is evaluated once
    format_name: Annotated[
        str,
        typer.Option(
            "--format",
            help="Output format: pdf, image, markdown, html, docx, ics, eml (default: pdf)",
        ),
    ] = "pdf",
    technique: Annotated[
        str,
        typer.Option(
            "--technique",
            "-t",
            help=(
                "Technique(s): all, phase1, phase2, none (visible-payload control), "
                "or specific names (comma-separated)"
            ),
        ),
    ] = "all",
    payload_type: Annotated[
        str,
        typer.Option(
            "--payload-type",
            help=(
                "Payload type: callback, exfil_summary, exfil_context, ssrf_internal, "
                "instruction_override, tool_abuse, persistence"
            ),
        ),
    ] = "callback",
    payload_style: Annotated[
        str,
        typer.Option(
            "--payload",
            "--payload-style",
            "-p",
            help="Payload style: obvious, citation, reviewer, "
            "helpful, academic, compliance, datasource",
        ),
    ] = "obvious",
    name: Annotated[str, typer.Option("--name", "-n", help="Base filename")] = "report",
    dangerous: Annotated[
        bool,
        typer.Option(
            "--dangerous",
            help="Enable non-callback payload types (exfil, ssrf, override, etc.)",
        ),
    ] = False,
    seed: Annotated[
        int | None,
        typer.Option(
            "--seed",
            help="Seed for deterministic UUID/token generation (reproducible corpus).",
        ),
    ] = None,
    encoding: Annotated[
        str,
        typer.Option(
            "--encoding",
            help="Encode payload text (none=plaintext, base16/hex=obfuscated)",
        ),
    ] = "none",
    template: Annotated[
        DocumentTemplate,
        typer.Option(
            "--template",
            help="Document context template for payload framing (default: generic).",
            case_sensitive=False,
        ),
    ] = DocumentTemplate.GENERIC,
    citation_frame_value: Annotated[
        str,
        typer.Option(
            "--citation-frame",
            help=(
                "Citation-style callback rendering: 'plain' uses pre-4.5 hardcoded"
                " text (report-context baseline); 'template-aware' (default) uses"
                " per-template callback rationale. No-op unless --payload-style is"
                " 'citation' AND --payload-type is 'callback'."
            ),
        ),
    ] = "template-aware",
    target: Annotated[
        str | None,
        typer.Option(
            "--target",
            help=(
                "Target ID. When supplied without an explicit --template,"
                " auto-selects the best template from the target's most"
                " recent sweep findings."
            ),
        ),
    ] = None,
) -> None:
    """Generate document(s) with hidden prompt injection payload.

    CALLBACK can be provided as the first positional argument, via
    --callback, or entered interactively when running in a terminal.

    Creates one or more documents containing hidden prompt injection
    payloads using the specified technique(s). Each generated document
    is registered in the database for callback tracking.

    For the full technique list across all formats, run
    'qai ipi techniques'. For supported output formats, run
    'qai ipi formats'.
    """
    # Resolve callback:
    #   positional > --callback > active-callback state file > interactive prompt
    provided_directly = callback is not None or callback_option is not None
    raw_callback = callback or callback_option or None

    if raw_callback is None:
        state, warning = read_valid_state()
        if warning is not None:
            console.print(f"[yellow]! {warning}[/yellow]")
        if state is not None:
            console.print(
                f"[dim]Using active callback: {state.public_url} ({state.provider} tunnel)[/dim]"
            )
            raw_callback = state.public_url

    callback_url = prompt_or_fail("CALLBACK", raw_callback, "Callback server URL")

    if not provided_directly and is_tty():
        tip = build_teaching_tip("qai ipi generate", [callback_url])
        console.print(f"[dim]{tip}[/dim]")

    format_name = validate_format(format_name)
    style = _parse_payload_style(payload_style)
    payload_type_enum = _parse_payload_type(payload_type)
    encoding = _parse_encoding(encoding)
    _enforce_dangerous_gate(payload_type_enum, dangerous)
    techniques = _resolve_techniques(technique, format_name)

    template = _resolve_template_for_target(ctx, target, template)
    citation_frame_enum: CitationFrame = _parse_citation_frame(citation_frame_value)

    result = generate_documents(
        callback_url=callback_url,
        output=output,
        format_name=format_name,
        techniques=techniques,
        payload_style=style,
        payload_type=payload_type_enum,
        base_name=name,
        seed=seed,
        encoding=encoding,
        template=template,
        citation_frame=citation_frame_enum,
    )

    from q_ai.ipi.mapper import persist_generate

    if result.campaigns:
        persist_generate(result.campaigns, source="cli")

    _display_generate_results(result, format_name, style, payload_type_enum, callback_url)
