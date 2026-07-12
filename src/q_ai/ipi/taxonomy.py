"""IPI probe taxonomy bridge — maps probe categories to qai audit categories.

This module is intentionally import-free of the removed ``q_ai.imports``
package. It carries only the IPI-probe bridges needed by
:mod:`q_ai.ipi.probe_service`.
"""

from __future__ import annotations

from dataclasses import dataclass

_IPI_PROBE_FRAMEWORK = "ipi_probe"

# Mapping: probe category -> (qai_category | None, confidence)
# All probe categories relate to prompt injection; direct for techniques
# that explicitly hijack instructions, adjacent for indirect methods.
_IPI_PROBE_BRIDGES: dict[str, tuple[str | None, str]] = {
    "instruction_override": ("prompt_injection", "direct"),
    "delimiter_confusion": ("prompt_injection", "direct"),
    "context_manipulation": ("prompt_injection", "adjacent"),
    "authority_spoofing": ("prompt_injection", "adjacent"),
    "task_hijacking": ("prompt_injection", "direct"),
    "exfil_framing": ("prompt_injection", "adjacent"),
    "encoding": ("prompt_injection", "adjacent"),
    "subtle_injection": ("prompt_injection", "adjacent"),
}


@dataclass
class TaxonomyBridge:
    """Mapping between an external taxonomy ID and a qai category.

    Attributes:
        external_framework: Framework name, e.g. ``"ipi_probe"``.
        external_id: Identifier within the framework, e.g. a probe category.
        qai_category: Mapped qai category, or ``None`` if no equivalent.
        confidence: ``"direct"``, ``"adjacent"``, or ``"none"``.
    """

    external_framework: str
    external_id: str
    qai_category: str | None
    confidence: str  # "direct" | "adjacent" | "none"


def resolve_bridge(external_framework: str, external_id: str) -> TaxonomyBridge:
    """Look up a taxonomy bridge for an IPI probe framework ID.

    Only the ``ipi_probe`` framework is registered here. Unknown frameworks
    or IDs return a bridge with ``confidence="none"`` and
    ``qai_category=None``.

    Args:
        external_framework: Framework name, e.g. ``"ipi_probe"``.
        external_id: Identifier within the framework (probe category).

    Returns:
        A :class:`TaxonomyBridge` with mapping details.
    """
    if external_framework != _IPI_PROBE_FRAMEWORK:
        return TaxonomyBridge(
            external_framework=external_framework,
            external_id=external_id,
            qai_category=None,
            confidence="none",
        )
    qai_category, confidence = _IPI_PROBE_BRIDGES.get(external_id, (None, "none"))
    return TaxonomyBridge(
        external_framework=external_framework,
        external_id=external_id,
        qai_category=qai_category,
        confidence=confidence,
    )
