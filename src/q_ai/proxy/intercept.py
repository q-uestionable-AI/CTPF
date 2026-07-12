"""Intercept engine for mcp-proxy.

Manages hold/release state for the message pipeline. When in INTERCEPT
mode, messages are held for user inspection. The user can then FORWARD,
MODIFY, or DROP each held message.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass

from mcp.types import JSONRPCMessage

from q_ai.proxy.models import (
    HeldMessage,
    InterceptAction,
    InterceptMode,
    InterceptState,
    ProxyMessage,
)


def _validate_decision(
    action: InterceptAction,
    modified_raw: JSONRPCMessage | None,
) -> None:
    if action == InterceptAction.MODIFY and modified_raw is None:
        raise ValueError("MODIFY decisions require a modified JSON-RPC message")
    if action != InterceptAction.MODIFY and modified_raw is not None:
        raise ValueError("Only MODIFY decisions may include a modified JSON-RPC message")


@dataclass(frozen=True)
class InterceptDecision:
    """Validated programmatic decision for one intercepted message.

    Args:
        action: FORWARD, DROP, or MODIFY.
        modified_raw: Replacement JSON-RPC message for MODIFY decisions.
    """

    action: InterceptAction
    modified_raw: JSONRPCMessage | None = None

    def __post_init__(self) -> None:
        """Reject incomplete or internally inconsistent decisions."""
        _validate_decision(self.action, self.modified_raw)


InterceptRule = Callable[[ProxyMessage], InterceptDecision | None]


class InterceptEngine:
    """Controls whether messages are held for inspection or passed through.

    Args:
        mode: Initial intercept mode. Defaults to PASSTHROUGH.
        rule: Optional programmatic decision rule. Exceptions propagate.

    Example:
        >>> engine = InterceptEngine(mode=InterceptMode.INTERCEPT)
        >>> if engine.should_hold(proxy_msg):
        ...     held = engine.hold(proxy_msg)
        ...     await held.release.wait()
    """

    def __init__(
        self,
        mode: InterceptMode = InterceptMode.PASSTHROUGH,
        rule: InterceptRule | None = None,
    ) -> None:
        self._mode = mode
        self._rule = rule
        self._held: list[HeldMessage] = []

    @property
    def mode(self) -> InterceptMode:
        """Current intercept mode."""
        return self._mode

    def set_mode(self, mode: InterceptMode) -> None:
        """Change the intercept mode.

        Args:
            mode: The new mode. When switching to PASSTHROUGH, all
                currently held messages are released with FORWARD action.
        """
        self._mode = mode
        if mode == InterceptMode.PASSTHROUGH:
            for held in list(self._held):
                self.release(held, InterceptAction.FORWARD)

    def should_hold(self, message: ProxyMessage) -> bool:
        """Check whether a message should be held for inspection.

        Args:
            message: The proxy message to check.

        Returns:
            True if the engine is in INTERCEPT mode.
        """
        return self._mode == InterceptMode.INTERCEPT

    def decide(self, message: ProxyMessage) -> InterceptDecision | None:
        """Return the programmatic decision for a message, if configured.

        Rule exceptions intentionally propagate so malformed or uncertain
        mutations stop the pipeline instead of forwarding data.

        Args:
            message: The proxy message to evaluate.

        Returns:
            A validated decision, or None to use interactive mode behavior.
        """
        if self._rule is None:
            return None
        return self._rule(message)

    def hold(self, message: ProxyMessage) -> HeldMessage:
        """Hold a message for user inspection.

        Args:
            message: The proxy message to hold.

        Returns:
            A HeldMessage with an unset release Event.
        """
        held = HeldMessage(
            proxy_message=message,
            release=asyncio.Event(),
            action=None,
            modified_raw=None,
        )
        self._held.append(held)
        return held

    def release(
        self,
        held: HeldMessage,
        action: InterceptAction,
        modified_raw: JSONRPCMessage | None = None,
    ) -> None:
        """Release a held message with the specified action.

        Args:
            held: The held message to release.
            action: FORWARD, DROP, or MODIFY.
            modified_raw: If action is MODIFY, the edited JSON-RPC message.
        """
        _validate_decision(action, modified_raw)
        held.action = action
        held.modified_raw = modified_raw
        held.release.set()
        if held in self._held:
            self._held.remove(held)

    def get_held(self) -> list[HeldMessage]:
        """Return the list of currently held messages.

        Returns:
            List of HeldMessage objects awaiting user action.
        """
        return list(self._held)

    def drop_held(self) -> None:
        """Release all held messages with DROP so waiters unblock."""
        for held in list(self._held):
            self.release(held, InterceptAction.DROP)

    def get_state(self) -> InterceptState:
        """Return a snapshot of the current intercept state.

        Returns:
            InterceptState with current mode and held messages.
        """
        return InterceptState(mode=self._mode, held_messages=list(self._held))
