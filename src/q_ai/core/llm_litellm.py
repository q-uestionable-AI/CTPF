"""LiteLLM implementation of ProviderClient.

This is the only file that imports litellm. If litellm ever needs
replacing, only this file changes.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from litellm import acompletion, get_model_info  # type: ignore[import-untyped]

from q_ai.core.llm import (
    NormalizedResponse,
    ProviderError,
    ToolCall,
    ToolSpec,
    UnsupportedCapabilityError,
    parse_model_string,
)

logger = logging.getLogger(__name__)

_DEFAULT_PROVIDER_TIMEOUT_S = 60.0
_ASSISTANT_TIMEOUT_S = 120.0

# litellm error messages indicating no tool support
_NO_TOOL_SUPPORT_SIGNALS = [
    "does not support tools",
    "does not support function",
    "tool_choice is not supported",
    "tools is not supported",
]


def _tool_spec_to_openai_format(tool: ToolSpec) -> dict[str, Any]:
    """Convert a ToolSpec to OpenAI-compatible tool format.

    Args:
        tool: Provider-agnostic tool specification.

    Returns:
        Dict in OpenAI tool format for litellm.
    """
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


def _build_completion_kwargs(
    model: str,
    messages: list[dict[str, Any]],
    timeout: float,
    api_base: str | None,
    api_key: str | None,
    stream: bool = False,
) -> dict[str, Any]:
    """Build shared keyword arguments for LiteLLM completion calls."""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "timeout": timeout,
    }
    if stream:
        kwargs["stream"] = True
    if api_base:
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key
    return kwargs


def _get_first_choice(response: Any, model: str, response_kind: str) -> Any | None:
    """Return the first choice from a LiteLLM payload when present."""
    choices = getattr(response, "choices", None)
    if not choices:
        logger.debug("LiteLLM %s payload missing choices for %s", response_kind, model)
        return None

    try:
        return choices[0]
    except (IndexError, KeyError, TypeError):
        logger.debug("LiteLLM %s payload had unusable choices for %s", response_kind, model)
        return None


async def complete_text(
    model: str,
    messages: list[dict[str, Any]],
    *,
    api_base: str | None = None,
    api_key: str | None = None,
    timeout: float = _ASSISTANT_TIMEOUT_S,
) -> str:
    """Complete a plain-text chat response through the LiteLLM boundary.

    Args:
        model: LiteLLM model string.
        messages: Chat messages to send.
        api_base: Optional base URL override.
        api_key: Optional provider credential.
        timeout: Request timeout in seconds.

    Returns:
        Response text content, or an empty string when absent.
    """
    response = await acompletion(
        **_build_completion_kwargs(model, messages, timeout, api_base, api_key)
    )
    choice = _get_first_choice(response, model, "completion")
    if choice is None:
        return ""

    message = getattr(choice, "message", None)
    if message is None:
        logger.debug("LiteLLM completion payload missing message for %s", model)
        return ""

    content = getattr(message, "content", None)
    if content is None:
        logger.debug("LiteLLM completion payload missing content for %s", model)
        return ""
    if isinstance(content, str):
        return content

    logger.debug("LiteLLM completion payload had non-string content for %s", model)
    return ""


async def stream_text(
    model: str,
    messages: list[dict[str, Any]],
    *,
    api_base: str | None = None,
    api_key: str | None = None,
    timeout: float = _ASSISTANT_TIMEOUT_S,
) -> AsyncIterator[str]:
    """Stream a plain-text chat response through the LiteLLM boundary.

    Args:
        model: LiteLLM model string.
        messages: Chat messages to send.
        api_base: Optional base URL override.
        api_key: Optional provider credential.
        timeout: Request timeout in seconds.

    Yields:
        Non-empty response text chunks as they arrive.
    """
    response: Any = await acompletion(
        **_build_completion_kwargs(model, messages, timeout, api_base, api_key, stream=True)
    )
    async for chunk in response:
        choice = _get_first_choice(chunk, model, "stream chunk")
        if choice is None:
            continue

        delta = getattr(choice, "delta", None)
        if delta is None:
            logger.debug("LiteLLM stream chunk missing delta for %s", model)
            continue

        content = getattr(delta, "content", None)
        if isinstance(content, str) and content:
            yield content
        elif content is not None:
            logger.debug("LiteLLM stream chunk had non-string content for %s", model)


def get_litellm_context_window(model: str) -> int | None:
    """Return the model context window reported by LiteLLM.

    Args:
        model: LiteLLM model string.

    Returns:
        Context window size in tokens, or ``None`` when unavailable.
    """
    info = get_model_info(model=model)
    if not isinstance(info, dict):
        return None

    max_input = info.get("max_input_tokens")
    if isinstance(max_input, int) and max_input > 0:
        return max_input

    max_tokens = info.get("max_tokens")
    if isinstance(max_tokens, int) and max_tokens > 0:
        return max_tokens

    return None


class LiteLLMClient:
    """ProviderClient implementation using litellm.

    Handles all providers via litellm's model-string routing.
    Normalizes responses to NormalizedResponse.
    """

    def __init__(
        self,
        *,
        api_base: str | None = None,
        api_key: str | None = None,
        generation_parameters: dict[str, Any] | None = None,
    ) -> None:
        """Configure one provider client without exposing credentials globally.

        Args:
            api_base: OpenAI-compatible API base URL.
            api_key: Credential supplied directly to LiteLLM.
            generation_parameters: Fixed supported generation parameters.
        """
        self._api_base = api_base
        self._api_key = api_key
        self._generation_parameters = dict(generation_parameters or {})

    async def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec],
        max_tokens: int = 1024,
    ) -> NormalizedResponse:
        """Call litellm.acompletion and normalize the response.

        Args:
            model: Model string (provider/model-id or bare).
            messages: Chat messages.
            tools: Tool specifications.
            max_tokens: Maximum tokens in response.

        Returns:
            Normalized response from the provider.

        Raises:
            UnsupportedCapabilityError: If the model doesn't support tool calling.
            ProviderError: For other provider errors.
        """
        provider, _model_id = parse_model_string(model)

        openai_tools = [_tool_spec_to_openai_format(t) for t in tools] if tools else None

        request: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": openai_tools,
            "max_tokens": max_tokens,
            "timeout": _DEFAULT_PROVIDER_TIMEOUT_S,
        }
        if self._api_base:
            request["api_base"] = self._api_base
        if self._api_key:
            request["api_key"] = self._api_key
        request.update(self._generation_parameters)

        try:
            # Security: Add timeout to prevent external provider calls from hanging indefinitely
            response = await acompletion(**request)  # type: ignore[no-untyped-call]
        except Exception as exc:
            error_msg = str(exc).lower()
            if any(signal in error_msg for signal in _NO_TOOL_SUPPORT_SIGNALS):
                raise UnsupportedCapabilityError(
                    f"Model {model} does not support tool calling: {exc}"
                ) from exc
            raise ProviderError(f"Provider {provider} error: {exc}") from exc

        # Extract response data
        choice = response.choices[0]  # type: ignore[union-attr]
        message = choice.message  # type: ignore[union-attr]

        # Normalize tool calls
        normalized_tool_calls: list[ToolCall] = []
        if message.tool_calls:  # type: ignore[union-attr]
            for tc in message.tool_calls:  # type: ignore[union-attr]
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError as json_exc:
                        raise ProviderError(
                            f"Malformed tool call arguments from {model}: {json_exc}"
                        ) from json_exc
                call_id = getattr(tc, "id", "")
                normalized_tool_calls.append(
                    ToolCall(
                        name=tc.function.name,
                        arguments=args,
                        id=call_id if isinstance(call_id, str) else "",
                    )
                )

        # Normalize content
        content = message.content or ""  # type: ignore[union-attr]

        # Build raw_response for evidence
        raw: Any = (
            response.model_dump()  # type: ignore[union-attr]
            if hasattr(response, "model_dump")
            else response
        )

        return NormalizedResponse(
            tool_calls=normalized_tool_calls,
            content=content,
            finish_reason=choice.finish_reason,  # type: ignore[union-attr]
            raw_response=raw,
            model=model,
        )
