#!/usr/bin/env python3
"""Fix Qwen3.5 reasoning routing when enable_thinking is toggled.

Validated on:
  - Sehyo/Qwen3.5-35B-A3B-NVFP4

Problem:
  - With `--default-chat-template-kwargs {"enable_thinking":false}` plus
    request-level `chat_template_kwargs={"enable_thinking": true}`, Qwen3.5
    can reopen thinking, but older vLLM parser logic leaks reasoning text into
    `content` because the generated stream often starts after a prompt-prefilled
    `<think>` token.

Fix:
  - Capture `enable_thinking` from chat template kwargs in the parser.
  - Route disabled-thinking requests as plain content.
  - Route explicit-thinking deltas into the reasoning lane even when the start
    token is already present in the prompt.
"""

from pathlib import Path


path = Path("/app/vllm/vllm/reasoning/qwen3_reasoning_parser.py")
content = path.read_text()

if "self.thinking_enabled = bool(chat_template_kwargs.get(\"enable_thinking\", True))" in content:
    print("SKIP: qwen35 reasoning routing fix already present")
    raise SystemExit(0)

replacement = """# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
)
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser
from vllm.tokenizers import TokenizerLike


class Qwen3ReasoningParser(BaseThinkingReasoningParser):
    \"""
    Reasoning parser for the Qwen3 model.

    Qwen3 uses <think>...</think> tokens to denote reasoning text, but when
    `<think>` is already present in the rendered prompt the generated stream
    often begins with reasoning text directly. This parser keeps the legacy
    Qwen3 non-streaming behavior while adding enough state to distinguish:
    - thinking disabled: return plain content
    - thinking enabled with prompt-prefilled <think>: treat deltas as reasoning
      until </think> appears
    \"""

    @property
    def start_token(self) -> str:
        return "<think>"

    @property
    def end_token(self) -> str:
        return "</think>"

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        chat_template_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        self.thinking_enabled = bool(chat_template_kwargs.get("enable_thinking", True))

    def extract_reasoning(
        self, model_output: str, request: ChatCompletionRequest | ResponsesRequest
    ) -> tuple[str | None, str | None]:
        \"""
        Extract reasoning content from the full model output.

        When thinking is disabled, all generated text should remain content.
        When thinking is enabled but the generation has not yet closed with
        </think>, treat the whole output as reasoning rather than leaking it
        into final content.
        \"""
        if not self.thinking_enabled:
            return None, model_output

        if self.end_token not in model_output:
            return model_output, None

        return super().extract_reasoning(model_output, request)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        \"""
        Extract reasoning deltas for Qwen3 streaming output.

        Qwen3 may start streaming reasoning text without re-emitting `<think>`
        because the token already exists in the prompt. In that case, deltas
        before `</think>` must still be routed to the reasoning lane.
        \"""
        if not self.thinking_enabled:
            return DeltaMessage(content=delta_text) if delta_text else None

        if (
            self.start_token_id not in previous_token_ids
            and self.start_token_id not in delta_token_ids
        ):
            if self.end_token_id in delta_token_ids:
                end_index = delta_text.find(self.end_token)
                reasoning = delta_text[:end_index] if end_index >= 0 else delta_text
                content = (
                    delta_text[end_index + len(self.end_token) :]
                    if end_index >= 0
                    else None
                )
                return DeltaMessage(
                    reasoning=reasoning or None,
                    content=content or None,
                )

            if self.end_token_id in previous_token_ids:
                return DeltaMessage(content=delta_text) if delta_text else None

            return DeltaMessage(reasoning=delta_text) if delta_text else None

        return super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )
"""

path.write_text(replacement)
print("Applied Qwen3.5 reasoning routing fix")
