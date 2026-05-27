from __future__ import annotations

from typing import Callable, Dict, Iterable, List

from utils.model_loader import OllamaChatModel


def approx_token_count(text: str) -> int:
    if not text.strip():
        return 0
    return max(1, int(len(text) / 4))


def build_messages(history: List[Dict[str, str]], system_prompt: str) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.extend(history)
    return messages


def count_tokens(text: str) -> int:
    return approx_token_count(text)


def _message_tokens(message: Dict[str, str]) -> int:
    return count_tokens(message.get("content", ""))


def trim_messages(
    messages: List[Dict[str, str]],
    max_context_tokens: int,
    max_output_tokens: int,
) -> List[Dict[str, str]]:
    if max_context_tokens <= max_output_tokens + 128:
        return messages[-4:]
    budget = max_context_tokens - max_output_tokens
    trimmed = list(messages)
    total = sum(_message_tokens(msg) for msg in trimmed)
    while total > budget and len(trimmed) > 1:
        remove_index = 1 if trimmed[0]["role"] == "system" and len(trimmed) > 1 else 0
        total -= _message_tokens(trimmed[remove_index])
        trimmed.pop(remove_index)
    return trimmed


def stream_response(
    model: OllamaChatModel,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    top_p: float,
    repeat_penalty: float,
    stop_sequences: List[str],
    should_stop: Callable[[], bool],
    stats: Dict[str, float] | None = None,
) -> Iterable[str]:
    request_options = {
        "temperature": temperature,
        "top_p": top_p,
        "repeat_penalty": repeat_penalty,
        "num_predict": max_tokens,
    }
    if stop_sequences:
        request_options["stop"] = stop_sequences

    stream = model.client.chat(
        model=model.model_name,
        messages=messages,
        options=request_options,
        stream=True,
    )
    for chunk in stream:
        if should_stop():
            break
        if stats is not None:
            if chunk.get("eval_count") is not None:
                stats["output_tokens"] = float(chunk["eval_count"])
            if chunk.get("prompt_eval_count") is not None:
                stats["input_tokens"] = float(chunk["prompt_eval_count"])
        content = chunk.get("message", {}).get("content", "")
        if content:
            yield content