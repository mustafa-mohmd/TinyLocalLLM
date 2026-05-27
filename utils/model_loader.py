from __future__ import annotations

from dataclasses import dataclass
import os
import shutil

import ollama

import streamlit as st


@dataclass(frozen=True)
class OllamaChatModel:
    client: ollama.Client
    model_name: str


def supports_gpu_offload() -> bool:
    return shutil.which("nvidia-smi") is not None


@st.cache_resource(show_spinner=False)
def load_model(model_name: str, host: str = "") -> OllamaChatModel:
    if not model_name.strip():
        raise ValueError("Model name cannot be empty")

    client = ollama.Client(host=host) if host.strip() else ollama.Client()
    try:
        client.list()
    except Exception as exc:
        raise RuntimeError(
            "Unable to connect to Ollama. Start the Ollama service and try again."
        ) from exc

    try:
        client.show(model_name)
    except Exception as exc:
        raise FileNotFoundError(
            f"Ollama model not found: {model_name}. Pull it first with `ollama pull {model_name}`."
        ) from exc

    return OllamaChatModel(client=client, model_name=model_name)


def clear_model_cache() -> None:
    load_model.clear()