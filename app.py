import os
import time
from typing import Dict

import streamlit as st



from config import (
    APP_NAME,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL_NAME,
    DEFAULT_REPEAT_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_OLLAMA_HOST,
    STYLES_DIR,
)
from utils.chat_engine import build_messages, count_tokens, stream_response, trim_messages
from utils.memory_manager import (
    clear_session,
    export_session,
    list_sessions,
    load_session,
    load_settings,
    new_session,
    save_session,
    save_settings,
    update_title_if_needed,
)
from utils.model_loader import load_model, supports_gpu_offload
from utils.ui_helpers import (
    get_gpu_stats,
    get_ram_stats,
    inject_copy_script,
    load_css,
    scroll_to_bottom,
    typing_indicator_html,
)


st.set_page_config(page_title=APP_NAME, layout="wide", page_icon="🤖")
load_css(os.path.join(STYLES_DIR, "custom.css"))
inject_copy_script()


DEFAULT_SETTINGS: Dict[str, object] = {
    "model_name": DEFAULT_MODEL_NAME,
    "ollama_host": DEFAULT_OLLAMA_HOST,
    "context_length": DEFAULT_CONTEXT_LENGTH,
    "max_tokens": DEFAULT_MAX_TOKENS,
    "temperature": DEFAULT_TEMPERATURE,
    "top_p": DEFAULT_TOP_P,
    "repeat_penalty": DEFAULT_REPEAT_PENALTY,
    "system_prompt": "",
}

#defs

def ensure_settings() -> Dict[str, object]:
    if "settings" not in st.session_state:
        saved = load_settings()
        st.session_state.settings = {**DEFAULT_SETTINGS, **saved}
    return st.session_state.settings


def ensure_session() -> Dict:
    sessions = list_sessions()
    if not sessions:
        session = new_session()
        st.session_state.current_chat_id = session["id"]
        st.session_state.messages = session["messages"]
        return session
    current_id = st.session_state.get("current_chat_id", sessions[0]["id"])
    session = load_session(current_id)
    if session is None:
        session = sessions[0]
        st.session_state.current_chat_id = session["id"]
    st.session_state.messages = session.get("messages", [])
    return session


def format_session_label(session: Dict) -> str:
    updated = time.strftime("%Y-%m-%d %H:%M", time.localtime(session.get("updated_at", 0)))
    title = session.get("title", "Chat")
    return f"{title} • {updated} • {session.get('id', '')[:8]}"


def get_model_instance(settings: Dict[str, object]):
    model_name = str(settings["model_name"])
    ollama_host = str(settings.get("ollama_host", ""))
    params = (model_name, ollama_host)
    if st.session_state.get("model_params") != params:
        st.session_state.model = None
        st.session_state.model_params = params
    if st.session_state.get("model") is None:
        with st.spinner("Loading model..."):
            try:
                st.session_state.model = load_model(
                    model_name=model_name,
                    host=ollama_host,
                )
            except (OSError, RuntimeError, ValueError, FileNotFoundError) as exc:
                st.error(f"Model load failed: {exc}")
                st.stop()
    return st.session_state.model


settings = ensure_settings()
session = ensure_session()


st.sidebar.title(APP_NAME)

if st.sidebar.button("New Chat"):
    session = new_session()
    st.session_state.current_chat_id = session["id"]
    st.session_state.messages = []
    st.session_state.last_user_message = ""
    st.session_state.last_tokens_per_s = 0.0
    st.rerun()

sessions = list_sessions()
labels = [format_session_label(item) for item in sessions]
id_by_label = {label: session["id"] for label, session in zip(labels, sessions)}
label_by_id = {session["id"]: label for label, session in zip(labels, sessions)}
current_label = label_by_id.get(st.session_state.get("current_chat_id", ""), labels[0])
selected_index = labels.index(current_label) if current_label in labels else 0
selected_label = st.sidebar.selectbox("Chat History", options=labels, index=selected_index)
selected_id = id_by_label[selected_label]
if selected_id != st.session_state.get("current_chat_id"):
    session = load_session(selected_id)
    if session:
        st.session_state.current_chat_id = session["id"]
        st.session_state.messages = session.get("messages", [])
        st.session_state.last_user_message = ""
        st.rerun()

st.sidebar.subheader("Model Settings")
model_name = st.sidebar.text_input("Ollama model", value=str(settings["model_name"]))
ollama_host = st.sidebar.text_input("Ollama host", value=str(settings.get("ollama_host", "")), help="Leave blank to use the default local Ollama service at http://localhost:11434.")
context_length = st.sidebar.slider("Context length", 1024, 8192, int(settings["context_length"]), step=256)
max_tokens = st.sidebar.slider("Max tokens", 64, 2048, int(settings["max_tokens"]), step=32)
temperature = st.sidebar.slider("Temperature", 0.0, 1.5, float(settings["temperature"]), step=0.05)
top_p = st.sidebar.slider("Top P", 0.1, 1.0, float(settings["top_p"]), step=0.05)
repeat_penalty = st.sidebar.slider("Repeat penalty", 1.0, 1.5, float(settings["repeat_penalty"]), step=0.05)

system_prompt = st.sidebar.text_area("System prompt", value=str(settings["system_prompt"]), height=100)

settings_update = {
    "model_name": model_name,
    "ollama_host": ollama_host,
    "context_length": context_length,
    "max_tokens": max_tokens,
    "temperature": temperature,
    "top_p": top_p,
    "repeat_penalty": repeat_penalty,
    "system_prompt": system_prompt,
}
if settings_update != st.session_state.settings:
    st.session_state.settings = settings_update
    save_settings(settings_update)

st.sidebar.subheader("Model Info")
cuda_support = supports_gpu_offload()
st.sidebar.caption(f"GPU detected: {'Yes' if cuda_support else 'No'}")
st.sidebar.caption("Ollama chooses GPU automatically when available, otherwise it runs on CPU.")

model_status = st.sidebar.empty()
if st.session_state.get("model") is None:
    model_status.info("Model not loaded")
else:
    model_status.success("Model loaded")

if st.sidebar.button("Load Model"):
    _ = get_model_instance(settings_update)
    model_status.success("Model loaded")

if st.sidebar.button("Clear Chat"):
    session = clear_session(session)
    st.session_state.messages = session["messages"]
    st.session_state.last_user_message = ""
    st.rerun()

if st.sidebar.button("Stop generation"):
    st.session_state.stop_generation = True

if st.sidebar.button("Retry response"):
    st.session_state.retry_response = True

st.sidebar.subheader("System Usage")
ram = get_ram_stats()
st.sidebar.metric("RAM", f"{ram['used_gb']} / {ram['total_gb']} GB", f"{ram['percent']}%")
gpu = get_gpu_stats()
if gpu:
    st.sidebar.metric("VRAM (system)", f"{gpu['used_mb']} / {gpu['total_mb']} MB", f"{gpu['util']}%")

if st.session_state.get("last_tokens_per_s"):
    st.sidebar.metric("Tokens/sec", f"{st.session_state.last_tokens_per_s:.2f}")
if st.session_state.get("last_token_count"):
    st.sidebar.metric("Last output tokens", int(st.session_state.last_token_count))

st.sidebar.subheader("Optional Features")
st.sidebar.checkbox("RAG support (placeholder)", value=False, disabled=True)
st.sidebar.checkbox("PDF upload (placeholder)", value=False, disabled=True)
st.sidebar.checkbox("Image input (placeholder)", value=False, disabled=True)
st.sidebar.checkbox("Voice input (placeholder)", value=False, disabled=True)

st.title(APP_NAME)
st.caption("Local-only ChatGPT-style assistant powered by Ollama. GPU is used automatically when available.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

scroll_to_bottom()

export_col, _ = st.columns([1, 3])
with export_col:
    export_json = export_session(session, "json")
    st.download_button("Export chat (JSON)", export_json, file_name=f"{session['id']}.json")

prompt = st.chat_input("Type your message")
retry_requested = st.session_state.pop("retry_response", False)
prompt_to_process = None
retry_mode = False

if retry_requested and st.session_state.get("last_user_message"):
    prompt_to_process = st.session_state.last_user_message
    retry_mode = True
elif prompt:
    prompt_to_process = prompt
#needs optimization btw
if prompt_to_process:
    if not retry_mode:
        st.session_state.messages.append({"role": "user", "content": prompt_to_process})
        update_title_if_needed(session, prompt_to_process)
        session["messages"] = st.session_state.messages
        save_session(session)
        st.session_state.last_user_message = prompt_to_process
    else:
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
            st.session_state.messages.pop()
            session["messages"] = st.session_state.messages
            save_session(session)

    st.session_state.stop_generation = False
    model = get_model_instance(settings_update)

    messages = build_messages(st.session_state.messages, system_prompt)
    messages = trim_messages(messages, context_length, max_tokens)

    with st.chat_message("assistant"):
        typing_placeholder = st.empty()
        typing_placeholder.markdown(typing_indicator_html(), unsafe_allow_html=True)
        message_placeholder = st.empty()
        start_time = time.time()
        assistant_text = ""
        stream_stats: Dict[str, float] = {}
        try:
            for delta in stream_response(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                repeat_penalty=repeat_penalty,
                stop_sequences=[],
                should_stop=lambda: st.session_state.get("stop_generation", False),
                stats=stream_stats,
            ):
                assistant_text += delta
                message_placeholder.markdown(assistant_text + "▌")
        except (RuntimeError, ValueError) as exc:
            st.error(f"Generation failed: {exc}")
            st.stop()
        message_placeholder.markdown(assistant_text)
        typing_placeholder.empty()

    elapsed = max(0.001, time.time() - start_time)
    token_count = int(stream_stats.get("output_tokens", count_tokens(assistant_text))) if assistant_text else 0
    st.session_state.last_tokens_per_s = token_count / elapsed
    st.session_state.last_token_count = token_count

    if assistant_text:
        st.session_state.messages.append({"role": "assistant", "content": assistant_text})
        session["messages"] = st.session_state.messages
        save_session(session)
#code to review and opt
    scroll_to_bottom()
