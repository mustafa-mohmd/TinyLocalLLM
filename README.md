# LocalLLM

A lightweight local LLM chat application built with Python, supporting GGUF models, persistent chat history, custom styling, and modular utilities for memory and model management.

## Features

- Run local GGUF language models
- Persistent chat history storage
- Modular chat engine architecture
- Memory/context management
- Custom CSS styling support
- Configurable application settings
- Lightweight and easy to extend

model : google/gemma-3-4b-it-qat-q4_0-gguf | huggingface

---

## Project Structure

```text
LOCALLLM/
│
├── assets/
│
├── chat_history/
│   ├── *.json
│   └── settings.json
│
├── models/
│   ├── gemma-3-4b-it-q4_0.gguf
│   ├── mmproj-model-f16-4B.gguf
│   └── README.md
│
├── styles/
│   └── custom.css
│
├── utils/
│   ├── chat_engine.py
│   ├── memory_manager.py
│   ├── model_loader.py
│   └── ui_helpers.py
│
├── app.py
├── config.py
├── requirements.txt
├── test.py
└── README.md