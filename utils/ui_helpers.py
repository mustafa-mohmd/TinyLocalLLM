from __future__ import annotations

import os
import shutil
import subprocess
from typing import Dict, Optional

import psutil
import streamlit as st
import streamlit.components.v1 as components


def load_css(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        css = handle.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def inject_copy_script() -> None:
    script = """
    <script>
    function addCopyButtons() {
      const blocks = window.parent.document.querySelectorAll('pre > code');
      blocks.forEach((block) => {
        const pre = block.parentElement;
        if (pre.querySelector('.copy-btn')) return;
        const button = document.createElement('button');
        button.className = 'copy-btn';
        button.innerText = 'Copy';
        button.onclick = () => {
          navigator.clipboard.writeText(block.innerText);
          button.innerText = 'Copied';
          setTimeout(() => button.innerText = 'Copy', 1200);
        };
        pre.style.position = 'relative';
        pre.appendChild(button);
      });
    }
    const observer = new MutationObserver(addCopyButtons);
    observer.observe(window.parent.document.body, { childList: true, subtree: true });
    setTimeout(addCopyButtons, 1000);
    </script>
    """
    components.html(script, height=0, width=0)


def typing_indicator_html() -> str:
    return """
    <div class="typing-indicator">
      <span></span><span></span><span></span>
    </div>
    """


def scroll_to_bottom() -> None:
    script = """
    <script>
    const container = window.parent.document.querySelector('.main');
    if (container) {
      container.scrollTop = container.scrollHeight;
    }
    </script>
    """
    components.html(script, height=0, width=0)


def get_ram_stats() -> Dict[str, float]:
    memory = psutil.virtual_memory()
    used_gb = (memory.total - memory.available) / (1024 ** 3)
    total_gb = memory.total / (1024 ** 3)
    return {
        "used_gb": round(used_gb, 2),
        "total_gb": round(total_gb, 2),
        "percent": round(memory.percent, 1),
    }


def get_gpu_stats() -> Optional[Dict[str, float]]:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError:
        return None
    parts = output.strip().split(",")
    if len(parts) < 3:
        return None
    try:
        used = float(parts[0].strip())
        total = float(parts[1].strip())
        util = float(parts[2].strip())
    except ValueError:
        return None
    return {"used_mb": used, "total_mb": total, "util": util}