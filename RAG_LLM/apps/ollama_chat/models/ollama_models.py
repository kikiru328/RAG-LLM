import subprocess
import socket
import os
from langchain_ollama import ChatOllama

def is_ollama_running(host="127.0.0.1", port=11434) -> bool:
    """Check Ollama from port 11434 (ollama port)"""
    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except (OSError, ConnectionRefusedError):
        return False

def start_ollama_serve_if_needed():
    """ollama serve if not started"""
    if is_ollama_running():
        print("Already started Ollama serve")
        return
    print("Start Ollama serve")
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=os.setpgrp  # 백그라운드 안정 실행
    )

def load_bllossom_latest():
    """
    load bllossom_latest Models from Ollama
    If not running, automatically start `ollama serve`
    """
    start_ollama_serve_if_needed()
    return ChatOllama(
        model="bllossom:latest",
        num_gpu=1,
        num_thread=4,
    )
