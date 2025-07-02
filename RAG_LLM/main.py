import os, sys
from dotenv import load_dotenv
import argparse

load_dotenv()

sys.path.append(os.path.abspath(os.getenv("PYTHONPATH", ".")))

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["ollama", "chatbot", "debug"], help="select chat mode")
args = parser.parse_args()

if args.mode == "ollama":
    from apps.ollama_chat.interfaces.ollama_chat import main
elif args.mode == "chatbot":
    from RAG_LLM.apps.chats.interfaces.chatbot import main
elif args.mode == "debug":
    from RAG_LLM.apps.chats.interfaces.debug import main

if __name__ == "__main__":
    main()
    