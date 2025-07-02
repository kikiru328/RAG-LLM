import logging
import os
from datetime import datetime

LOG_DIR: str = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_filename = os.path.join(LOG_DIR, f"{datetime.now().strftime('%Y-%m-%d')}.log")
class MyLoggerOnly(logging.Filter):
    def filter(self, record):
        return record.name == "GPT-CHAT-BOT"

file_handler = logging.FileHandler(log_filename, encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.addFilter(MyLoggerOnly())

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[file_handler, console_handler],
)

logger = logging.getLogger("GPT-CHAT-BOT")
logger.setLevel(logging.DEBUG)