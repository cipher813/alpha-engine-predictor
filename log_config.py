"""
Structured logging configuration.

JSON mode activates when ALPHA_ENGINE_JSON_LOGS=1.
Text mode (default) preserves human-readable format for local dev.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "func": record.funcName,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            entry["exc"] = self.formatException(record.exc_info)
        if hasattr(record, "ctx"):
            entry["ctx"] = record.ctx
        return json.dumps(entry, default=str)


def setup_logging(name: str = "alpha-engine") -> None:
    json_mode = os.environ.get("ALPHA_ENGINE_JSON_LOGS", "0") == "1"
    handler = logging.StreamHandler()
    if json_mode:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            f"%(asctime)s %(levelname)s [{name}] %(message)s"
        ))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
