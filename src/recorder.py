from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlRecorder:
    def __init__(self, path: str, flush_interval: int = 15) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.flush_interval = max(1, int(flush_interval))
        self._counter = 0
        self._fp = self.path.open("a", encoding="utf-8")

    def write(self, payload: dict[str, Any]) -> None:
        self._fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._counter += 1
        if self._counter % self.flush_interval == 0:
            self._fp.flush()

    def close(self) -> None:
        if self._fp.closed:
            return
        self._fp.flush()
        self._fp.close()
