"""
Logging
"""

from abc import ABC
import json
import time
from collections import defaultdict
from typing import Any, Dict, Optional, Sequence, Tuple, DefaultDict


DEBUG = 10
INFO = 20
WARNING = 30
ERROR = 40
CRITICAL = 50

DISABLED = 60

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[0m"

level_names = {
    DEBUG   : "DEBUG",
    INFO    : "INFO",
    WARNING : "WARNING",
    ERROR   : "ERROR",
    CRITICAL: "CRITICAL",
}

level_names_max_len = max(len(name) for name in level_names.values())


## Backends
class LogEvent:
    step: Optional[int] = None
    data: Dict[str, Tuple[str, Any]]
    timestamp: float
    def __init__(self, data: Dict[str, Any], step: Optional[int], timestamp: float) -> None:
        self.data = data
        self.step = step
        self.timestamp = timestamp

class Backend (ABC):
    """
    Abstract base class for logging backends
    """
    def log(self, event: LogEvent) -> None:
        raise NotImplementedError
    def log_multiple(self, events: Sequence[LogEvent]) -> None:
        for event in events:
            self.log(event)
    def close(self) -> None:
        pass

class JSONBackend(Backend):
    """
    Log to a JSON file
    """
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.file = open(self.filename, "wt")

    def log(self, event: LogEvent) -> None:
        if event.step is None:
            record = {
                "timestamp": event.timestamp,
                "data": event.data,
            }
        else:
            record = {
                "timestamp": event.timestamp,
                "step": event.step,
                "data": event.data,
            }
        self.file.write(json.dumps(record) + "\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()

class StdoutBackend(Backend):
    """
    Log to stdout
    """
    def log(self, event: LogEvent) -> None:
        out = f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(event.timestamp))}]    "
        if event.step is not None:
            out += f"Step: {event.step}    "
        out += "\n"
        for k, v in sorted(event.data.items()):
            out += f"\t ({v[0]}){' '*(level_names_max_len - len(v[0]))} {k}: {v[1]}\n"

        print(out, end="")
    def close(self) -> None:
        pass

class GroupedBackends(Backend):
    """
    Log to multiple backends
    """
    def __init__(self, backends: Sequence[Backend]) -> None:
        self.backends = backends

    def log(self, event: LogEvent) -> None:
        for backend in self.backends:
            backend.log(event)

    def close(self) -> None:
        for backend in self.backends:
            backend.close()



## API
class Logger:
    """
    Collects kvs and flushes them to the backend
    """
    CURRENT: Optional["Logger"] = None

    @staticmethod
    def get_current() -> "Logger":
        if Logger.CURRENT is None:
            raise ValueError("No current Logger set")
        return Logger.CURRENT
    @staticmethod
    def set_current(logger: "Logger") -> None:
        Logger.CURRENT = logger

    def __init__(self, backend: Backend, step_key: str = "step", level: int = INFO) -> None:
        self.kvs: DefaultDict[str, Any] = defaultdict(list)
        self.backend = backend
        self.step_key = step_key
        self.step: Optional[int] = None
        self.level = level

    def set_step(self, step: int) -> None:
        self.step = step

    def logkv(self, key: str, value: Any, level: int = INFO) -> None:
        if self.level == DISABLED:
            return

        if level < self.level:
            return
        self.kvs[key] = (level_names[level], value)

    def dumpkvs(self) -> None:
        if self.level == DISABLED:
            return

        event = LogEvent(data=self.kvs, step=self.step, timestamp=time.time())
        self.backend.log(event)
        self.kvs.clear()

    def close(self) -> None:
        self.backend.close()



