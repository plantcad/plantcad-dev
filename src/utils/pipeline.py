import json
from typing import Any
from pydantic import Field
from pydantic.dataclasses import dataclass
from upath import UPath
from src.io import open_file
import asyncio
import ray


@ray.remote(max_restarts=-1)
class AsyncLock:
    def __init__(self):
        self._lock = asyncio.Lock()
        self._holder = None

    async def acquire(self, token: str, timeout_sec: float = None) -> bool:
        try:
            if timeout_sec is None:
                await self._lock.acquire()
            else:
                await asyncio.wait_for(self._lock.acquire(), timeout=timeout_sec)
            self._holder = token
            return True
        except asyncio.TimeoutError:
            return False

    async def release(self, token: str):
        if self._lock.locked() and self._holder == token:
            self._holder = None
            self._lock.release()

    async def is_locked(self) -> bool:
        return self._lock.locked()


@dataclass
class BaseStepConfig:
    """Base configuration for pipeline steps."""

    input_path: Any = Field(default=None, description="Input path for pipeline data")
    output_path: Any = Field(default=None, description="Output path for pipeline data")


def save_step_json(output_path: UPath, data: dict) -> None:
    """Save step data to JSON file."""
    with open_file(output_path / "step.json", "w") as f:
        json.dump(data, f, indent=2)


def load_step_json(input_path: UPath) -> dict:
    """Load step data from JSON file."""
    with open_file(input_path / "step.json", "r") as f:
        return json.load(f)
