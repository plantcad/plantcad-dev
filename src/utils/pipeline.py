import json
from typing import Any
from pydantic import Field
from pydantic.dataclasses import dataclass
from upath import UPath
from src.io import open_file


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
