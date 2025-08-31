from typing import Callable, Any
import ray
from thalas.execution import (
    Executor,
    ExecutorStep,
    output_path_of,
    this_output_path,
    versioned,
)
from thalas.execution.executor import ConfigT as StepConfig
from dataclasses import dataclass


@dataclass(frozen=True)
class MyConfig:
    input_path: str
    output_path: str
    n: int
    m: int


def executor_step(name: str, config: StepConfig = None, description: str | None = None):
    """Decorator to create ExecutorStep from a function."""

    def decorator(func: Callable[[StepConfig], Any]) -> ExecutorStep[StepConfig]:
        return ExecutorStep(name=name, fn=func, config=config, description=description)

    return decorator


@executor_step(name="a", description="step1")
def decostep1(config: StepConfig = None):
    print("in step1")


@executor_step(
    name="b",
    config=MyConfig(
        input_path=output_path_of(decostep1, "sub"),
        output_path=this_output_path(),
        n=versioned(4),
        m=5,
    ),
    description="step2",
)
def decostep2(config: MyConfig):
    print("in step2")
    print(f"Config: {config}")


def step1(config: StepConfig = None):
    print("in step1")


def step2(config: MyConfig):
    print("in step2")
    print(f"Config: {config}")


class Pipeline:
    """Pipeline class with step methods that return ExecutorStep objects."""

    def __init__(self, config: dict):
        self.config = config
        print(f"Pipeline config: {self.config}")

    def step1(self) -> ExecutorStep:
        """First step of the pipeline."""
        return ExecutorStep(
            name="pipeline_step1", fn=step1, config={}, description="Pipeline step1"
        )

    def step2(self) -> ExecutorStep:
        """Second step of the pipeline."""
        config = MyConfig(
            input_path=output_path_of(self.step1(), "sub"),
            output_path=this_output_path(),
            n=self.config.get("param1", 4),
            m=self.config.get("param2", 5),
        )
        return ExecutorStep(
            name="pipeline_step2", fn=step2, config=config, description="Pipeline step2"
        )


def main():
    ray.init(
        address="local",
        namespace="pipeline",
        ignore_reinit_error=True,
        resources={"head_node": 1},
    )
    temp_dir = "/tmp/thalas_test"

    # Run original decorator-based example
    print("=== Running decorator-based example ===")
    executor1 = Executor(
        prefix=temp_dir + "/decorator", executor_info_base_path=temp_dir
    )
    executor1.run(steps=[decostep1, decostep2])

    # Run new Pipeline class example
    print("\n=== Running Pipeline class example ===")
    pipeline_config = {
        "param1": "test_value",
        "param2": 42,
        "description": "Example pipeline configuration",
    }
    pipeline = Pipeline(config=pipeline_config)
    executor2 = Executor(
        prefix=temp_dir + "/pipeline", executor_info_base_path=temp_dir
    )
    executor2.run(steps=[pipeline.step1(), pipeline.step2()])


if __name__ == "__main__":
    main()
