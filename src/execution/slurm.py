"""Generic SLURM execution utilities using submitit."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TypeVar
from types import ModuleType

logger = logging.getLogger(__name__)

T = TypeVar("T")


def import_submitit() -> ModuleType:
    """Import and return submitit module, raising helpful error if not available."""
    try:
        # Guard this import as it is an optional dependency
        import submitit

        return submitit
    except ImportError:
        raise ImportError(
            "submitit is required for SLURM execution. Install with: uv sync --extra slurm"
        )


@dataclass
class ProcessGroup:
    """Information about the current SLURM process group."""

    rank: int
    world_size: int


def process_group() -> ProcessGroup:
    """Get the current SLURM process group information.

    Returns
    -------
    ProcessGroup
        Information about the current process rank and world size.
    """
    submitit = import_submitit()
    job_env = submitit.JobEnvironment()
    return ProcessGroup(
        rank=job_env.global_rank,
        world_size=job_env.num_tasks,
    )


def run_slurm_function(
    func: Callable[[], T],
    log_dir: Path,
    partition: str,
    timeout_min: int = 30,
    nodes: int = 1,
    tasks_per_node: int = 1,
    cluster: str = "slurm",
) -> list[T]:
    """Run a function on SLURM cluster with distributed processing.

    This function submits a job to SLURM that will run the provided function
    across multiple tasks/nodes. The function should be designed to handle
    SLURM job environment (rank/world_size) internally if needed.

    Parameters
    ----------
    func
        Function to execute on SLURM. Use functools.partial
        to create parameterless functions from functions that require arguments.
    log_dir
        Directory where SLURM job logs will be written.
    partition
        SLURM partition to use.
    timeout_min
        Job timeout in minutes.
    nodes
        Number of nodes to request.
    tasks_per_node
        Number of tasks per node.
    cluster
        Cluster type (usually "slurm").

    Returns
    -------
    list[T]
        List of results from each SLURM task. Length equals nodes * tasks_per_node.

    Raises
    ------
    ImportError
        If submitit is not available.
    """
    submitit = import_submitit()

    # Ensure log directory exists
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    executor = submitit.AutoExecutor(
        folder=log_dir,
        cluster=cluster,
    )

    executor_params = {
        "timeout_min": timeout_min,
        "slurm_partition": partition,
        "tasks_per_node": tasks_per_node,
        "nodes": nodes,
    }
    executor.update_parameters(**executor_params)

    total_tasks = nodes * tasks_per_node
    logger.info(f"Submitting SLURM job: {func}")
    logger.info(f"  Parameters: {executor_params}")
    logger.info(f"  Total tasks: {total_tasks}")

    # Submit the job
    job = executor.submit(func)

    logger.info(f"Submitted job {job.job_id}, awaiting completion...")
    results = job.results()

    logger.info(f"SLURM job {job.job_id} completed successfully")
    logger.info(f"Collected {len(results)} results from {total_tasks} tasks")

    return results
