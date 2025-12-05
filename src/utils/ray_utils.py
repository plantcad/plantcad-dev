from collections.abc import Callable
from typing import Any, TypeVar

import ray
from ray.util.placement_group import (
    placement_group,
    remove_placement_group,
)

T = TypeVar("T")


def num_cluster_gpus() -> int:
    """Get the number of GPUs in the Ray cluster.

    Returns
    -------
    int
        Number of GPUs in the cluster

    Raises
    ------
    ValueError
        If GPU resource is not present in the cluster
    """
    cluster_resources = ray.cluster_resources()
    gpu_count = cluster_resources.get("GPU")
    if gpu_count is None:
        raise ValueError("GPU resource not found in Ray cluster")
    return int(gpu_count)


def num_cluster_cpus() -> int:
    """Get the number of CPUs in the Ray cluster.

    Returns
    -------
    int
        Number of CPUs in the cluster

    Raises
    ------
    ValueError
        If CPU resource is not present in the cluster
    """
    cluster_resources = ray.cluster_resources()
    cpu_count = cluster_resources.get("CPU")
    if cpu_count is None:
        raise ValueError("CPU resource not found in Ray cluster")
    return int(cpu_count)


def num_cluster_nodes() -> int:
    """Get the number of nodes in the Ray cluster.

    Returns
    -------
    int
        Number of nodes in the cluster

    Raises
    ------
    ValueError
        If no nodes are found in the cluster
    """
    nodes = ray.nodes()
    if not nodes:
        raise ValueError("No nodes found in Ray cluster")
    return len(nodes)


def num_cpus_per_node() -> int:
    """Get the number of CPUs per node in the Ray cluster.

    WARNING: This function assumes that all nodes have the same number of CPUs.

    Returns
    -------
    int
        Number of CPUs per node in the cluster

    Raises
    ------
    ValueError
        If CPU or node resources cannot be determined
    """
    cpus = num_cluster_cpus()
    nodes = num_cluster_nodes()
    return cpus // nodes


@ray.remote
class NodeFunctionExecutor:
    def __init__(self, func: Callable[..., Any]):
        self.func = func

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        return self.func(*args, **kwargs)


def run_once_per_node(
    func: Callable[..., T],
    num_cpus: int = 1,
    num_gpus: int = 0,
    options: dict[str, Any] | None = None,
    *args: Any,
    **kwargs: Any,
) -> list[T]:
    """Execute a function once per node in the Ray cluster and return results.

    Automatically cleans up actors and placement group after execution.

    Parameters
    ----------
    func : Callable[..., T]
        Function to execute on each node
    num_cpus : int, default=1
        Number of CPUs to allocate per node for the placement group
    num_gpus : int, default=0
        Number of GPUs to allocate per node for the placement group
    options : dict[str, Any] | None, optional
        Options to pass to ActorClass.options(). See:
        https://docs.ray.io/en/latest/ray-core/api/doc/ray.actor.ActorClass.options.html
        Note: placement_group will be automatically set.
    *args : Any
        Positional arguments to pass to the function
    **kwargs : Any
        Keyword arguments to pass to the function

    Returns
    -------
    list[T]
        List of results from executing the function on each node
    """
    # Validate options don't conflict with explicit resource arguments
    if options is not None:
        if "num_cpus" in options:
            raise ValueError("num_cpus cannot be set in options; use num_cpus argument")
        if "num_gpus" in options:
            raise ValueError("num_gpus cannot be set in options; use num_gpus argument")

    num_nodes = num_cluster_nodes()
    bundle: dict[str, int] = {"CPU": num_cpus}
    if num_gpus > 0:
        bundle["GPU"] = num_gpus
    bundles = [bundle for _ in range(num_nodes)]
    pg = placement_group(bundles=bundles, strategy="STRICT_SPREAD")
    ray.get(pg.ready())

    # Create one actor per node, each bound to a different bundle in the placement group
    opts: dict[str, Any] = {
        **(options or {}),
        "placement_group": pg,
        "num_cpus": num_cpus,
        "num_gpus": num_gpus,
    }
    actors = [
        # pyrefly: ignore[missing-attribute]
        NodeFunctionExecutor.options(
            **{**opts, "placement_group_bundle_index": i}
        ).remote(func)
        for i in range(num_nodes)
    ]

    try:
        futures = [actor.execute.remote(*args, **kwargs) for actor in actors]
        results = ray.get(futures)
        return results
    finally:
        # Clean up actors and placement group
        for actor in actors:
            ray.kill(actor)
        remove_placement_group(pg)


def run_once_per_gpu(
    func: Callable[..., Any],
    args: list[tuple[tuple[Any, ...], dict[str, Any]]],
    options: dict[str, Any] | None = None,
) -> list[ray.ObjectRef]:
    """Execute a function once per GPU with different arguments for each invocation.

    Parameters
    ----------
    func : Callable[..., Any]
        Function to execute on each GPU
    args : list[tuple[tuple[Any, ...], dict[str, Any]]]
        List of (args, kwargs) tuples, one per GPU invocation
    options : dict[str, Any] | None, optional
        Options to pass to RemoteFunction.options(). See:
        https://docs.ray.io/en/latest/ray-core/api/doc/ray.remote_function.RemoteFunction.options.html
        Note: num_gpus will be automatically set to 1 per task.

    Returns
    -------
    list[ray.ObjectRef]
        List of futures for the function execution results
    """
    opts = {**(options or {}), "num_gpus": 1}
    remote_func = ray.remote(func).options(**opts)
    return [remote_func.remote(*a, **kw) for a, kw in args]


def run_on_exclusive_node(
    func: Callable[..., Any],
    options: dict[str, Any] | None = None,
    *args: Any,
    **kwargs: Any,
) -> ray.ObjectRef:
    """Execute a function on a single node using all available CPUs.

    Parameters
    ----------
    func : Callable[..., Any]
        Function to execute on the node
    options : dict[str, Any] | None, optional
        Options to pass to RemoteFunction.options(). See:
        https://docs.ray.io/en/latest/ray-core/api/doc/ray.remote_function.RemoteFunction.options.html
        Note: num_cpus will be automatically set to use all CPUs on a node.
    *args : Any
        Positional arguments to pass to the function
    **kwargs : Any
        Keyword arguments to pass to the function

    Returns
    -------
    ray.ObjectRef
        Future for the function execution result
    """
    num_cpus = num_cpus_per_node()
    opts = {**(options or {}), "num_cpus": num_cpus}
    remote_func = ray.remote(func).options(**opts)
    return remote_func.remote(*args, **kwargs)
