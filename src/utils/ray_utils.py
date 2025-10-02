import ray


def get_available_gpus() -> int | None:
    """Get the number of available GPUs in the Ray cluster.

    Returns
    -------
    int | None
        Number of available GPUs in the cluster, or None if GPU resource is not present
    """
    cluster_resources = ray.cluster_resources()
    gpu_count = cluster_resources.get("GPU")
    return int(gpu_count) if gpu_count is not None else None
