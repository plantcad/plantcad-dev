import ray


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
