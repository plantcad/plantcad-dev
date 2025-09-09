"""
Fabric tasks for SkyPilot/Ray cluster management.

Examples:
    fab exec --cmd="nvidia-smi"
    fab exec --cmd="cat /tmp/ray-pc-dev/session_latest/logs/worker-*"
"""

from fabric import task
from invoke import Context
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HOST_LINE = re.compile(r"^\s*Host\s+(.+)$")


def parse_hosts(path):
    hosts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip().startswith("#"):
                continue
            m = HOST_LINE.match(line)
            if not m:
                continue
            for token in m.group(1).split():
                if "*" not in token:
                    hosts.append(token)
    return hosts


def _get_cluster_name():
    from src.constants import RAY_CLUSTER_NAME

    return RAY_CLUSTER_NAME


def _default_ssh_config():
    return os.path.expanduser(f"~/.sky/generated/ssh/{_get_cluster_name()}")


@task
def exec(c: Context, cmd: str, ssh_config: str | None = None):
    """Run an arbitrary shell command on all hosts from the ssh config."""
    ssh_config = os.path.expanduser(ssh_config or _default_ssh_config())
    if not Path(ssh_config).exists():
        print(f"SSH config not found: {ssh_config}")
        return

    hosts = parse_hosts(ssh_config)
    if not hosts:
        print("No hosts found")
        return

    for host in hosts:
        print(f"\n========== {host} ==========")
        subprocess.run(
            ["ssh", "-F", ssh_config, host, cmd],
            check=False,
        )


@task
def ray_address(c: Context):
    """Get the Ray address from the cluster for local Ray client connections."""
    cluster_name = _get_cluster_name()
    result = subprocess.run(
        ["ssh", cluster_name, f"cat /tmp/ray-{cluster_name}/ray_current_cluster"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0 and result.stdout.strip():
        ray_address = result.stdout.strip()
        print(ray_address)
    else:
        print(f"# Failed to get Ray address from {cluster_name}", file=sys.stderr)
        if result.stderr:
            print(f"# SSH error: {result.stderr.strip()}", file=sys.stderr)


@task
def cluster_nodes(c: Context, ssh_config: str | None = None):
    """List all cluster hosts with internal IPs and types as JSON objects."""
    ssh_config = os.path.expanduser(ssh_config or _default_ssh_config())
    if not Path(ssh_config).exists():
        print(f"SSH config not found: {ssh_config}")
        return

    hosts = parse_hosts(ssh_config)
    if not hosts:
        print("No hosts found")
        return

    host_results = []

    for host in hosts:
        # Get internal IP via SSH
        result = subprocess.run(
            ["ssh", host, "hostname -I | awk '{print $1}'"],
            capture_output=True,
            text=True,
            check=False,
        )

        internal_ip = result.stdout.strip() if result.returncode == 0 else None

        # Determine host type based on suffix
        host_type = "worker" if re.search(r"-worker\d+$", host) else "head"

        host_info = {"hostname": host, "internal_ip": internal_ip, "type": host_type}

        host_results.append(host_info)

    # Validate exactly one head node
    head_count = sum(1 for h in host_results if h["type"] == "head")
    if head_count != 1:
        print(
            f"Error: Expected exactly 1 head node, found {head_count}", file=sys.stderr
        )
        return

    # Print results
    for host_info in host_results:
        print(json.dumps(host_info))
