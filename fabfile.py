"""
Fabric tasks for SkyPilot/Ray cluster management.

Examples:
    fab exec --cmd="nvidia-smi"
    fab exec --cmd="cat /tmp/ray_pc-dev/session_latest/logs/worker-*"
"""

from fabric import task
from invoke import Context
import os
import re
import subprocess
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


def _default_ssh_config():
    from src.constants import RAY_CLUSTER_NAME

    return os.path.expanduser(f"~/.sky/generated/ssh/{RAY_CLUSTER_NAME}")


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
