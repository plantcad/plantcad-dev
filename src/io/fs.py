"""Generic filesystem operations with backend-specific optimizations.

This module provides a unified interface for file operations across different
storage backends (local, S3, HuggingFace). It delegates to backend-specific
implementations where appropriate for optimal performance.
"""

import logging
import shutil
from pathlib import Path

from fsspec import AbstractFileSystem
from upath import UPath

from src.io import hf as hf_module
from src.io import s3 as s3_module

logger = logging.getLogger("ray")

PathLike = str | Path | UPath


def resolve_path(path: PathLike) -> UPath:
    """Resolve a path to a UPath.

    Parameters
    ----------
    path : PathLike
        Path to resolve

    Returns
    -------
    UPath
        Resolved UPath with valid protocol
    """
    if not isinstance(path, UPath):
        path = UPath(path)

    # Handle empty protocol by resolving to absolute file path
    if path.protocol == "":
        path = UPath(path.as_uri())

    if not path.protocol:
        raise ValueError(f"Path must have a protocol, got {path.protocol}")

    return path


def resolve_filesystem(path: UPath) -> AbstractFileSystem:
    """Resolve a UPath to its corresponding filesystem.

    For s3 and hf protocols, delegates to backend-specific filesystem
    implementations with optimized configurations.

    Parameters
    ----------
    path : UPath
        Path to resolve filesystem for

    Returns
    -------
    AbstractFileSystem
        The filesystem instance for the given path
    """
    if path.protocol == "s3":
        return s3_module.filesystem()
    if path.protocol == "hf":
        return hf_module.filesystem()
    return path.fs


def _download_path(
    path: UPath,
    cache_dir: Path,
    recursive: bool,
    force: bool = True,
) -> str:
    """Download a remote path to local cache using fsspec and return the local path.

    Parameters
    ----------
    path : UPath
        Remote path to download
    cache_dir : Path
        Local cache directory
    recursive : bool
        Whether to download recursively (for directories)
    force : bool
        If True, always download even if local path exists.
        If False, skip download if local path already exists.

    Returns
    -------
    str
        Local filesystem path
    """
    # Create a local path within cache_dir that mirrors remote structure
    remote_path = path.path.lstrip("/")
    local_path = cache_dir / remote_path

    # Skip download if path exists and force=False
    if not force and local_path.exists():
        logger.info(f"Using cached path (force=False): {local_path}")
        return str(local_path)

    # Clear local path if it exists
    if local_path.exists():
        if local_path.is_dir():
            shutil.rmtree(local_path)
        else:
            local_path.unlink()
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # Execute download
    fs = resolve_filesystem(path)
    logger.info(f"Downloading {path} to {local_path}")
    fs.get(path.path, str(local_path), recursive=recursive)

    return str(local_path)


def download_file(path: PathLike, cache_dir: Path, force: bool = True) -> str:
    """Download a file to local cache and return the local path.

    For local paths, returns the path directly without copying.
    For HF paths, delegates to HF-specific download with caching.
    For other protocols, uses generic fsspec operations.

    Parameters
    ----------
    path : PathLike
        Path to download
    cache_dir : Path
        Cache directory for remote downloads
    force : bool
        If True, always download even if local path exists.
        If False, skip download if local path already exists.

    Returns
    -------
    str
        Local filesystem path to the file
    """
    path = resolve_path(path)

    # Short-circuit for local paths
    if path.protocol == "file":
        return path.path

    # HF has specialized download with caching
    if path.protocol == "hf":
        return hf_module.download_hf_file(path, cache_dir=cache_dir)

    # Generic fsspec download for other protocols
    cache_dir.mkdir(parents=True, exist_ok=True)
    return _download_path(path, cache_dir, recursive=False, force=force)


def download_folder(path: PathLike, cache_dir: Path, force: bool = True) -> str:
    """Download a folder to local cache and return the local path.

    For local paths, returns the path directly without copying.
    For HF paths, delegates to HF-specific download with caching.
    For other protocols, uses generic fsspec operations.

    Parameters
    ----------
    path : PathLike
        Path to download
    cache_dir : Path
        Cache directory for remote downloads
    force : bool
        If True, always download even if local path exists.
        If False, skip download if local path already exists.

    Returns
    -------
    str
        Local filesystem path to the folder
    """
    path = resolve_path(path)

    # Short-circuit for local paths
    if path.protocol == "file":
        return path.path

    # HF has specialized download with caching
    if path.protocol == "hf":
        return hf_module.download_hf_folder(path, cache_dir=cache_dir)

    # Generic fsspec download for other protocols
    cache_dir.mkdir(parents=True, exist_ok=True)
    return _download_path(path, cache_dir, recursive=True, force=force)


def upload_file(source: Path, destination: PathLike) -> None:
    """Upload a local file to a remote destination.

    For local destinations, copies the file.
    For HF destinations, delegates to HF-specific upload.
    For other protocols, uses generic fsspec operations.

    Parameters
    ----------
    source : Path
        Local file path to upload
    destination : PathLike
        Target path
    """
    destination = resolve_path(destination)

    # Short-circuit for local paths
    if destination.protocol == "file":
        dest_path = Path(destination.path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest_path)
        return

    # HF has specialized upload
    if destination.protocol == "hf":
        hf_module.upload_hf_file(source, destination)
        return

    # Generic fsspec upload for other protocols
    fs = resolve_filesystem(destination)
    logger.info(f"Uploading {source} to {destination}")
    fs.put(str(source), destination.path)


def upload_folder(source: Path, destination: PathLike) -> None:
    """Upload a local folder to a remote destination.

    For local destinations, copies the folder.
    For HF destinations, delegates to HF-specific upload.
    For other protocols, uses generic fsspec operations.

    Parameters
    ----------
    source : Path
        Local folder path to upload
    destination : PathLike
        Target path
    """
    destination = resolve_path(destination)

    # Short-circuit for local paths
    if destination.protocol == "file":
        dest_path = Path(destination.path)
        shutil.copytree(source, dest_path, dirs_exist_ok=True)
        return

    # HF has specialized upload
    if destination.protocol == "hf":
        hf_module.upload_hf_folder(source, destination)
        return

    # Generic fsspec upload for other protocols
    fs = resolve_filesystem(destination)
    logger.info(f"Uploading {source} to {destination}")
    fs.put(str(source), destination.path, recursive=True)
