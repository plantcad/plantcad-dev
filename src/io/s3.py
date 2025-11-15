import logging
import shutil
from pathlib import Path

import s3fs
from upath import UPath

logger = logging.getLogger("ray")

PathLike = str | UPath


def resolve_path(path: PathLike) -> UPath:
    """Convert a PathLike to UPath and validate s3:// protocol."""
    if not isinstance(path, UPath):
        path = UPath(path)
    if path.protocol != "s3":
        raise ValueError(f"Only s3:// protocol is supported, got: {path.protocol}")
    return path


def filesystem() -> s3fs.S3FileSystem:
    """Create an S3FileSystem instance with retry configuration.

    Returns
    -------
    s3fs.S3FileSystem
        An S3FileSystem instance configured with retries
    """
    # Configure retries via botocore config
    # See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/retries.html
    return s3fs.S3FileSystem(
        config_kwargs={
            "retries": {
                "max_attempts": 10,
                "mode": "adaptive",
            }
        }
    )


def _download_s3_path(
    path: PathLike,
    cache_dir: Path,
    recursive: bool,
) -> str:
    """Download an S3 path to local cache and return the local path."""
    path = resolve_path(path)

    # Create a local path within cache_dir that mirrors S3 structure
    # e.g. s3://bucket/path/to/file.txt -> cache_dir/bucket/path/to/file.txt
    s3_path = path.path.lstrip("/")
    local_path = cache_dir / s3_path

    # Clear local path if it exists
    if local_path.exists():
        if local_path.is_dir():
            shutil.rmtree(local_path)
        else:
            local_path.unlink()
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # Execute download
    fs = filesystem()
    logger.info(f"Downloading {path} to {local_path}")
    fs.get(path.path, str(local_path), recursive=recursive)

    return str(local_path)


def download_s3_file(path: PathLike, cache_dir: str | Path) -> str:
    """Download an S3 file to local cache and return the local path.

    Parameters
    ----------
    path : PathLike (str | UPath)
        Path with protocol "s3" pointing to a file in S3
    cache_dir : str | Path
        Cache directory

    Returns
    -------
    str
        Local filesystem path to the downloaded file
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return _download_s3_path(path, cache_dir, recursive=False)


def download_s3_folder(path: PathLike, cache_dir: str | Path) -> str:
    """Download an S3 folder to local cache and return the local path.

    Parameters
    ----------
    path : PathLike (str | UPath)
        Path with protocol "s3" pointing to a folder in S3
    cache_dir : str | Path
        Cache directory

    Returns
    -------
    str
        Local filesystem path to the downloaded folder
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return _download_s3_path(path, cache_dir, recursive=True)


def upload_s3_file(source: str | Path, destination: str | UPath) -> None:
    """Upload a local file to S3.

    Parameters
    ----------
    source : str | Path
        Local file path to upload
    destination : str | UPath
        Target path with s3:// protocol (e.g., "s3://bucket/path/file.txt")
    """
    destination = resolve_path(destination)
    fs = filesystem()
    logger.info(f"Uploading {source} to {destination}")
    fs.put(str(source), destination.path)


def upload_s3_folder(source: str | Path, destination: str | UPath) -> None:
    """Upload a local folder to S3.

    Parameters
    ----------
    source : str | Path
        Local folder path to upload
    destination : str | UPath
        Target path with s3:// protocol (e.g., "s3://bucket/path/folder")
    """
    destination = resolve_path(destination)
    fs = filesystem()
    logger.info(f"Uploading {source} to {destination}")
    fs.put(str(source), destination.path, recursive=True)
