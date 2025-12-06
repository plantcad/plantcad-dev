import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Literal, TypeVar
from upath import UPath
import logging
import pandas as pd
import xarray as xr

from src.io.fs import (
    download_file,
    download_folder,
    resolve_path,
    upload_file,
    upload_folder,
)
from src.io.hf import initialize_hf_path

T = TypeVar("T")

logger = logging.getLogger("ray")


def resolve_cache_dir(cache_dir: str | Path | None) -> Path:
    """Resolve cache directory to a Path.

    Parameters
    ----------
    cache_dir : str | Path | None
        Cache directory path or None

    Returns
    -------
    Path
        Resolved cache directory
    """
    if cache_dir is not None:
        return Path(cache_dir)

    # Check for PIPELINE_CACHE_DIR environment variable
    env_cache_dir = os.getenv("PIPELINE_CACHE_DIR")
    if env_cache_dir:
        return Path(env_cache_dir)

    # Return default cache directory
    default_cache_dir = Path.home() / ".cache" / "pipeline"
    default_cache_dir.mkdir(parents=True, exist_ok=True)
    return default_cache_dir


def initialize_path(path: str | UPath) -> bool:
    """Initialize a path by creating the associated directory or repository for it.

    Note that S3 bucket initialization is not supported; this function is a NOOP for S3 paths.

    Parameters
    ----------
    path : str | UPath
        The path to initialize.

    Returns
    -------
    bool
        True if the path was initialized, False otherwise if the path protocol
        does not support automatic initialization
    """
    upath = resolve_path(path)

    # Create directory or repository for local paths or HF paths; all others are a NOOP
    if upath.protocol == "file":
        upath.mkdir(parents=True, exist_ok=True)
        return True
    if upath.protocol == "hf":
        initialize_hf_path(upath)
        return True

    return False


def resolve_temp_dir(temp_dir: str | Path | None) -> Path:
    """Resolve temporary directory to a Path.

    Parameters
    ----------
    temp_dir : str | Path | None
        Temporary directory path or None

    Returns
    -------
    Path
        Resolved temporary directory
    """
    if temp_dir is not None:
        return Path(temp_dir)

    # Check for PIPELINE_TEMP_DIR environment variable
    env_temp_dir = os.getenv("PIPELINE_TEMP_DIR")
    if env_temp_dir:
        return Path(env_temp_dir)

    # Return default temporary directory
    default_temp_dir = Path(tempfile.gettempdir()) / "pipeline_temp"
    default_temp_dir.mkdir(parents=True, exist_ok=True)
    return default_temp_dir


def _hash_upath(upath: UPath) -> str:
    """Generate a hash string for a UPath."""
    return hashlib.md5(str(upath).encode()).hexdigest()


def _remove_path(path: Path) -> None:
    """Remove a file or directory."""
    if path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def resolve_local_path(
    upath: UPath,
    kind: Literal["file", "directory"],
    cache_dir: str | Path | None = None,
    force: bool = True,
) -> Path:
    """Resolve a UPath to a local path, downloading from remote storage if necessary.

    Parameters
    ----------
    upath : UPath
        Path to resolve
    kind : Literal["file", "directory"]
        Type of path being resolved
    cache_dir : str | Path | None
        Cache directory for remote downloads; defaults to PIPELINE_CACHE_DIR
        if set and ~/.cache/pipeline otherwise
    force : bool
        If True, always download even if local path exists.
        If False, skip download if local path already exists.

    Returns
    -------
    Path
        Local filesystem path
    """
    resolved_cache_dir = resolve_cache_dir(cache_dir)
    if kind == "directory":
        local_path = download_folder(upath, cache_dir=resolved_cache_dir, force=force)
    else:
        local_path = download_file(upath, cache_dir=resolved_cache_dir, force=force)
    return Path(local_path)


def _upload_local_path(
    local_path: Path, upath: UPath, kind: Literal["file", "directory"]
) -> None:
    """Upload local path to remote storage based on protocol."""
    if kind == "file":
        upload_file(local_path, upath)
    else:
        upload_folder(local_path, upath)


def read(
    path: str | UPath,
    reader: Callable[[Path], T],
    kind: Literal["file", "directory"],
    cache_dir: str | Path | None = None,
) -> T:
    """Read data from a path using backend-specific semantics.

    Parameters
    ----------
    path : str | UPath
        Path to read from
    reader : Callable[[Path], T]
        Function to read data from local path
    kind : Literal["file", "directory"]
        Type of path being read
    cache_dir : str | Path | None, optional
        Cache directory for remote downloads

    Returns
    -------
    T
        Result from reader function
    """
    upath = resolve_path(path)
    resolved_cache_dir = resolve_cache_dir(cache_dir)
    local_path = resolve_local_path(upath, kind, resolved_cache_dir)

    return reader(local_path)


def write(
    path: str | UPath,
    writer: Callable[[Path], T],
    kind: Literal["file", "directory"],
    temp_dir: str | Path | None = None,
    delete: bool = True,
) -> T:
    """Write data to a path using backend-specific semantics.

    Parameters
    ----------
    path : str | UPath
        Path to write to
    writer : Callable[[Path], T]
        Function to write data to local path
    kind : Literal["file", "directory"]
        Type of path being written
    temp_dir : str | Path | None, optional
        Temporary directory for local writes
    delete : bool, optional
        Whether to delete the local path after writing, by default True

    Returns
    -------
    T
        Result from writer function
    """
    upath = resolve_path(path)
    resolved_temp_dir = resolve_temp_dir(temp_dir)

    # Create local path for writes
    path_hash = _hash_upath(upath)
    local_path = resolved_temp_dir / path_hash

    # Clear existing path if necessary
    if local_path.exists():
        logger.info(f"Clearing existing temp path: {local_path}")
        _remove_path(local_path)

    local_path.mkdir(parents=True, exist_ok=False)

    # Handle file paths
    if kind == "file":
        if upath.name:
            local_path = local_path / upath.name
        else:
            raise ValueError(
                f"File path must have a name; got {upath.name=} for {upath}"
            )

    # Ensure parent directory exists
    local_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Call writer function
        result = writer(local_path)

        # Upload to remote storage
        _upload_local_path(local_path, upath, kind)

        return result
    finally:
        if delete:
            logger.info(f"Cleaning up temp path: {local_path}")
            _remove_path(local_path)


# -----------------------------------------------------------------------------
# Pandas I/O
# -----------------------------------------------------------------------------


def read_pandas_parquet(
    path: str | UPath, cache_dir: str | Path | None = None, **kwargs: Any
) -> pd.DataFrame:
    """Read a parquet file into a pandas DataFrame."""

    def reader(p: Path) -> pd.DataFrame:
        return pd.read_parquet(p, **kwargs)

    return read(path, reader, kind="file", cache_dir=cache_dir)


def write_pandas_parquet(
    df: pd.DataFrame,
    path: str | UPath,
    temp_dir: str | Path | None = None,
    delete: bool = True,
    **kwargs: Any,
) -> None:
    """Write a pandas DataFrame to a parquet file."""

    def writer(p: Path) -> None:
        df.to_parquet(p, **kwargs)

    write(path, writer, kind="file", temp_dir=temp_dir, delete=delete)


# -----------------------------------------------------------------------------
# Xarray I/O
# -----------------------------------------------------------------------------


def read_xarray_netcdf(
    path: str | UPath, cache_dir: str | Path | None = None, **kwargs: Any
) -> xr.Dataset:
    """Read a NetCDF file into an xarray Dataset."""

    def reader(p: Path) -> xr.Dataset:
        return xr.open_dataset(p, **kwargs)

    return read(path, reader, kind="file", cache_dir=cache_dir)


def write_xarray_netcdf(
    ds: xr.Dataset,
    path: str | UPath,
    temp_dir: str | Path | None = None,
    delete: bool = True,
    **kwargs: Any,
) -> None:
    """Write an xarray Dataset to a NetCDF file."""

    def writer(p: Path) -> None:
        ds.to_netcdf(p, **kwargs)

    write(path, writer, kind="file", temp_dir=temp_dir, delete=delete)


def read_xarray_mfdataset(
    path: str | UPath,
    concat_dim: str | None,
    cache_dir: str | Path | None = None,
    combine: Literal["by_coords", "nested"] = "nested",
    glob: str = "*.nc",
    **kwargs: Any,
) -> xr.Dataset:
    """Read multiple NetCDF files from a directory into a single xarray Dataset."""

    def reader(p: Path) -> xr.Dataset:
        nc_files = list(p.glob(glob))
        if not nc_files:
            raise ValueError(f"No files found in {p} matching {glob!r}")
        return xr.open_mfdataset(
            nc_files, combine=combine, concat_dim=concat_dim, **kwargs
        )

    return read(path, reader, kind="directory", cache_dir=cache_dir)
