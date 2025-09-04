import logging
import os
import time
from pathlib import Path
from huggingface_hub import HfFileSystem, HfApi, RepoUrl
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import (
    REPO_TYPE_MODEL,
    REPO_TYPE_DATASET,
    REPO_TYPE_SPACE,
)

# Import mapping of URL repo types to API repo types; e.g. at TOW:
# {"datasets": "dataset", "spaces": "space", "models": "model"}
from huggingface_hub.constants import (
    REPO_TYPES_MAPPING,
)
from fsspec import AbstractFileSystem
from dataclasses import dataclass
from enum import StrEnum
from contextlib import contextmanager
from src.utils.ray_utils import AsyncLock
from upath import UPath
import ray

from src import HF_ENTITY


# ------------------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------------------

logger = logging.getLogger("ray")

# Inverted mapping of repo type path components to repo type;
# e.g. {"dataset": "datasets", "space": "spaces", "model": "models"}
REPO_PATHS_MAPPING = {v: k for k, v in REPO_TYPES_MAPPING.items()}

PathLike = str | UPath


class RepoType(StrEnum):
    """Hugging Face repository type.

    See:
    - https://huggingface.co/docs/huggingface_hub/main/en/guides/hf_file_system#integrations
    - https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/constants.py#L104-L106
    """

    MODEL = REPO_TYPE_MODEL
    DATASET = REPO_TYPE_DATASET
    SPACE = REPO_TYPE_SPACE


INTERNAL_PREFIX = "_dev_"


# ------------------------------------------------------------------------------
# Path Utilities
# ------------------------------------------------------------------------------


def resolve_path(path: PathLike) -> UPath:
    """Convert a PathLike to UPath and validate hf:// protocol."""
    if not isinstance(path, UPath):
        path = UPath(path)
    if path.protocol != "hf":
        raise ValueError(f"Only hf:// protocol is supported, got: {path.protocol}")
    return path


@dataclass
class HfRepo:
    entity: str
    name: str
    type: RepoType
    internal: bool

    def repo_id(self) -> str:
        """Generate the repository ID for this HfRepo.

        Returns
        -------
        str
            Repository ID in the format "entity/name" with internal naming applied if needed

        Examples
        --------
        >>> repo = HfRepo(
        ...     entity="my-org", name="dataset", type="dataset", internal=False
        ... )
        >>> repo.repo_id()
        'my-org/dataset'
        >>> repo_internal = HfRepo(
        ...     entity="my-org", name="dataset", type="dataset", internal=True
        ... )
        >>> repo_internal.repo_id()
        'my-org/_dev_dataset'
        """
        name = f"{INTERNAL_PREFIX}{self.name}" if self.internal else self.name
        return f"{self.entity}/{name}"


@dataclass
class HfPath(HfRepo):
    """Represents a Hugging Face Hub path with convenient path and URL generation.

    This class provides a structured way to work with Hugging Face paths,
    generating paths and fsspec-compatible URLs for use with HfFileSystem. It is similar to
    `RepoUrl` from the HfApi but supports:

    - Naming conventions for "internal" vs "external" repositories
    - Paths within a repository
    - Parsing of repository IDs (e.g. "my-org/my-repo" -> {entity="my-org", name="my-repo"})

    For related implementations, see:

    - https://github.com/huggingface/huggingface_hub/blob/8a6d4e3e8d18c7ba4ecf8b2e9a23138e48257330/src/huggingface_hub/hf_api.py#L536
    - https://github.com/huggingface/huggingface_hub/blob/8a6d4e3e8d18c7ba4ecf8b2e9a23138e48257330/src/huggingface_hub/hf_file_system.py#L139

    Attributes
    ----------
    entity : str
        Hugging Face organization or username
    name : str
        Repository name
    type : RepoType
        Repository type ("dataset", "model", or "space")
    internal : bool
        Whether to use internal naming convention (adds _dev_ prefix, e.g. "_dev_my-dataset")
    path_in_repo : str | None
        Optional path within the repository (e.g., "data/train.csv")
    """

    path_in_repo: str | None = None

    def join(self, *path: str) -> "HfPath":
        """Create a new HfPath with the given path components joined to path_in_repo.

        This method creates a new HfPath instance with the same repository information
        but with the path_in_repo set to the joined path components.

        Parameters
        ----------
        *path : str
            Path components to join (e.g., "data", "train.csv")

        Returns
        -------
        HfPath
            A new HfPath instance with the joined path_in_repo

        Examples
        --------
        >>> repo = HfPath(
        ...     entity="my-org", name="dataset", type="dataset", internal=False
        ... )
        >>> path = repo.join("data", "train.csv")
        >>> path.path_in_repo
        'data/train.csv'
        >>> path.to_url()
        'hf://datasets/my-org/dataset/data/train.csv'

        >>> # Overwrites existing path_in_repo
        >>> base_path = HfPath(
        ...     entity="my-org",
        ...     name="dataset",
        ...     type="dataset",
        ...     internal=False,
        ...     path_in_repo="old/path",
        ... )
        >>> path = base_path.join("subfolder", "file.txt")
        >>> path.path_in_repo
        'subfolder/file.txt'
        """
        # Create new path_in_repo from the provided path components, overwriting any existing path_in_repo
        new_path_in_repo = "/".join(path) if path else None

        return HfPath(
            entity=self.entity,
            name=self.name,
            type=self.type,
            internal=self.internal,
            path_in_repo=new_path_in_repo,
        )

    def to_string(self) -> str:
        """Generate a repository path including any path_in_repo.

        Returns
        -------
        str
            A formatted path string for the repository and optional file path

        Examples
        --------
        >>> repo = HfPath(
        ...     entity="my-org", name="dataset", type="dataset", internal=True
        ... )
        >>> repo.to_string()
        'datasets/my-org/_dev_dataset'
        >>> repo_with_file = HfPath(
        ...     entity="my-org",
        ...     name="dataset",
        ...     type="dataset",
        ...     internal=True,
        ...     path_in_repo="data/train.csv",
        ... )
        >>> repo_with_file.to_string()
        'datasets/my-org/_dev_dataset/data/train.csv'
        """
        name = f"{INTERNAL_PREFIX}{self.name}" if self.internal else self.name
        # Construct path components noting that repos of type "model" require
        # no path prefix while dataset and space do ("datasets" and "spaces" @ TOW)
        type_to_prefix = {
            k: v for k, v in REPO_PATHS_MAPPING.items() if k != REPO_TYPE_MODEL
        }
        prefix = [type_to_prefix[self.type]] if self.type in type_to_prefix else []
        parts = prefix + [self.entity, name]
        if self.path_in_repo:
            parts.append(self.path_in_repo)
        return "/".join(parts)

    def to_url(self) -> str:
        """Generate an fsspec-compatible URL for use with HfFileSystem.

        See https://huggingface.co/docs/huggingface_hub/main/en/guides/hf_file_system#integrations
        for details on URL formatting.

        Returns
        -------
        str
            A formatted URL string following the hf:// scheme

        Examples
        --------
        >>> repo = HfPath(entity="my-org", name="model", type="model", internal=False)
        >>> repo.to_url()
        'hf://my-org/model'
        >>> repo_with_file = HfPath(
        ...     entity="my-org",
        ...     name="model",
        ...     type="model",
        ...     internal=False,
        ...     path_in_repo="config.json",
        ... )
        >>> repo_with_file.to_url()
        'hf://my-org/model/config.json'
        >>> # Use with pandas
        >>> import pandas as pd
        >>> path = HfPath(
        ...     entity="my-org",
        ...     name="dataset",
        ...     type="dataset",
        ...     internal=False,
        ...     path_in_repo="data.csv",
        ... )
        >>> df = pd.read_csv(path.to_url())
        """
        return UPath(self.to_string(), protocol="hf").as_uri()

    def to_upath(self) -> UPath:
        """Generate a UPath object for use with filesystem operations.

        Returns
        -------
        UPath
            A UPath object with hf:// protocol for direct filesystem operations

        Examples
        --------
        >>> repo = HfPath(
        ...     entity="my-org", name="dataset", type="dataset", internal=False
        ... )
        >>> upath = repo.to_upath()
        >>> upath.exists()  # Check if repository exists
        False
        >>> path = repo.join("data.csv")
        >>> upath = path.to_upath()
        >>> with upath.open("r") as f:
        ...     content = f.read()  # Read file content directly
        ...
        """
        return UPath(self.to_string(), protocol="hf")

    def to_repo(self) -> "HfRepo":
        """Convert this HfPath to an HfRepo by removing the path_in_repo component.

        Returns
        -------
        HfRepo
            An HfRepo instance with the same repository information but no path_in_repo

        Examples
        --------
        >>> path = HfPath(
        ...     entity="my-org",
        ...     name="dataset",
        ...     type="dataset",
        ...     internal=False,
        ...     path_in_repo="data/train.csv",
        ... )
        >>> repo = path.to_repo()
        >>> repo.entity
        'my-org'
        >>> repo.name
        'dataset'
        >>> hasattr(repo, "path_in_repo")
        False
        """
        return HfRepo(
            entity=self.entity, name=self.name, type=self.type, internal=self.internal
        )

    def split_path_in_repo(
        self,
    ) -> tuple[str, str] | tuple[None, str] | tuple[None, None]:
        """Split path_in_repo into subfolder and filename components.

        Returns
        -------
        tuple[str, str] | tuple[None, str] | tuple[None, None]
            Always returns a 2-tuple (subfolder, filename):
            - If path_in_repo is None: returns (None, None)
            - If path_in_repo has no directory separators: returns (None, filename)
            - If path_in_repo has directory separators: returns (subfolder, filename)

        Examples
        --------
        >>> path = HfPath("org", "repo", "dataset", False, "data/train.csv")
        >>> path.split_path_in_repo()
        ('data', 'train.csv')

        >>> path = HfPath("org", "repo", "dataset", False, "file.txt")
        >>> path.split_path_in_repo()
        (None, 'file.txt')

        >>> path = HfPath("org", "repo", "dataset", False, None)
        >>> path.split_path_in_repo()
        (None, None)
        """
        if not self.path_in_repo:
            return (None, None)

        subfolder, filename = os.path.split(self.path_in_repo)
        subfolder = subfolder if subfolder else None
        return (subfolder, filename)

    @staticmethod
    def from_url(url: str) -> "HfPath":
        """Create an HfPath instance from a repository URL.

        Parameters
        ----------
        url : str
            Repository URL (e.g., "hf://datasets/org/repo-name/file.csv")

        Returns
        -------
        HfPath
            An HfPath instance with path_in_repo set if the URL includes a file path

        Raises
        ------
        ValueError
            If the URL cannot be parsed

        Examples
        --------
        >>> repo = HfPath.from_url("hf://microsoft/DialoGPT-medium")
        HfPath(entity='microsoft', name='DialoGPT-medium', type='model', internal=False, path_in_repo=None)

        >>> HfPath.from_url("hf://datasets/huggingface/squad/train.json")
        HfPath(entity='huggingface', name='squad', type='dataset', internal=False, path_in_repo='train.json')

        >>> HfPath.from_url("hf://spaces/gradio/calculator/app.py")
        HfPath(entity='gradio', name='calculator', type='space', internal=False, path_in_repo='app.py')

        >>> HfPath.from_url("hf://plantcad/_dev_training_dataset")  # "internal" model
        HfPath(entity='plantcad', name='training_dataset', type='model', internal=True, path_in_repo=None)
        """
        try:
            # Strip protocol
            # Note: We could use fsspec.core.strip_protocol for this, but it only splits on "://" anyhow:
            # https://github.com/fsspec/filesystem_spec/blob/f84b99f0d1f079f990db1a219b74df66ab3e7160/fsspec/core.py#L551-L552
            path = url
            if "://" in url:
                path = url.split("://", 1)[1]

            # Split path components
            parts = path.split("/")

            if len(parts) < 2:
                raise ValueError(
                    f"URL path must have at least 2 components, got: {parts}"
                )

            # Parse repo type and extract entity/name/path_in_repo
            if parts[0] in REPO_TYPES_MAPPING:
                if len(parts) < 3:
                    raise ValueError(f"Not enough components for typed repo: {parts}")
                repo_type = REPO_TYPES_MAPPING[parts[0]]
                entity = parts[1]
                name = parts[2]
                path_in_repo_parts = parts[3:] if len(parts) > 3 else []
            else:
                repo_type = REPO_TYPE_MODEL
                entity = parts[0]
                name = parts[1]
                path_in_repo_parts = parts[2:] if len(parts) > 2 else []

            # Handle internal naming
            internal = name.startswith(INTERNAL_PREFIX)
            if internal:
                name = name.removeprefix(INTERNAL_PREFIX)

            # Construct path_in_repo
            path_in_repo = "/".join(path_in_repo_parts) if path_in_repo_parts else None

            return HfPath(
                entity=entity,
                name=name,
                type=repo_type,
                internal=internal,
                path_in_repo=path_in_repo,
            )
        except Exception as e:
            raise ValueError(f"Failed to parse repository URL '{url}': {e}") from e

    @staticmethod
    def from_upath(path: UPath) -> "HfPath":
        """Create an HfPath instance from a UPath object.

        Parameters
        ----------
        path : UPath
            A UPath object representing a repository path

        Returns
        -------
        HfPath
            An HfPath instance parsed from the UPath's URI

        Examples
        --------
        >>> from upath import UPath
        >>> upath = UPath("datasets/huggingface/squad/train.json", protocol="hf")
        >>> repo = HfPath.from_upath(upath)
        >>> repo.entity
        'huggingface'
        >>> repo.name
        'squad'
        >>> repo.path_in_repo
        'train.json'
        """
        return HfPath.from_url(path.as_uri())


def hf_repo(
    name: str,
    entity: str = HF_ENTITY,
    type: RepoType = "dataset",
    internal: bool = True,
) -> HfRepo:
    """Create an HfRepo instance for working with Hugging Face repositories.

    This factory function simplifies creation of HfRepo objects for repository management.

    Parameters
    ----------
    name : str
        Repository name
    entity : str, optional
        Hugging Face organization or username, by default HF_ENTITY
    type : RepoType, optional
        Repository type - "dataset", "model", or "space", by default "dataset"
    internal : bool, optional
        Use internal naming convention with _dev_ prefix, by default True

    Returns
    -------
    HfRepo
        An HfRepo instance configured for the specified repository

    Examples
    --------
    >>> # Create a dataset repository
    >>> repo = hf_repo("my-dataset")
    >>> repo.entity
    'my-org'
    >>> repo.name
    'my-dataset'
    >>> repo.type
    'dataset'
    """
    return HfRepo(entity, name, type, internal)


def filesystem() -> AbstractFileSystem:
    """Create an HfFileSystem instance for working with Hugging Face repositories.

    Returns
    -------
    AbstractFileSystem
        An HfFileSystem instance that can be used with fsspec-compatible operations
    """
    return HfFileSystem()


# ------------------------------------------------------------------------------
# Repository Management
# ------------------------------------------------------------------------------


def initialize_hf_path(
    path: PathLike, max_attempts: int = 30, poll_interval: float = 2.0
) -> None:
    """Initialize a path by creating the associated repository for a Hugging Face URL.

    This function checks if the provided path is a Hugging Face repository URL
    (using the "hf://" protocol) and creates the repository on the Hub if it doesn't
    already exist. After creation, it polls the repository URL to verify existence
    before returning. For non-HF paths, this function has no effect.

    Parameters
    ----------
    path : PathLike (str | UPath)
        The path to initialize. Can be a local path, remote URL, or Hugging Face
        repository URL (e.g., "hf://datasets/my-org/my-dataset").
    max_attempts : int, optional
        Maximum number of polling attempts to verify repository existence after creation,
        by default 30
    poll_interval : float, optional
        Time to wait between polling attempts in seconds, by default 2.0

    Raises
    ------
    RuntimeError
        If the repository cannot be verified to exist after max_attempts polling attempts

    Examples
    --------
    >>> # Initialize a Hugging Face dataset repository
    >>> initialize_hf_path("hf://datasets/my-org/new-dataset")

    >>> # No effect for local paths
    >>> initialize_hf_path("/local/path/to/data")

    >>> # Initialize with custom polling parameters
    >>> initialize_hf_path(
    ...     "hf://datasets/my-org/new-dataset", max_attempts=5, poll_interval=1.0
    ... )
    """
    path = resolve_path(path)
    if path.exists():
        return

    hf_path = HfPath.from_upath(path)
    logger.info(
        f"Creating repository {hf_path.repo_id()} on Hugging Face for path {path}"
    )
    create_on_hub(hf_path, exist_ok=True)

    # Poll for repository existence after creation
    for attempt in range(1, max_attempts + 1):
        # It is essential to recreate the UPath object to avoid caching of existence status
        repo_path = UPath(hf_path.to_url())
        logger.info(
            f"Polling for repository ({repo_path}) existence (attempt {attempt}/{max_attempts})"
        )

        # It is also essential to clear the fsspec cache between checks;
        repo_path.fs.clear_instance_cache()

        if repo_path.exists():
            logger.info(f"Repository {hf_path.repo_id()} successfully created")
            return
        if attempt < max_attempts:
            time.sleep(poll_interval)

    raise RuntimeError(
        f"Repository {hf_path.repo_id()} could not be verified to exist after "
        f"{max_attempts} attempts. Repository may still be propagating."
    )


def create_on_hub(repo: HfRepo, *, private: bool | None = None, **kwargs) -> RepoUrl:
    """Create a new repository on the Hugging Face Hub.

    This function wraps the HuggingFace Hub API's create_repo method to provide
    a convenient way to create repositories programmatically using HfRepo instances.

    See https://huggingface.co/docs/huggingface_hub/v0.34.4/en/package_reference/hf_api#huggingface_hub.HfApi.create_repo

    Parameters
    ----------
    repo : HfRepo
        An HfRepo instance specifying the repository to create
    private : bool, optional
        Whether to make the repository private. If None, the default visibility
        setting for the organization/user will be used.
    **kwargs
        Additional keyword arguments passed to HfApi.create_repo().
        Common options include:
        - token: Authentication token (if not using default)
        - exist_ok: If True, do not raise error if repo already exists
        - space_sdk: For spaces, the SDK to use ("gradio", "streamlit", etc.)

    Returns
    -------
    RepoUrl
        URL to the newly created repository

    Examples
    --------
    >>> # Create a public dataset repository
    >>> repo = hf_repo("test-dataset", type="dataset")
    >>> url = create_on_hub(repo, private=False)
    >>>
    >>> # Create a private model repository
    >>> repo = hf_repo("my-model", type="model")
    >>> url = create_on_hub(repo, private=True)
    >>>
    >>> # Create with additional options
    >>> repo = hf_repo("existing-repo", type="dataset")
    >>> url = create_on_hub(repo, exist_ok=True)
    """
    api = HfApi()
    return api.create_repo(
        repo_id=repo.repo_id(), repo_type=repo.type, private=private, **kwargs
    )


# ------------------------------------------------------------------------------
# File I/O
# ------------------------------------------------------------------------------


def download_hf_file(
    path: PathLike, force_download: bool = False, cache_dir: str | Path | None = None
) -> str:
    """Download a Hugging Face Hub file to local cache and return the local path.

    Parameters
    ----------
    path : PathLike (str | UPath)
        Path with protocol "hf" pointing to a file in a Hugging Face repository
    force_download : bool, optional
        Whether to force re-download even if cached, by default False
    cache_dir : str | Path | None, optional
        Custom cache directory, by default None (uses HF default)

    Returns
    -------
    str
        Local filesystem path to the downloaded/cached file

    Notes
    -----
    Cache location controlled by cache_dir parameter or HF_HUB_CACHE/HF_HOME environment variables.
    """
    # Parse path components
    path = resolve_path(path)
    hf_path = HfPath.from_upath(path)

    # Split path into subfolder and filename components
    subfolder, filename = hf_path.split_path_in_repo()

    # Download file
    local_path = hf_hub_download(
        repo_id=hf_path.repo_id(),
        repo_type=hf_path.type,
        subfolder=subfolder,
        filename=filename,
        force_download=force_download,
        cache_dir=cache_dir,
    )
    return local_path


def download_hf_folder(
    path: PathLike, force_download: bool = False, cache_dir: str | Path | None = None
) -> str:
    """Download a Hugging Face Hub folder to local cache and return the local path.

    Parameters
    ----------
    path : PathLike (str | UPath)
        Path with protocol "hf" pointing to a folder in a Hugging Face repository
    force_download : bool, optional
        Whether to force re-download even if cached, by default False
    cache_dir : str | Path | None, optional
        Custom cache directory, by default None (uses HF default)

    Returns
    -------
    str
        Local filesystem path to the downloaded/cached folder

    Notes
    -----
    Lists all files under the provided path and downloads each using download_hf_file.
    """
    # Parse path components
    path = resolve_path(path)
    folder_upath = UPath(path)

    # Build list of remote paths that exist under the provided path
    remote_paths = []
    for item in folder_upath.rglob("*"):
        if item.is_file():
            remote_paths.append(item)

    # Download each file and save all returned paths
    local_paths = []
    for remote_path in remote_paths:
        local_path = download_hf_file(
            remote_path, force_download=force_download, cache_dir=cache_dir
        )
        local_paths.append((remote_path, Path(local_path)))

    if not local_paths:
        raise ValueError(f"No files found in {folder_upath}")

    # Find the base path by removing the relative path from any downloaded file
    first_remote_path, first_local_path = local_paths[0]
    # e.g. hf://repo/dir1/dir2/file.txt relative to hf://repo/dir1 -> dir2/file.txt
    relative_path = first_remote_path.relative_to(folder_upath)
    # e.g. /cache/hash/dir2/file.txt go up 2 levels -> /cache/hash
    base_path = first_local_path.parents[len(relative_path.parts) - 1]

    return str(base_path)


def upload_hf_folder(source: str | Path, destination: str | UPath, **kwargs) -> str:
    """Upload a local folder to a Hugging Face repository.

    See:
    - https://huggingface.co/docs/huggingface_hub/v0.34.4/en/package_reference/hf_api#huggingface_hub.HfApi.upload_folder

    Parameters
    ----------
    source : str | Path
        Local folder path to upload
    destination : str | UPath
        Target path with hf:// protocol (e.g., "hf://datasets/org/repo/data.zarr")
    **kwargs
        Additional arguments passed to HfApi.upload_folder()

    Returns
    -------
    str
        The commit hash of the upload

    Examples
    --------
    >>> upload_hf_folder("/local/data/folder", "hf://datasets/my-org/data")
    """
    destination = resolve_path(destination)

    # Parse destination path components
    parsed_path = HfPath.from_upath(destination)
    path_in_repo = parsed_path.path_in_repo

    # Upload to Hub
    api = HfApi()
    return api.upload_folder(
        repo_id=parsed_path.repo_id(),
        repo_type=parsed_path.type,
        folder_path=source,
        path_in_repo=path_in_repo,
        **kwargs,
    )


def upload_hf_file(source: str | Path, destination: str | UPath, **kwargs) -> str:
    """Upload a local file to a Hugging Face repository.

    See:
    - https://huggingface.co/docs/huggingface_hub/v0.34.4/en/package_reference/hf_api#huggingface_hub.HfApi.upload_file

    Parameters
    ----------
    source : str | Path
        Local file path to upload
    destination : str | UPath
        Target path with hf:// protocol (e.g., "hf://datasets/org/repo/data.csv")
    **kwargs
        Additional arguments passed to HfApi.upload_file()

    Returns
    -------
    str
        The commit hash of the upload

    Examples
    --------
    >>> upload_hf_file("/local/data/file.csv", "hf://datasets/my-org/data/file.csv")
    """
    destination = resolve_path(destination)

    # Parse destination path components
    parsed_path = HfPath.from_upath(destination)
    path_in_repo = parsed_path.path_in_repo

    # Upload to Hub
    api = HfApi()
    return api.upload_file(
        repo_id=parsed_path.repo_id(),
        repo_type=parsed_path.type,
        path_or_fileobj=source,
        path_in_repo=path_in_repo,
        **kwargs,
    )


# ------------------------------------------------------------------------------
# Locking
# ------------------------------------------------------------------------------


def get_hf_lock() -> ray.actor.ActorHandle:
    """Get the Ray Actor for HF write locking, creating it if it doesn't exist.

    Returns
    -------
    ray.actor.ActorHandle
        The Ray actor handle for the HF write lock
    """
    try:
        # Check if the actor already exists;
        # see: https://docs.ray.io/en/latest/ray-core/api/doc/ray.get_actor.html
        lock = ray.get_actor("hf-write-lock")
        logger.info("HF write lock actor already running")
        return lock
    except ValueError:
        # Actor doesn't exist, create it
        lock = AsyncLock.options(name="hf-write-lock").remote()
        logger.info("Started HF write lock actor")
        return lock


@contextmanager
def lock_hf_path(
    path: PathLike, lock: ray.actor.ActorHandle, timeout_sec: int | None = None
):
    """Context manager for uploading to HF repositories with distributed locking.

    This context manager handles the acquisition and release of a distributed lock
    for safe concurrent uploads to Hugging Face repositories. It yields the path
    for use within the context.

    Parameters
    ----------
    path : PathLike (str | UPath)
        Target path with hf:// protocol for the upload
    lock
        Ray actor handle for HF write locking
    timeout_sec : int, optional
        Timeout in seconds for lock acquisition, by default None (no timeout)

    Yields
    ------
    UPath
        The same path passed in, for convenience in the context

    Raises
    ------
    ValueError
        If lock acquisition fails within the timeout period

    Examples
    --------
    >>> # Get the lock actor and use it
    >>> lock = get_hf_lock()
    >>> path = "hf://datasets/org/repo/logits.zarr"
    >>> with lock_hf_path(path, lock) as locked_path:
    ...     upload_hf_path(locked_path, lambda temp_path: dataset.to_zarr(temp_path))
    ...
    """
    path = resolve_path(path)
    logger.info(f"Saving data to {path}")
    repo = HfPath.from_upath(path)
    key = f"{repo.type}/{repo.entity}/{repo.name}"

    logger.info(f"Acquiring lock for {key=}")
    acquired = ray.get(lock.acquire.remote(key, timeout_sec=timeout_sec))
    if not acquired:
        raise ValueError(
            f"Failed to acquire lock for {key=} after {timeout_sec} seconds"
        )

    try:
        logger.info(f"Lock acquired for {key=}")
        yield path
        logger.info(f"Successfully completed operation for {path}")
    finally:
        logger.info(f"Releasing lock for {key=}")
        ray.get(lock.release.remote(key))
