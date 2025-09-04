import logging
import time
import tempfile
from pathlib import Path
from huggingface_hub import HfFileSystem, HfApi, RepoUrl
from huggingface_hub.constants import (
    REPO_TYPE_MODEL,
)
from huggingface_hub.constants import (
    REPO_TYPES_MAPPING,
)  # e.g. {"datasets": "dataset", "spaces": "space", "models": "model"}
from fsspec import AbstractFileSystem
from dataclasses import dataclass
from typing import Any, ContextManager, Literal, Callable
from contextlib import contextmanager
from upath import UPath
import ray

from src import HF_ENTITY

# Inverted mapping of repo type path components to repo type;
# e.g. {"dataset": "datasets", "space": "spaces", "model": "models"}
REPO_PATHS_MAPPING = {v: k for k, v in REPO_TYPES_MAPPING.items()}

RepoType = Literal["space", "dataset", "model"]
""" Hugging Face repository type; see:
- https://huggingface.co/docs/huggingface_hub/main/en/guides/hf_file_system#integrations
- https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/constants.py#L104-L106
"""

INTERNAL_PREFIX = "_dev_"


logger = logging.getLogger("ray")


def open_file(path: UPath, *args: Any, **kwargs: Any) -> ContextManager[Any]:
    """Open a file using UPath filesystem interface.

    TODO: implement retry logic, which is likely inevitable with fsspec
          and why this function exists rather than using path.open() directly

    Parameters
    ----------
    path
        UPath object representing the file to open
    *args
        Additional positional arguments passed to fs.open()
    **kwargs
        Additional keyword arguments passed to fs.open()

    Returns
    -------
    File-like object that supports context manager protocol
    """
    return path.open(*args, **kwargs)


def initialize_path(
    path: str, max_attempts: int = 30, poll_interval: float = 2.0
) -> None:
    """Initialize a path by creating the associated repository for a Hugging Face URL.

    This function checks if the provided path is a Hugging Face repository URL
    (using the "hf://" protocol) and creates the repository on the Hub if it doesn't
    already exist. After creation, it polls the repository URL to verify existence
    before returning. For non-HF paths, this function has no effect.

    Parameters
    ----------
    path : str
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
    >>> initialize_path("hf://datasets/my-org/new-dataset")

    >>> # No effect for local paths
    >>> initialize_path("/local/path/to/data")

    >>> # Initialize with custom polling parameters
    >>> initialize_path(
    ...     "hf://datasets/my-org/new-dataset", max_attempts=5, poll_interval=1.0
    ... )
    """
    path = UPath(path)
    if path.protocol != "hf":
        return
    if path.exists():
        return

    hf_path = HfPath.from_url(path)
    logger.info(
        f"Creating repository {hf_path.repo_id()} on Hugging Face for path {path}"
    )
    create_on_hub(hf_repo, exist_ok=True)

    # Poll for repository existence after creation
    for attempt in range(1, max_attempts + 1):
        # It is essential to recreate the UPath object to avoid caching of existence status
        repo_path = UPath(hf_repo.to_url())
        logger.info(
            f"Polling for repository ({repo_path}) existence (attempt {attempt}/{max_attempts})"
        )

        # It is also essential to clear the fsspec cache between checks;
        repo_path.fs.clear_instance_cache()

        if repo_path.exists():
            logger.info(f"Repository {hf_repo.repo_id()} successfully created")
            return
        if attempt < max_attempts:
            time.sleep(poll_interval)

    raise RuntimeError(
        f"Repository {hf_repo.repo_id()} could not be verified to exist after "
        f"{max_attempts} attempts. Repository may still be propagating."
    )


@dataclass
class HfPath:
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

    entity: str
    name: str
    type: RepoType
    internal: bool
    path_in_repo: str | None = None

    def repo_id(self) -> str:
        """Generate the repository ID for this HfPath.

        Returns
        -------
        str
            Repository ID in the format "entity/name" with internal naming applied if needed

        Examples
        --------
        >>> repo = HfPath(
        ...     entity="my-org", name="dataset", type="dataset", internal=False
        ... )
        >>> repo.repo_id()
        'my-org/dataset'
        >>> repo_internal = HfPath(
        ...     entity="my-org", name="dataset", type="dataset", internal=True
        ... )
        >>> repo_internal.repo_id()
        'my-org/_dev_dataset'
        """
        name = f"{INTERNAL_PREFIX}{self.name}" if self.internal else self.name
        return f"{self.entity}/{name}"

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


def hf_internal_repo(
    name: str, entity: str = HF_ENTITY, type: RepoType = "dataset"
) -> HfPath:
    """Create an HfPath instance with internal naming convention.

    Parameters
    ----------
    name : str
        Repository name
    entity : str, optional
        Hugging Face organization or username, by default HF_ENTITY
    type : RepoType, optional
        Repository type - "dataset", "model", or "space", by default "dataset"

    Returns
    -------
    HfPath
        An HfPath instance with internal=True

    Examples
    --------
    >>> # Create an internal dataset repository
    >>> repo = hf_internal_repo("my-data")
    >>> repo.to_string()
    'datasets/my-org/_dev_my-data'
    >>> path = repo.join("train.csv")
    >>> path.to_url()
    'hf://datasets/my-org/_dev_my-data/train.csv'

    >>> # Create an internal model repository
    >>> model_repo = hf_internal_repo("my-model", type="model")
    >>> path = model_repo.join("config.json")
    >>> path.to_string()
    'my-org/_dev_my-model/config.json'

    >>> # Use with custom entity
    >>> repo = hf_internal_repo("dataset", entity="custom-org")
    >>> repo.to_url()
    'hf://datasets/custom-org/_dev_dataset'
    """
    return HfPath(entity, name, type, internal=True, path_in_repo=None)


def hf_repo(
    name: str,
    entity: str = HF_ENTITY,
    type: RepoType = "dataset",
    internal: bool = False,
) -> HfPath:
    """Create an HfPath instance for working with Hugging Face repositories.

    This factory function simplifies creation of HfPath objects, which generate
    fsspec-compatible URLs for use with HfFileSystem and pandas, duckdb, zarr, etc.

    Parameters
    ----------
    name : str
        Repository name
    entity : str, optional
        Hugging Face organization or username, by default HF_ENTITY
    type : RepoType, optional
        Repository type - "dataset", "model", or "space", by default "dataset"
    internal : bool, optional
        Use internal naming convention with _dev_ prefix, by default False

    Returns
    -------
    HfPath
        An HfPath instance configured for the specified repository

    Examples
    --------
    >>> # Create a dataset repository
    >>> repo = hf_repo("my-dataset")
    >>> path = repo.join("train.csv")
    >>> path.to_url()
    'hf://datasets/my-org/my-dataset/train.csv'

    >>> # Read data with pandas
    >>> import pandas as pd
    >>> path = repo.join("data.csv")
    >>> df = pd.read_csv(path.to_url())

    >>> # List files with HfFileSystem
    >>> from huggingface_hub import HfFileSystem
    >>> fs = HfFileSystem()
    >>> path = repo.join("data")
    >>> fs.ls(path.to_string())
    """
    return HfPath(entity, name, type, internal, path_in_repo=None)


def filesystem() -> AbstractFileSystem:
    """Create an HfFileSystem instance for working with Hugging Face repositories.

    Returns
    -------
    AbstractFileSystem
        An HfFileSystem instance that can be used with fsspec-compatible operations
    """
    return HfFileSystem()


def create_on_hub(repo: HfPath, *, private: bool | None = None, **kwargs) -> RepoUrl:
    """Create a new repository on the Hugging Face Hub.

    This function wraps the HuggingFace Hub API's create_repo method to provide
    a convenient way to create repositories programmatically using HfPath instances.

    See https://huggingface.co/docs/huggingface_hub/v0.34.4/en/package_reference/hf_api#huggingface_hub.HfApi.create_repo

    Parameters
    ----------
    repo : HfPath
        An HfPath instance specifying the repository to create
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


def upload(path: UPath, writer: Callable[[Path], None], **kwargs) -> str:
    """Write data to a Hugging Face repository using a custom writer function.

    This function creates a temporary directory, calls the writer function to
    write data to that directory, then uploads the entire directory to the Hub.

    Parameters
    ----------
    path : UPath
        Target path with hf:// protocol (e.g., "hf://datasets/org/repo/data.zarr")
    writer : Callable[[Path], None]
        Function that takes a local path and writes data to it
    **kwargs
        Additional arguments passed to HfApi.upload_folder()

    Returns
    -------
    str
        The commit hash of the upload

    Examples
    --------
    >>> from upath import UPath
    >>> from pathlib import Path
    >>>
    >>> def write_csv(local_path: Path):
    ...     import pandas as pd
    ...
    ...     df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    ...     df.to_csv(local_path, index=False)
    >>>
    >>> commit_hash = upload(UPath("hf://datasets/my-org/data/file.csv"), write_csv)
    """
    if path.protocol != "hf":
        raise ValueError(f"Only hf:// protocol is supported, got: {path.protocol}")

    # Parse repo info and path from URL
    parsed_path = HfPath.from_url(str(path))
    path_in_repo = parsed_path.path_in_repo

    # Write to temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        writer(temp_path)

        # Upload to Hub
        api = HfApi()
        return api.upload_folder(
            repo_id=parsed_path.repo_id(),
            repo_type=parsed_path.type,
            folder_path=temp_dir,
            path_in_repo=path_in_repo,
            **kwargs,
        )


@contextmanager
def hf_repo_lock(path: UPath, lock, timeout_sec: int = 60):
    """Context manager for uploading to HF repositories with distributed locking.

    This context manager handles the acquisition and release of a distributed lock
    for safe concurrent uploads to Hugging Face repositories. It yields the path
    for use within the context.

    Parameters
    ----------
    path : UPath
        Target path with hf:// protocol for the upload
    lock
        Ray actor implementing acquire/release methods for distributed locking
    timeout_sec : int, optional
        Timeout in seconds for lock acquisition, by default 60

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
    >>> from upath import UPath
    >>>
    >>> # Assuming 'lock' is a Ray actor with acquire/release methods
    >>> logits_path = UPath("hf://datasets/org/repo/logits.zarr")
    >>> with hf_repo_lock(logits_path, lock) as path:
    ...     upload(path, lambda temp_path: dataset.to_zarr(temp_path))
    ...
    """

    logger.info(f"Saving data to {path}")
    repo = HfPath.from_url(str(path))
    token = f"{repo.type}/{repo.entity}/{repo.name}"

    logger.info(f"Acquiring lock for {token=}")
    acquired = ray.get(lock.acquire.remote(token, timeout_sec=timeout_sec))
    if not acquired:
        raise ValueError(
            f"Failed to acquire lock for {token} after {timeout_sec} seconds"
        )

    try:
        logger.info(f"Lock acquired for {token}")
        yield path
        logger.info(f"Successfully completed operation for {path}")
    finally:
        logger.info(f"Releasing lock for {token=}")
        ray.get(lock.release.remote(token))
