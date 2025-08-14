from huggingface_hub import HfFileSystem, HfApi, RepoUrl
from fsspec import AbstractFileSystem
from dataclasses import dataclass
from typing import Literal

from src import HF_ENTITY

RepoType = Literal["space", "dataset", "model"]
"""Hugging Face repository type; see https://huggingface.co/docs/huggingface_hub/main/en/guides/hf_file_system#integrations"""


@dataclass
class HfRepo:
    """Represents a Hugging Face Hub repository with convenient path and URL generation.

    This class provides a structured way to work with Hugging Face repositories,
    generating proper paths and fsspec-compatible URLs for use with HfFileSystem.

    Attributes
    ----------
    entity : str
        The Hugging Face organization or username
    name : str
        The repository name
    type : RepoType
        Repository type ("dataset", "model", or "space")
    internal : bool
        Whether to use internal naming convention (adds __ prefix/suffix, e.g. "__my-dataset__")
    """

    entity: str
    name: str
    type: RepoType
    internal: bool

    def to_repo_id(self) -> str:
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
        >>> repo.to_repo_id()
        'my-org/dataset'
        >>> repo_internal = HfRepo(
        ...     entity="my-org", name="dataset", type="dataset", internal=True
        ... )
        >>> repo_internal.to_repo_id()
        'my-org/__dataset__'
        """
        name = f"__{self.name}__" if self.internal else self.name
        return f"{self.entity}/{name}"

    def path(self, *path: str) -> str:
        """Generate a repository path with optional subdirectories.

        Parameters
        ----------
        *path : str
            Optional path components within the repository

        Returns
        -------
        str
            A formatted path string for the repository

        Examples
        --------
        >>> repo = HfRepo(
        ...     entity="my-org", name="dataset", type="dataset", internal=True
        ... )
        >>> repo.path()
        'my-org/__dataset__'
        >>> repo.path("data", "train.csv")
        'my-org/__dataset__/data/train.csv'
        """
        name = f"__{self.name}__" if self.internal else self.name
        # Repos of type "model" require no path prefix; dataset and space do
        # Map singular types to plural path prefixes
        type_to_prefix = {"dataset": "datasets", "space": "spaces"}
        prefix = [type_to_prefix[self.type]] if self.type in type_to_prefix else []
        parts = prefix + [self.entity, name] + list(path)
        return "/".join(parts)

    def url(self, *path: str) -> str:
        """Generate an fsspec-compatible URL for use with HfFileSystem.

        See https://huggingface.co/docs/huggingface_hub/main/en/guides/hf_file_system#integrations
        for details on URL formatting.

        Parameters
        ----------
        *path : str
            Optional path components within the repository

        Returns
        -------
        str
            A formatted URL string following the hf:// scheme

        Examples
        --------
        >>> repo = HfRepo(entity="my-org", name="model", type="model", internal=False)
        >>> repo.url()
        'hf://my-org/model'
        >>> repo.url("config.json")
        'hf://my-org/model/config.json'
        >>> # Use with pandas
        >>> import pandas as pd
        >>> df = pd.read_csv(repo.url("data.csv"))
        """
        return f"hf://{self.path(*path)}"

    @staticmethod
    def from_repo_id(repo_id: str, *, type: RepoType = "dataset") -> "HfRepo":
        """Create an HfRepo instance from a repository ID string.

        Parameters
        ----------
        repo_id : str
            Repository ID in the format "entity/name" (e.g., "my-org/my-dataset")
        type : RepoType, optional
            Repository type - "dataset", "model", or "space", by default "dataset"

        Returns
        -------
        HfRepo
            An HfRepo instance with internal=False

        Raises
        ------
        ValueError
            If repo_id doesn't contain exactly one slash

        Examples
        --------
        >>> repo = HfRepo.from_repo_id("my-org/my-dataset")
        >>> repo.entity
        'my-org'
        >>> repo.name
        'my-dataset'
        >>> repo.type
        'dataset'
        >>> repo.internal
        False

        >>> model_repo = HfRepo.from_repo_id(
        ...     "huggingface/CodeBERTa-small-v1", type="model"
        ... )
        >>> model_repo.url()
        'hf://huggingface/CodeBERTa-small-v1'
        """
        parts = repo_id.split("/")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise ValueError(
                f"Invalid repository ID: '{repo_id}'. Expected format: 'entity/name'"
            )

        entity, name = parts
        return HfRepo(entity=entity, name=name, type=type, internal=False)


def hf_internal_repo(
    name: str, entity: str = HF_ENTITY, type: RepoType = "dataset"
) -> HfRepo:
    """Create an HfRepo instance with internal naming convention.

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
    HfRepo
        An HfRepo instance with internal=True

    Examples
    --------
    >>> # Create an internal dataset repository
    >>> repo = hf_internal_repo("my-data")
    >>> repo.path()
    'datasets/my-org/__my-data__'
    >>> repo.url("train.csv")
    'hf://datasets/my-org/__my-data__/train.csv'

    >>> # Create an internal model repository
    >>> model_repo = hf_internal_repo("my-model", type="model")
    >>> model_repo.path("config.json")
    'my-org/__my-model__/config.json'

    >>> # Use with custom entity
    >>> repo = hf_internal_repo("dataset", entity="custom-org")
    >>> repo.url()
    'hf://datasets/custom-org/__dataset__'
    """
    return HfRepo(entity, name, type, internal=True)


def hf_repo(
    name: str,
    entity: str = HF_ENTITY,
    type: RepoType = "dataset",
    internal: bool = False,
) -> HfRepo:
    """Create an HfRepo instance for working with Hugging Face repositories.

    This factory function simplifies creation of HfRepo objects, which generate
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
        Use internal naming convention with __ prefix/suffix, by default False

    Returns
    -------
    HfRepo
        An HfRepo instance configured for the specified repository

    Examples
    --------
    >>> # Create a dataset repository
    >>> repo = hf_repo("my-dataset")
    >>> repo.url("train.csv")
    'hf://my-org/my-dataset/train.csv'

    >>> # Read data with pandas
    >>> import pandas as pd
    >>> df = pd.read_csv(repo.url("data.csv"))

    >>> # List files with HfFileSystem
    >>> from huggingface_hub import HfFileSystem
    >>> fs = HfFileSystem()
    >>> fs.ls(repo.path("data"))
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
        repo_id=repo.to_repo_id(), repo_type=repo.type, private=private, **kwargs
    )
