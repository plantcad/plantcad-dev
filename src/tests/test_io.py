import pytest
from src.io import HfPath, hf_repo, hf_internal_repo


class TestHfPath:
    """Test cases for HfPath class."""

    @pytest.mark.parametrize(
        "type_,internal,expected",
        [
            ("dataset", False, "datasets/org/repo"),
            ("model", False, "org/repo"),
            ("dataset", True, "datasets/org/_dev_repo"),
        ],
    )
    def test_path(self, type_, internal, expected):
        """Test HfPath.to_string() for different repo types and internal naming."""
        repo = HfPath(
            entity="org", name="repo", type=type_, internal=internal, path_in_repo=None
        )
        assert repo.to_string() == expected

    def test_path_with_subpath(self):
        """Test HfPath.to_string() with path_in_repo."""
        repo = HfPath(
            entity="org",
            name="data",
            type="dataset",
            internal=False,
            path_in_repo="train/file.csv",
        )
        assert repo.to_string() == "datasets/org/data/train/file.csv"

    @pytest.mark.parametrize(
        "type_,internal,expected_base",
        [
            ("dataset", False, "hf://datasets/org/repo"),
            ("model", False, "hf://org/repo"),
            ("space", False, "hf://spaces/org/repo"),
            ("dataset", True, "hf://datasets/org/_dev_repo"),
            ("model", True, "hf://org/_dev_repo"),
        ],
    )
    def test_to_uri(self, type_, internal, expected_base):
        """Test HfPath.to_url() generates correct hf:// URLs for different repo types and internal settings."""
        repo = HfPath(
            entity="org", name="repo", type=type_, internal=internal, path_in_repo=None
        )
        assert repo.to_url() == expected_base
        repo_with_file = HfPath(
            entity="org",
            name="repo",
            type=type_,
            internal=internal,
            path_in_repo="config.json",
        )
        assert repo_with_file.to_url() == f"{expected_base}/config.json"

    def test_from_url_simple(self):
        """Test HfPath.from_url() with a simple model URL."""
        repo = HfPath.from_url("hf://openai/gpt-3.5-turbo")
        assert repo.entity == "openai"
        assert repo.name == "gpt-3.5-turbo"
        assert repo.type == "model"
        assert repo.internal is False
        assert repo.path_in_repo is None
        assert repo.to_url() == "hf://openai/gpt-3.5-turbo"

    @pytest.mark.parametrize(
        "entity,name,internal,expected",
        [
            ("my-org", "dataset", False, "my-org/dataset"),
            ("my-org", "dataset", True, "my-org/_dev_dataset"),
        ],
    )
    def test_repo_id(self, entity, name, internal, expected):
        """Test HfPath.repo_id() generates correct repository IDs."""
        repo = HfPath(entity=entity, name=name, type="dataset", internal=internal)
        assert repo.repo_id() == expected

    @pytest.mark.parametrize(
        "url,expected_entity,expected_name,expected_type,expected_internal",
        [
            (
                "hf://microsoft/DialoGPT-medium",
                "microsoft",
                "DialoGPT-medium",
                "model",
                False,
            ),
            (
                "hf://datasets/my-org/_dev_internal-dataset",
                "my-org",
                "internal-dataset",
                "dataset",
                True,
            ),
            (
                "hf://spaces/gradio/calculator",
                "gradio",
                "calculator",
                "space",
                False,
            ),
            (
                "hf://datasets/org/_dev_test-repo",
                "org",
                "test-repo",
                "dataset",
                True,
            ),
            (
                "hf://spaces/user/_dev_space-name",
                "user",
                "space-name",
                "space",
                True,
            ),
        ],
    )
    def test_from_url_valid_urls(
        self, url, expected_entity, expected_name, expected_type, expected_internal
    ):
        """Test HfPath.from_url() with valid repository URLs including internal repos."""
        repo = HfPath.from_url(url)
        assert repo.entity == expected_entity
        assert repo.name == expected_name
        assert repo.type == expected_type
        assert repo.internal == expected_internal

    @pytest.mark.parametrize(
        "invalid_url",
        [
            "hf://",  # Missing repo path
            "not-a-url",  # Invalid URL format
            "",  # Empty string
            "hf://single-part",  # Missing entity/name format
        ],
    )
    def test_from_url_invalid_urls(self, invalid_url):
        """Test HfPath.from_url() raises ValueError for invalid URLs."""
        with pytest.raises(ValueError, match="Failed to parse repository URL"):
            HfPath.from_url(invalid_url)

    def test_from_url_without_protocol(self):
        """Test HfPath.from_url() accepts URLs without protocol."""
        # Test URL without protocol
        no_protocol = HfPath.from_url("datasets/org/repo")
        assert no_protocol.entity == "org"
        assert no_protocol.name == "repo"
        assert no_protocol.type == "dataset"
        assert no_protocol.path_in_repo is None

    def test_from_url_creates_correct_paths(self):
        """Test that from_url creates HfPath instances that generate correct paths and URIs."""
        # Test model repository
        model_repo = HfPath.from_url("hf://microsoft/DialoGPT-medium")
        assert model_repo.to_string() == "microsoft/DialoGPT-medium"
        assert model_repo.to_url() == "hf://microsoft/DialoGPT-medium"

        # Test internal dataset repository
        internal_repo = HfPath.from_url("hf://datasets/my-org/_dev_test-data")
        assert internal_repo.name == "test-data"
        assert internal_repo.internal is True
        assert internal_repo.to_string() == "datasets/my-org/_dev_test-data"
        assert internal_repo.to_url() == "hf://datasets/my-org/_dev_test-data"

        # Test space repository
        space_repo = HfPath.from_url("hf://spaces/gradio/hello-world")
        assert space_repo.to_string() == "spaces/gradio/hello-world"
        assert space_repo.to_url() == "hf://spaces/gradio/hello-world"

        # Test URL with file path
        file_repo = HfPath.from_url("hf://datasets/my-org/dataset/data/train.csv")
        assert file_repo.entity == "my-org"
        assert file_repo.name == "dataset"
        assert file_repo.path_in_repo == "data/train.csv"
        assert file_repo.to_string() == "datasets/my-org/dataset/data/train.csv"
        assert file_repo.to_url() == "hf://datasets/my-org/dataset/data/train.csv"


class TestFactoryFunctions:
    """Test cases for factory functions."""

    def test_hf_repo(self):
        """Test hf_repo factory function."""
        repo = hf_repo("test")
        assert repo.internal is False
        assert repo.type == "dataset"

    def test_hf_internal_repo(self):
        """Test hf_internal_repo factory function."""
        repo = hf_internal_repo("test")
        assert repo.internal is True
        assert repo.type == "dataset"

    def test_path_in_repo_functionality(self):
        """Test HfPath functionality with path_in_repo field."""
        # Test creating HfPath with path_in_repo
        file_path = HfPath(
            entity="my-org",
            name="dataset",
            type="dataset",
            internal=False,
            path_in_repo="data/train.csv",
        )
        assert file_path.path_in_repo == "data/train.csv"
        assert file_path.to_string() == "datasets/my-org/dataset/data/train.csv"
        assert file_path.to_url() == "hf://datasets/my-org/dataset/data/train.csv"

        # Test repo without path_in_repo
        repo_only = HfPath(
            entity="my-org",
            name="dataset",
            type="dataset",
            internal=False,
            path_in_repo=None,
        )
        assert repo_only.path_in_repo is None
        assert repo_only.to_string() == "datasets/my-org/dataset"
        assert repo_only.to_url() == "hf://datasets/my-org/dataset"

    def test_to_upath(self):
        """Test HfPath.to_upath() method functionality."""
        # Test basic repo
        repo = HfPath(
            entity="my-org",
            name="dataset",
            type="dataset",
            internal=False,
            path_in_repo=None,
        )
        upath = repo.to_upath()

        # Check type and properties
        from upath import UPath

        assert isinstance(upath, UPath)
        assert upath.protocol == "hf"
        assert str(upath) == "hf://datasets/my-org/dataset"

        # Test with path_in_repo
        path = repo.join("data", "train.csv")
        upath = path.to_upath()
        assert str(upath) == "hf://datasets/my-org/dataset/data/train.csv"

        # Verify consistency with to_url
        assert path.to_url() == upath.as_uri()

    def test_join_method(self):
        """Test HfPath.join() method functionality."""
        # Test basic join
        repo = HfPath(
            entity="my-org",
            name="dataset",
            type="dataset",
            internal=False,
            path_in_repo=None,
        )

        # Join single file
        file_path = repo.join("train.csv")
        assert file_path.path_in_repo == "train.csv"
        assert file_path.to_url() == "hf://datasets/my-org/dataset/train.csv"
        assert file_path.entity == repo.entity
        assert file_path.name == repo.name
        assert file_path.type == repo.type
        assert file_path.internal == repo.internal

        # Join multiple path components
        nested_file = repo.join("data", "subfolder", "file.txt")
        assert nested_file.path_in_repo == "data/subfolder/file.txt"
        assert (
            nested_file.to_url()
            == "hf://datasets/my-org/dataset/data/subfolder/file.txt"
        )

        # Join overwrites existing path_in_repo (does not append)
        base_path = repo.join("data")
        final_path = base_path.join("train.csv")
        assert (
            final_path.path_in_repo == "train.csv"
        )  # overwrites "data", doesn't append
        assert final_path.to_url() == "hf://datasets/my-org/dataset/train.csv"

        # Test with internal repo
        internal_repo = HfPath(
            entity="my-org",
            name="dataset",
            type="dataset",
            internal=True,
            path_in_repo=None,
        )
        internal_file = internal_repo.join("config.json")
        assert internal_file.path_in_repo == "config.json"
        assert internal_file.to_url() == "hf://datasets/my-org/_dev_dataset/config.json"

        # Test empty join (should return same path_in_repo)
        empty_join = repo.join()
        assert empty_join.path_in_repo is None
        assert empty_join.to_url() == repo.to_url()

        # Test explicit overwrite behavior
        repo_with_path = HfPath(
            entity="my-org",
            name="dataset",
            type="dataset",
            internal=False,
            path_in_repo="existing/path/file.txt",
        )
        overwritten = repo_with_path.join("new", "path.csv")
        assert (
            overwritten.path_in_repo == "new/path.csv"
        )  # completely overwrites existing path
        assert overwritten.to_url() == "hf://datasets/my-org/dataset/new/path.csv"
