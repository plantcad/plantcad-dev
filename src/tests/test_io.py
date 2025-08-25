import pytest
from src.io import HfRepo, hf_repo, hf_internal_repo


class TestHfRepo:
    """Test cases for HfRepo class."""

    @pytest.mark.parametrize(
        "type_,internal,expected",
        [
            ("dataset", False, "datasets/org/repo"),
            ("model", False, "org/repo"),
            ("dataset", True, "datasets/org/_dev_repo"),
        ],
    )
    def test_path(self, type_, internal, expected):
        """Test HfRepo.path() for different repo types and internal naming."""
        repo = HfRepo(entity="org", name="repo", type=type_, internal=internal)
        assert repo.path() == expected

    def test_path_with_subpath(self):
        """Test HfRepo.path() with additional path components."""
        repo = HfRepo(entity="org", name="data", type="dataset", internal=False)
        assert repo.path("train", "file.csv") == "datasets/org/data/train/file.csv"

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
    def test_url(self, type_, internal, expected_base):
        """Test HfRepo.url() generates correct hf:// URLs for different repo types and internal settings."""
        repo = HfRepo(entity="org", name="repo", type=type_, internal=internal)
        assert repo.url() == expected_base
        assert repo.url("config.json") == f"{expected_base}/config.json"

    def test_from_repo_id_default_type(self):
        """Test HfRepo.from_repo_id() with default dataset type."""
        repo = HfRepo.from_repo_id("my-org/my-dataset")
        assert repo.entity == "my-org"
        assert repo.name == "my-dataset"
        assert repo.type == "dataset"
        assert repo.internal is False
        assert repo.path() == "datasets/my-org/my-dataset"
        assert repo.url() == "hf://datasets/my-org/my-dataset"

    @pytest.mark.parametrize(
        "repo_id,type_,expected_entity,expected_name,expected_url,expected_internal",
        [
            (
                "huggingface/bert-base-uncased",
                "model",
                "huggingface",
                "bert-base-uncased",
                "hf://huggingface/bert-base-uncased",
                False,
            ),
            (
                "org/_dev_internal-dataset",
                "dataset",
                "org",
                "internal-dataset",
                "hf://datasets/org/_dev_internal-dataset",
                True,
            ),
            (
                "gradio/hello-world",
                "space",
                "gradio",
                "hello-world",
                "hf://spaces/gradio/hello-world",
                False,
            ),
            (
                "my-org/_dev_test-model",
                "model",
                "my-org",
                "test-model",
                "hf://my-org/_dev_test-model",
                True,
            ),
        ],
    )
    def test_from_repo_id_with_type(
        self,
        repo_id,
        type_,
        expected_entity,
        expected_name,
        expected_url,
        expected_internal,
    ):
        """Test HfRepo.from_repo_id() with different repository types including internal repos."""
        repo = HfRepo.from_repo_id(repo_id, type=type_)
        assert repo.entity == expected_entity
        assert repo.name == expected_name
        assert repo.type == type_
        assert repo.internal == expected_internal
        assert repo.url() == expected_url

    def test_from_repo_id_strips_whitespace(self):
        """Leading and trailing whitespace in repo_id should be ignored."""
        repo = HfRepo.from_repo_id("  org / repo  ")
        assert repo.entity == "org"
        assert repo.name == "repo"
        assert repo.path() == "datasets/org/repo"

    @pytest.mark.parametrize(
        "invalid_repo_id",
        [
            "no-slash",
            "too/many/slashes",
            "org/repo/extra/path",
            "",
            "/",
            "org/",
            "/repo",
        ],
    )
    def test_from_repo_id_invalid_format(self, invalid_repo_id):
        """Test HfRepo.from_repo_id() raises ValueError for invalid formats."""
        with pytest.raises(
            ValueError, match="Invalid repository ID.*Expected format: 'entity/name'"
        ):
            HfRepo.from_repo_id(invalid_repo_id)

    def test_from_url_simple(self):
        """Test HfRepo.from_url() with a simple model URL."""
        repo = HfRepo.from_url("https://huggingface.co/openai/gpt-3.5-turbo")
        assert repo.entity == "openai"
        assert repo.name == "gpt-3.5-turbo"
        assert repo.type == "model"
        assert repo.internal is False
        assert repo.url() == "hf://openai/gpt-3.5-turbo"

    @pytest.mark.parametrize(
        "entity,name,internal,expected",
        [
            ("my-org", "dataset", False, "my-org/dataset"),
            ("my-org", "dataset", True, "my-org/_dev_dataset"),
        ],
    )
    def test_to_repo_id(self, entity, name, internal, expected):
        """Test HfRepo.to_repo_id() generates correct repository IDs."""
        repo = HfRepo(entity=entity, name=name, type="dataset", internal=internal)
        assert repo.to_repo_id() == expected

    @pytest.mark.parametrize(
        "url,expected_entity,expected_name,expected_type,expected_internal",
        [
            (
                "https://huggingface.co/microsoft/DialoGPT-medium",
                "microsoft",
                "DialoGPT-medium",
                "model",
                False,
            ),
            (
                "https://huggingface.co/datasets/my-org/_dev_internal-dataset",
                "my-org",
                "internal-dataset",
                "dataset",
                True,
            ),
            (
                "https://huggingface.co/spaces/gradio/calculator",
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
                "hf://spaces/user/space-name",
                "user",
                "space-name",
                "space",
                False,
            ),
        ],
    )
    def test_from_url_valid_urls(
        self, url, expected_entity, expected_name, expected_type, expected_internal
    ):
        """Test HfRepo.from_url() with valid repository URLs including internal repos."""
        repo = HfRepo.from_url(url)
        assert repo.entity == expected_entity
        assert repo.name == expected_name
        assert repo.type == expected_type
        assert repo.internal == expected_internal

    @pytest.mark.parametrize(
        "invalid_url",
        [
            "https://github.com/user/repo",  # Not a HuggingFace URL
            "https://huggingface.co/",  # Missing repo path
            "not-a-url",  # Invalid URL format
            "",  # Empty string
            "https://huggingface.co/single-part",  # Missing entity/name format
        ],
    )
    def test_from_url_invalid_urls(self, invalid_url):
        """Test HfRepo.from_url() raises ValueError for invalid URLs."""
        with pytest.raises(ValueError, match="Failed to parse repository URL"):
            HfRepo.from_url(invalid_url)

    def test_from_url_creates_correct_paths(self):
        """Test that from_url creates HfRepo instances that generate correct paths and URLs."""
        # Test model repository
        model_repo = HfRepo.from_url("https://huggingface.co/microsoft/DialoGPT-medium")
        assert model_repo.path() == "microsoft/DialoGPT-medium"
        assert model_repo.url() == "hf://microsoft/DialoGPT-medium"

        # Test internal dataset repository
        internal_repo = HfRepo.from_url(
            "https://huggingface.co/datasets/my-org/_dev_test-data"
        )
        assert internal_repo.name == "test-data"
        assert internal_repo.internal is True
        assert internal_repo.path() == "datasets/my-org/_dev_test-data"
        assert internal_repo.url() == "hf://datasets/my-org/_dev_test-data"

        # Test space repository
        space_repo = HfRepo.from_url("https://huggingface.co/spaces/gradio/hello-world")
        assert space_repo.path() == "spaces/gradio/hello-world"
        assert space_repo.url() == "hf://spaces/gradio/hello-world"


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
