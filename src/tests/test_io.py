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

    def test_from_url_simple(self):
        """Test HfRepo.from_url() with a simple model URL."""
        repo = HfRepo.from_url("hf://openai/gpt-3.5-turbo")
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
    def test_repo_id(self, entity, name, internal, expected):
        """Test HfRepo.repo_id() generates correct repository IDs."""
        repo = HfRepo(entity=entity, name=name, type="dataset", internal=internal)
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
        """Test HfRepo.from_url() with valid repository URLs including internal repos."""
        repo = HfRepo.from_url(url)
        assert repo.entity == expected_entity
        assert repo.name == expected_name
        assert repo.type == expected_type
        assert repo.internal == expected_internal

    @pytest.mark.parametrize(
        "invalid_url",
        [
            "https://github.com/user/repo",  # Not using hf:// protocol
            "hf://",  # Missing repo path
            "not-a-url",  # Invalid URL format
            "",  # Empty string
            "hf://single-part",  # Missing entity/name format
        ],
    )
    def test_from_url_invalid_urls(self, invalid_url):
        """Test HfRepo.from_url() raises ValueError for invalid URLs."""
        with pytest.raises(ValueError, match="Failed to parse repository URL"):
            HfRepo.from_url(invalid_url)

    def test_from_url_creates_correct_paths(self):
        """Test that from_url creates HfRepo instances that generate correct paths and URLs."""
        # Test model repository
        model_repo = HfRepo.from_url("hf://microsoft/DialoGPT-medium")
        assert model_repo.path() == "microsoft/DialoGPT-medium"
        assert model_repo.url() == "hf://microsoft/DialoGPT-medium"

        # Test internal dataset repository
        internal_repo = HfRepo.from_url("hf://datasets/my-org/_dev_test-data")
        assert internal_repo.name == "test-data"
        assert internal_repo.internal is True
        assert internal_repo.path() == "datasets/my-org/_dev_test-data"
        assert internal_repo.url() == "hf://datasets/my-org/_dev_test-data"

        # Test space repository
        space_repo = HfRepo.from_url("hf://spaces/gradio/hello-world")
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

    def test_parse_path(self):
        """Test HfRepo.parse_path() method."""
        # Test dataset repository
        repo = HfRepo(entity="my-org", name="dataset", type="dataset", internal=False)

        # Test basic file path
        assert (
            repo.parse_path("hf://datasets/my-org/dataset/data/train.csv")
            == "data/train.csv"
        )

        # Test nested file path
        assert (
            repo.parse_path("hf://datasets/my-org/dataset/models/weights.bin")
            == "models/weights.bin"
        )

        # Test empty path (repo root)
        assert repo.parse_path("hf://datasets/my-org/dataset/") == ""
        assert repo.parse_path("hf://datasets/my-org/dataset") == ""

        # Test model repository
        model_repo = HfRepo(
            entity="openai", name="gpt-model", type="model", internal=False
        )
        assert (
            model_repo.parse_path("hf://openai/gpt-model/config.json") == "config.json"
        )

        # Test space repository
        space_repo = HfRepo(
            entity="gradio", name="calculator", type="space", internal=False
        )
        assert space_repo.parse_path("hf://spaces/gradio/calculator/app.py") == "app.py"

        # Test error case - wrong repository
        with pytest.raises(ValueError, match="does not start with repository prefix"):
            repo.parse_path("hf://datasets/other-org/other-dataset/file.txt")
