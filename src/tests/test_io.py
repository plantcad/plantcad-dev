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
        "repo_id,type_,expected_entity,expected_name,expected_url",
        [
            (
                "huggingface/bert-base-uncased",
                "model",
                "huggingface",
                "bert-base-uncased",
                "hf://huggingface/bert-base-uncased",
            ),
            (
                "gradio/hello-world",
                "space",
                "gradio",
                "hello-world",
                "hf://spaces/gradio/hello-world",
            ),
            (
                "org/dataset-name",
                "dataset",
                "org",
                "dataset-name",
                "hf://datasets/org/dataset-name",
            ),
        ],
    )
    def test_from_repo_id_with_type(
        self, repo_id, type_, expected_entity, expected_name, expected_url
    ):
        """Test HfRepo.from_repo_id() with different repository types."""
        repo = HfRepo.from_repo_id(repo_id, type=type_)
        assert repo.entity == expected_entity
        assert repo.name == expected_name
        assert repo.type == type_
        assert repo.internal is False
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
