import logging
import warnings


def initialize_logging(
    level: int = logging.INFO, format: str = "%(asctime)s - %(levelname)s - %(message)s"
) -> None:
    """
    Initialize logging configuration.

    Parameters
    ----------
    level : int, optional
        Logging level, by default logging.INFO
    format : str, optional
        Log message format, by default "%(asctime)s - %(levelname)s - %(message)s"
    """
    logging.basicConfig(level=level, format=format)


def filter_known_warnings() -> None:
    """Filter known warnings."""
    # Ignore universal-pathlib warning about not having an explicit Hugging Face implementation; see:
    # https://github.com/fsspec/universal_pathlib?tab=readme-ov-file#currently-supported-filesystems-and-protocols
    warnings.filterwarnings(
        "ignore",
        category=UserWarning,
        message="UPath 'hf' filesystem not explicitly implemented.",
    )
