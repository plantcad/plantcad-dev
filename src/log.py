import logging


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
