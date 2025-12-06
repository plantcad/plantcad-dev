import s3fs


def filesystem() -> s3fs.S3FileSystem:
    """Create an S3FileSystem instance with retry configuration.

    Returns
    -------
    s3fs.S3FileSystem
        An S3FileSystem instance configured with retries
    """
    # Configure retries via botocore config
    # See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/retries.html
    return s3fs.S3FileSystem(
        config_kwargs={
            "retries": {
                "max_attempts": 10,
                "mode": "adaptive",
            }
        }
    )
