"""Shared model and tokenizer loading utilities."""

import torch
from typing import Type, TypeVar
from transformers import AutoModel, AutoTokenizer, AutoConfig, AutoModelForMaskedLM

# Bounded type alias for AutoModel subclasses
AutoModelType = TypeVar("AutoModelType", bound=AutoModel)


def _load_auto_model(
    path: str,
    model_class: Type[AutoModelType],
    revision: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> AutoModelType:
    config = AutoConfig.from_pretrained(path, revision=revision, trust_remote_code=True)
    model = model_class.from_pretrained(
        path, config=config, revision=revision, trust_remote_code=True, dtype=dtype
    )
    return model


def load_model(
    path: str, revision: str | None = None, dtype: torch.dtype = torch.bfloat16
) -> AutoModel:
    """
    Load a pre-trained AutoModel.

    Parameters
    ----------
    path : str
        Path or model identifier to load from
    revision : str | None, optional
        Model revision to load, by default None
    dtype : torch.dtype, optional
        Data type for the model, by default torch.bfloat16

    Returns
    -------
    AutoModel
        Loaded pre-trained model ready for inference
    """
    return _load_auto_model(path, AutoModel, revision, dtype)


def load_model_for_masked_lm(
    path: str, revision: str | None = None, dtype: torch.dtype = torch.bfloat16
) -> AutoModelForMaskedLM:
    """
    Load a pre-trained AutoModelForMaskedLM.

    Parameters
    ----------
    path : str
        Path or model identifier to load from
    revision : str | None, optional
        Model revision to load, by default None
    dtype : torch.dtype, optional
        Data type for the model, by default torch.bfloat16

    Returns
    -------
    AutoModelForMaskedLM
        Loaded pre-trained model ready for inference
    """
    return _load_auto_model(path, AutoModelForMaskedLM, revision, dtype)


def load_tokenizer(path: str, revision: str | None = None) -> AutoTokenizer:
    """
    Load a pre-trained AutoTokenizer.

    Parameters
    ----------
    path : str
        Path or model identifier to load from
    revision : str | None, optional
        Model revision to load, by default None

    Returns
    -------
    AutoTokenizer
        Loaded tokenizer
    """
    return AutoTokenizer.from_pretrained(
        path, revision=revision, trust_remote_code=True
    )
