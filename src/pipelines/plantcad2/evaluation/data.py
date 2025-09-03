"""Data loading and processing utilities for PlantCAD2 evaluation."""

from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """Dataset for masked language model evaluation on sequences."""

    def __init__(
        self, sequences: list[str], names: list[str], tokenizer, mask_token_id: int
    ):
        self.sequences = sequences
        self.names = names
        self.tokenizer = tokenizer
        self.mask_token_id = mask_token_id

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        sequence = self.sequences[idx]
        name = self.names[idx]
        encoding = self.tokenizer.encode_plus(
            sequence,
            return_tensors="pt",
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids = encoding["input_ids"]
        input_ids[0, self.mask_token_id] = self.tokenizer.mask_token_id
        return {"sequence": sequence, "name": name, "input_ids": input_ids}
