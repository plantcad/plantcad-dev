# Usage:
# uv run src/pipelines/plantcad2/evaluation.py run --model_path /path/to/your/model

import pandas as pd
import torch
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
from metaflow import FlowSpec, step, Parameter


class SequenceDataset(Dataset):
    def __init__(self, sequences, names, tokenizer, mask_token_id):
        self.sequences = sequences
        self.names = names
        self.tokenizer = tokenizer
        self.mask_token_id = mask_token_id

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
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


class EvolutionaryConstraintFlow(FlowSpec):
    """A Metaflow pipeline for evolutionary constraint evaluation."""

    model_path = Parameter(
        "model_path", help="Path to the pre-trained model", required=True
    )

    dataset_split = Parameter(
        "dataset_split", help="Dataset split to use (valid or test)", default="valid"
    )

    sample_size = Parameter(
        "sample_size",
        help="Number of samples to downsample to (None for full dataset)",
        default=None,
        type=int,
    )

    output_path = Parameter(
        "output_path",
        help="Output directory for intermediate files",
        default="data/evolutionary_constraint",
    )

    device = Parameter("device", help="Device to run the model on", default="cuda:0")

    batch_size = Parameter(
        "batch_size", help="Batch size for model inference", default=128, type=int
    )

    token_idx = Parameter(
        "token_idx", help="Index of the nucleotide to mask", default=255, type=int
    )

    @step
    def start(self):
        """Initialize the pipeline and setup directories."""
        print("Starting Evolutionary Constraint evaluation pipeline")

        # Setup directories
        self.output_dir = Path(self.output_path).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Output directory: {self.output_dir}")
        print(f"Model path: {self.model_path}")
        print(f"Dataset split: {self.dataset_split}")
        print(f"Sample size: {self.sample_size}")

        self.next(self.downsample_dataset)

    @step
    def downsample_dataset(self):
        """Load and downsample the HuggingFace dataset."""
        print("Loading and downsampling dataset")

        repo_id = "kuleshov-group/cross-species-single-nucleotide-annotation"

        # Load dataset
        data = load_dataset(
            repo_id,
            data_files={
                self.dataset_split: f"Evolutionary_constraint/{self.dataset_split}.tsv"
            },
        )
        df = data[self.dataset_split].to_pandas()

        print(f"Original dataset size: {len(df)}")

        # Downsample if requested
        if self.sample_size is not None and self.sample_size < len(df):
            df = df.sample(n=self.sample_size, random_state=42).reset_index(drop=True)
            print(f"Downsampled to: {len(df)} samples")

        # Save downsampled dataset
        self.dataset_path = (
            self.output_dir / f"downsampled_{self.dataset_split}.parquet"
        )
        df.to_parquet(self.dataset_path)

        # Store essential info for next step
        self.num_samples = len(df)

        print(f"Saved downsampled dataset to: {self.dataset_path}")

        self.next(self.generate_logits)

    @step
    def generate_logits(self):
        """Generate logits using the pre-trained model."""
        print("Generating logits with pre-trained model")

        # Load the dataset
        df = pd.read_parquet(self.dataset_path)

        # Initialize model and tokenizer
        try:
            model = AutoModelForMaskedLM.from_pretrained(
                self.model_path, trust_remote_code=True, dtype=torch.bfloat16
            ).to(self.device)
        except Exception as _:
            model = AutoModelForMaskedLM.from_pretrained(
                self.model_path, trust_remote_code=True
            ).to(self.device)
            print("Note: Model not supported for torch.bfloat16, using torch.float32")

        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )

        # Prepare data
        sequences = df["sequences"].tolist()
        names = df["pos"].tolist()

        # Check sequence lengths
        sequence_lengths = df["sequences"].str.len()
        print("Sequence length summary:")
        print(f"  Min: {sequence_lengths.min()}")
        print(f"  Max: {sequence_lengths.max()}")
        print(f"  Mean: {sequence_lengths.mean():.2f}")

        # Check if all sequences have the same length
        unique_lengths = sequence_lengths.unique()
        if len(unique_lengths) > 1:
            raise ValueError(
                f"Found sequences with different lengths. All sequences must have the same length. "
                f"Lengths found: {sorted(unique_lengths)}"
            )

        seq_length = unique_lengths[0]
        print(f"✓ All sequences are {seq_length} characters long")

        # Create dataset and dataloader
        dataset = SequenceDataset(
            sequences=sequences,
            tokenizer=tokenizer,
            names=names,
            mask_token_id=self.token_idx,
        )
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )

        # Generate logits
        nucleotides = list("acgt")
        all_logits = []

        print(f"Processing {len(dataset)} sequences in {len(loader)} batches")

        for batch in tqdm(loader, desc="Generating logits"):
            cur_ids = batch["input_ids"].to(self.device)
            cur_ids = cur_ids.squeeze(1)

            with torch.inference_mode():
                outputs = model(input_ids=cur_ids)
                batch_logits = outputs.logits

                # Extract logits for the masked position and nucleotides
                logits = batch_logits[
                    :, self.token_idx, [tokenizer.get_vocab()[nc] for nc in nucleotides]
                ]
                probs = torch.nn.functional.softmax(logits.cpu(), dim=1).numpy()
                all_logits.append(probs)

        # Combine all logits
        self.logits_matrix = np.vstack(all_logits)

        # Save logits
        self.logits_path = self.output_dir / "logits.tsv"
        np.savetxt(self.logits_path, self.logits_matrix, delimiter="\t")

        print(f"Generated logits for {len(self.logits_matrix)} sequences")
        print(f"Saved logits to: {self.logits_path}")

        self.next(self.merge_scores)

    @step
    def merge_scores(self):
        """Merge scores for plantcad evaluation."""
        print("Merging scores with labels")

        # Load dataset and logits
        df = pd.read_parquet(self.dataset_path)
        logits_df = pd.DataFrame(self.logits_matrix, columns=["A", "C", "G", "T"])

        # Extract reference nucleotide at position 255 (0-indexed)
        REF = df["sequences"].str[255]
        print("Reference nucleotide distribution:")
        ref_counts = REF.value_counts()
        for nuc, count in ref_counts.items():
            print(f"  {nuc}: {count}")

        scores = df.apply(
            lambda row: logits_df.loc[row.name, REF.loc[row.name]]
            if REF.loc[row.name] in "ATCG"
            else 0,
            axis=1,
        )

        # Add scores to dataframe
        df["plantcad_scores"] = scores

        # Save dataset with scores
        self.scored_dataset_path = self.output_dir / "dataset_with_scores.parquet"
        df.to_parquet(self.scored_dataset_path)

        # Store for next step
        self.y_true = df["label"].values
        self.y_scores = scores.values

        print(f"Generated scores for {len(scores)} samples")
        print(f"Non-zero scores: {sum(scores != 0)}")
        print(f"Saved scored dataset to: {self.scored_dataset_path}")

        self.next(self.compute_roc)

    @step
    def compute_roc(self):
        """Compute and print ROC AUC score."""
        print("Computing ROC AUC score")

        # Calculate ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_scores)
        roc_auc = auc(fpr, tpr)

        # Print results
        print("=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"ROC AUC Score: {roc_auc:.4f}")
        print(f"Number of samples: {len(self.y_true)}")
        print(f"Positive labels: {sum(self.y_true)}")
        print(f"Negative labels: {sum(1 - self.y_true)}")
        print("=" * 50)

        # Store final results
        self.roc_auc = roc_auc
        self.num_samples_final = len(self.y_true)
        self.num_positive = sum(self.y_true)
        self.num_negative = sum(1 - self.y_true)

        self.next(self.end)

    @step
    def end(self):
        """Final step of the pipeline."""
        print("Pipeline completed successfully!")
        print(f"Final ROC AUC: {self.roc_auc:.4f}")
        print(f"Processed {self.num_samples_final} samples")
        print(f"Results saved to: {self.output_dir}")


if __name__ == "__main__":
    EvolutionaryConstraintFlow()
