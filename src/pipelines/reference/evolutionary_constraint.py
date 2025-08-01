import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import os
from tqdm import tqdm
from datasets import load_dataset

repo_id = "kuleshov-group/cross-species-single-nucleotide-annotation"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-input",
        dest="inputDF",
        type=str,
        default=None,
        help="The name of the input dataset, either 'valid' or 'test'.",
    )
    parser.add_argument(
        "-outLogit", dest="outLogit", default=None, help="The directory of output"
    )
    parser.add_argument(
        "-model",
        dest="modelDir",
        default=None,
        help="The directory of pre-trained model",
    )
    parser.add_argument(
        "-device", dest="device", default="cuda:0", help="The device to run the model"
    )
    parser.add_argument(
        "-batchSize",
        dest="batchSize",
        default=128,
        type=int,
        help="The batch size for the model",
    )
    parser.add_argument(
        "-numWorkers",
        dest="numWorkers",
        default=4,
        type=int,
        help="The number of workers for the model",
    )
    parser.add_argument(
        "-tokenIdx",
        dest="tokenIdx",
        default=255,
        type=int,
        help="The index of the nucleotide",
    )
    args = parser.parse_args()
    return args


args = parse_args()


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
        input_ids[0, self.mask_token_id] = (
            self.tokenizer.mask_token_id
        )  # mask the 255th token
        return {"sequence": sequence, "name": name, "input_ids": input_ids}


device = args.device
model_path = args.modelDir

try:
    model = AutoModelForMaskedLM.from_pretrained(
        model_path, trust_remote_code=True, dtype=torch.bfloat16
    ).to(device)
except:  # noqa: E722
    model = AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True).to(
        device
    )
    print(
        "Note: The model is not supported for torch.bfloat16, running with torch.float32"
    )

model.to(device)
model.eval()


# Initialize your tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

if args.inputDF == "valid":
    data = load_dataset(
        repo_id, data_files={"valid": "Evolutionary_constraint/valid.tsv"}
    )
    df = data["valid"].to_pandas()
elif args.inputDF == "test":
    data = load_dataset(
        repo_id, data_files={"test": "Evolutionary_constraint/test.tsv"}
    )
    df = data["test"].to_pandas()
else:
    # if the file not exists, exit
    try:
        df = pd.read_csv(args.inputDF, sep="\t")
    except FileNotFoundError:
        print(f"File {args.inputDF} not found. Please provide a valid input file.")
        exit(1)


sequences = df["sequences"].tolist()
names = df["pos"].tolist()
# Create dataset
dataset = SequenceDataset(
    sequences=sequences, tokenizer=tokenizer, names=names, mask_token_id=args.tokenIdx
)
# Create your data loader
loader = DataLoader(dataset, batch_size=args.batchSize, shuffle=False, num_workers=4)

# if output exists, remove it
if args.outLogit and os.path.exists(args.outLogit):
    os.remove(args.outLogit)


nucleotides = list("acgt")
for batch in tqdm(loader, desc="Inference..."):
    curName = np.array(batch["name"])[:, np.newaxis]
    curIDs = batch["input_ids"].to(device)
    curIDs = curIDs.squeeze(1)
    with torch.inference_mode():
        outputs = model(input_ids=curIDs)

    # if user specify the logits output
    if args.outLogit:
        all_logits = outputs.logits
        logits = all_logits[
            :, args.tokenIdx, [tokenizer.get_vocab()[nc] for nc in nucleotides]
        ]  # get the logits for the mask>
        probs = torch.nn.functional.softmax(logits.cpu(), dim=1).numpy()
        with open(args.outLogit, "a") as f:
            np.savetxt(f, probs, delimiter="\t")
