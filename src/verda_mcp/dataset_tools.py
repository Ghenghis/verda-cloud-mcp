"""
Dataset Tools - Download, prepare, validate datasets.
MEGA-TOOL bundling 12 functions into 1 tool.
"""

from dataclasses import dataclass
from enum import Enum


class DatasetSource(Enum):
    HUGGINGFACE = "huggingface"
    KAGGLE = "kaggle"
    URL = "url"


@dataclass
class DatasetInfo:
    name: str
    source: DatasetSource
    size_gb: float
    samples: int
    format: str
    description: str


DATASETS = {
    "alpaca": DatasetInfo("tatsu-lab/alpaca", DatasetSource.HUGGINGFACE, 0.05, 52000, "json", "Instruction following"),
    "dolly-15k": DatasetInfo(
        "databricks/databricks-dolly-15k", DatasetSource.HUGGINGFACE, 0.01, 15000, "json", "Instruction tuning"
    ),
    "openassistant": DatasetInfo(
        "OpenAssistant/oasst1", DatasetSource.HUGGINGFACE, 0.1, 161000, "json", "Conversations"
    ),
    "squad": DatasetInfo("rajpurkar/squad", DatasetSource.HUGGINGFACE, 0.03, 100000, "json", "QA"),
    "code-alpaca": DatasetInfo(
        "sahil2801/CodeAlpaca-20k", DatasetSource.HUGGINGFACE, 0.02, 20000, "json", "Code instructions"
    ),
    "ultrachat": DatasetInfo("stingning/ultrachat", DatasetSource.HUGGINGFACE, 2.0, 1500000, "json", "Multi-turn chat"),
    "slim-orca": DatasetInfo(
        "Open-Orca/SlimOrca", DatasetSource.HUGGINGFACE, 0.8, 518000, "json", "Diverse instructions"
    ),
    "magicoder": DatasetInfo(
        "ise-uiuc/Magicoder-OSS-Instruct-75K", DatasetSource.HUGGINGFACE, 0.1, 75000, "json", "Code generation"
    ),
    "wikitext": DatasetInfo("wikitext", DatasetSource.HUGGINGFACE, 0.5, 4500000, "text", "Language modeling"),
    "fineweb": DatasetInfo(
        "HuggingFaceFW/fineweb", DatasetSource.HUGGINGFACE, 44000, 15000000000, "parquet", "Web pretraining"
    ),
}


def dataset_tools(action: str = "list", dataset: str = "", **kwargs) -> str:
    """
    MEGA-TOOL: Dataset Tools (12 functions).

    Actions: list, search, info, download_script, preview, prepare_script,
    split, tokenize, validate, convert, stats, custom
    """
    if action == "list":
        lines = ["📊 POPULAR DATASETS", "=" * 70]
        for k, d in DATASETS.items():
            lines.append(f"{k:15} | {d.samples:>12,} samples | {d.size_gb:8.2f}GB | {d.description}")
        return "\n".join(lines)

    elif action == "search":
        results = [
            f"{k}: {d.description}"
            for k, d in DATASETS.items()
            if dataset.lower() in k or dataset.lower() in d.description.lower()
        ]
        return "\n".join(results) if results else f"No datasets matching '{dataset}'"

    elif action == "info":
        if dataset not in DATASETS:
            return f"Dataset '{dataset}' not found"
        d = DATASETS[dataset]
        return f"📊 {dataset}\nName: {d.name}\nSamples: {d.samples:,}\nSize: {d.size_gb}GB\nFormat: {d.format}"

    elif action == "download_script":
        if dataset not in DATASETS:
            return "Dataset not found"
        d = DATASETS[dataset]
        return f'''from datasets import load_dataset
ds = load_dataset("{d.name}")
ds.save_to_disk("/workspace/data/{dataset}")
print(f"Saved {{len(ds['train'])}} samples")'''

    elif action == "preview":
        return """from datasets import load_dataset
ds = load_dataset("dataset_name", split="train[:5]")
for ex in ds: print(ex)"""

    elif action == "split":
        return """ds = dataset.train_test_split(test_size=0.1)
train, val = ds["train"], ds["test"]"""

    elif action == "tokenize":
        return """def tokenize(examples):
    return tokenizer(examples["text"], truncation=True, max_length=2048)
tokenized = dataset.map(tokenize, batched=True)"""

    elif action == "validate":
        return "# Check for null values, duplicates, format consistency"

    elif action == "convert":
        return "# dataset.to_json() / dataset.to_parquet() / dataset.to_csv()"

    elif action == "stats":
        if dataset in DATASETS:
            d = DATASETS[dataset]
            return f"{dataset}: {d.samples:,} samples, {d.size_gb}GB"
        return "Dataset not found"

    elif action == "custom":
        return """from datasets import Dataset
import json
with open("data.json") as f: data = json.load(f)
dataset = Dataset.from_list(data)"""

    return "Actions: list, search, info, download_script, preview, split, tokenize, validate, convert, stats, custom"
