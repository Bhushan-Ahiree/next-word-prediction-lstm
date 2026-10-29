from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"

SPLIT_FILES = {
    "train": "train-00000-of-00001.parquet",
    "validation": "validation-00000-of-00001.parquet",
    "test": "test-00000-of-00001.parquet",
}

def load_split(split_name: str) -> pd.DataFrame:
    """Load one original WikiText-2 split."""
    
    if split_name not in SPLIT_FILES:
        raise ValueError(
            f"Unknown split: {split_name}. "
            f"Choose from {list(SPLIT_FILES)}."
        )

    file_path = DATA_DIR / SPLIT_FILES[split_name]
    
    if not file_path.exists():
        raise FileNotFoundError("Dataset file not found: {file_path}")
    
    return pd.read_parquet(file_path)

def tokenize_rows(df: pd.DataFrame) -> list[list[str]]:
    """Convert non-empty text rows into whitespace-separated tokens. Each row remains a separate sequence. No text is modified on disk."""
    
    if "text" not in df.columns:
        raise ValueError("Expected a 'text' column.")
    
    sequences = []
    
    for text in df["text"].fillna("").astype(str):
        tokens = text.split()
        
        if tokens:
            sequences.append(tokens)
        
    return sequences



