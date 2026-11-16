"""Loading and cleaning the SemEval-2020 Task 9 (SentiMix) Hinglish data.

Raw format (CoNLL-like):
    meta<TAB><uid><TAB><label>      <- label missing in the test file
    <token><TAB><lang>              <- lang in {Hin, Eng, O, EMT}
    <blank line>
Test labels live in a separate CSV (Uid,Sentiment).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import ftfy
import pandas as pd

LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}

RAW_FILES = {
    "train": "Hinglish_train_14k_split_conll.txt",
    "val": "Hinglish_dev_3k_split_conll.txt",
    "test": "Hinglish_test_unalbelled_conll_updated.txt",
}
TEST_LABELS = "Hinglish_test_labels.txt"


def parse_conll(path: Path) -> list[dict]:
    records, cur = [], None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("meta\t"):
                parts = line.split("\t")
                cur = {
                    "uid": int(parts[1]),
                    "label": parts[2].strip() if len(parts) > 2 else None,
                    "tokens": [],
                    "langs": [],
                }
                records.append(cur)
            elif line.strip() and cur is not None:
                parts = line.split("\t")
                cur["tokens"].append(ftfy.fix_text(parts[0]))
                cur["langs"].append(parts[1].strip() if len(parts) > 1 else "O")
    return records


# --- text normalisation -------------------------------------------------------
# The CoNLL tokeniser split URLs, @mentions and #hashtags into pieces
# ("https // t co / XPCz", "@ narendramodi", "# IndVsAus"). We stitch them back.
_URL = re.compile(r"https?\s*:?\s*//\s*\S+(?:\s*[/.]\s*\S+)*|\bt\s+co\s*/\s*\S+")
_MENTION = re.compile(r"@\s*_?\s*[^\s@#]+(?:\s+_\s+[^\s@#_]+)*(?:\s+_(?=\s|$))?")
_HASHTAG = re.compile(r"#\s+(\S+)")
_SPACES = re.compile(r"\s+")


def normalise(text: str) -> str:
    text = _URL.sub(" http ", text)
    text = _MENTION.sub(" @user ", text)
    text = _HASHTAG.sub(r"#\1", text)
    text = text.replace("…", " ")
    return _SPACES.sub(" ", text).strip()


def code_mixing_index(langs: list[str]) -> float:
    """CMI (Das & Gambäck 2014): 0 = monolingual, higher = more mixed."""
    hin, eng = langs.count("Hin"), langs.count("Eng")
    n = hin + eng
    if n == 0:
        return 0.0
    return 100.0 * (1 - max(hin, eng) / n)


def to_frame(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    df["text_raw"] = df["tokens"].apply(" ".join)
    df["text"] = df["text_raw"].apply(normalise)
    df["n_tokens"] = df["tokens"].apply(len)
    for tag in ["Hin", "Eng", "O", "EMT"]:
        df[f"n_{tag.lower()}"] = df["langs"].apply(lambda ls, t=tag: ls.count(t))
    df["cmi"] = df["langs"].apply(code_mixing_index)
    return df


def load_raw_splits(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    splits = {k: to_frame(parse_conll(raw_dir / v)) for k, v in RAW_FILES.items()}
    test_labels = pd.read_csv(raw_dir / TEST_LABELS)
    test_labels.columns = ["uid", "label"]
    splits["test"] = splits["test"].drop(columns="label").merge(test_labels, on="uid", how="left")
    for df in splits.values():
        df["label"] = df["label"].str.strip().str.lower()
        df["label_id"] = df["label"].map(LABEL2ID)
    return splits


# --- processed files (what every later notebook reads) -----------------------
KEEP = ["uid", "text", "text_raw", "label", "label_id", "tokens", "langs", "n_tokens", "cmi"]


def save_splits(splits: dict[str, pd.DataFrame], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        df[KEEP].to_json(out_dir / f"{name}.jsonl", orient="records", lines=True, force_ascii=False)


def load_splits(processed_dir: str | Path) -> dict[str, pd.DataFrame]:
    processed_dir = Path(processed_dir)
    return {
        name: pd.read_json(processed_dir / f"{name}.jsonl", lines=True)
        for name in ["train", "val", "test"]
    }
