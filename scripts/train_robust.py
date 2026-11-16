"""Experiment 2 (notebook 05) without Jupyter: HingBERT with spelling augmentation (3 seeds)
and plain HingBERT seeds 42/44 for a fair 3-vs-3 comparison (seed 43 comes from runs/).

    python scripts/train_robust.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hinglish.data import load_splits
from hinglish.robust import augment_train, english_vocab, mine_variants
from hinglish.train import TrainConfig, run_finetune

S = load_splits(ROOT / "data/processed")
eng = english_vocab(S["train"])
V = mine_variants(S["train"], eng)  # train only
OUT = ROOT / "runs_robust"
for kind, seed in [("aug", 42), ("aug", 43), ("aug", 44), ("plain", 42), ("plain", 44)]:
    splits = dict(S)
    if kind == "aug":
        splits["train"] = augment_train(S["train"], V, eng, seed=seed)
    print(f"\n=== hingbert {kind} seed {seed} === {time.strftime('%H:%M')}", flush=True)
    r = run_finetune("hingbert", seed, splits, OUT / kind, TrainConfig(), save_model=True)
    print(f"DONE hingbert {kind} seed{seed}: clean test wF1 {r['test']['weighted_f1']:.4f} | {r['train_minutes']} min", flush=True)
