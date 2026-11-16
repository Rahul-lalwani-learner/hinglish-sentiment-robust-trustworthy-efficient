"""Experiment 1 (notebook 03) without Jupyter: fine-tune mBERT, XLM-R, MuRIL, HingBERT x 3 seeds.

    python scripts/train_transformers.py            # all 12 runs, finished ones are skipped
    python scripts/train_transformers.py hingbert   # one model

Writes runs/<model>/seed<k>/{result.json, logits.npz} and keeps the best-val seed's checkpoint.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hinglish.data import load_splits
from hinglish.train import MODELS, TrainConfig, keep_best_checkpoint_only, run_finetune

S = load_splits(ROOT / "data/processed")
RUNS = ROOT / "runs"
for key in sys.argv[1:] or ["mbert", "xlmr", "muril", "hingbert"]:
    for seed in [42, 43, 44]:
        print(f"\n=== {key} ({MODELS[key]}) seed {seed} === {time.strftime('%H:%M')}", flush=True)
        r = run_finetune(key, seed, S, RUNS, TrainConfig(), save_model=True)
        print(f"DONE {key} seed{seed}: test wF1 {r['test']['weighted_f1']:.4f} | {r['train_minutes']} min", flush=True)
    print(f"kept checkpoint: {key} seed{keep_best_checkpoint_only(RUNS, key)}", flush=True)
