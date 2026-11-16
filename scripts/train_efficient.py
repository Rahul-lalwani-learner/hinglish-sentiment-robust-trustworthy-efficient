"""Experiment 4 (notebook 06) without Jupyter: trim the teacher's vocabulary, build the 6-layer
student, and train it with and without the teacher (3 seeds each). Needs runs_robust/aug/hingbert/seed43.

    python scripts/train_efficient.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from hinglish.data import load_splits
from hinglish.efficient import DistillConfig, make_student, run_distill, trim_model, trimmed_vocab
from hinglish.robust import augment_train, english_vocab, mine_variants, perturb
from hinglish.train import get_device, make_loader, predict_logits

S = load_splits(ROOT / "data/processed")
tr = S["train"]
eng = english_vocab(tr)
V = mine_variants(tr, eng)
OUT = ROOT / "runs_efficient"
OUT.mkdir(exist_ok=True)
T_DIR = ROOT / "runs_robust/aug/hingbert/seed43/model"
tok = AutoTokenizer.from_pretrained(T_DIR)
teacher = AutoModelForSequenceClassification.from_pretrained(T_DIR).eval()

if not (OUT / "student_init/config.json").exists():
    texts = [tr.text, S["val"].text, augment_train(tr, V, eng, share=1.0, seed=7).text] + \
            [perturb(tr, 1.0, "rule", V, eng, seed=s).text for s in (1, 2)]  # never test
    keep = trimmed_vocab(tok, texts)
    tm, ttok = trim_model(teacher, tok, keep, OUT / "teacher_trimmed")
    st = make_student(tm)
    st.save_pretrained(OUT / "student_init")
    ttok.save_pretrained(OUT / "student_init")
    print(f"trimmed vocab {len(keep)} tokens; student init saved", flush=True)

device = get_device()
teacher.to(device)
for kind, alpha in [("student_kd", 0.5), ("student_nokd", 0.0)]:
    for seed in [42, 43, 44]:
        splits = dict(S)
        splits["train"] = augment_train(tr, V, eng, seed=seed)  # same data the teacher saw
        tl = None
        if alpha > 0 and not (OUT / kind / f"seed{seed}/result.json").exists():
            tl = predict_logits(teacher, make_loader(tok, splits["train"].text, splits["train"].label_id, 128, 128, False), device)
        print(f"\n=== {kind} seed {seed} === {time.strftime('%H:%M')}", flush=True)
        r = run_distill(kind, seed, OUT / "student_init", splits, tl, OUT, DistillConfig(alpha=alpha))
        print(f"DONE {kind} seed{seed}: test wF1 {r['test']['weighted_f1']:.4f} | {r['train_minutes']} min", flush=True)
