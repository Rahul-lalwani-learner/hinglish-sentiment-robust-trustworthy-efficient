"""Fine-tuning encoders (mBERT, XLM-R, MuRIL, HingBERT) with identical settings.

Plain PyTorch loop (no HF Trainer) so it is transparent, stable across library
versions, and reusable for distillation later.

Every run saves to  <out_dir>/<model_key>/seed<k>/ :
    result.json   metrics + config + per-epoch history
    logits.npz    val/test logits + labels + uids  (all trust analysis uses these)
    model/        best checkpoint (optional; large)
"""
from __future__ import annotations

import json
import random
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from .calibration import apply_temperature, ece_score, fit_temperature
from .data import LABELS

MODELS = {
    "mbert": "google-bert/bert-base-multilingual-cased",
    "xlmr": "FacebookAI/xlm-roberta-base",
    "muril": "google/muril-base-cased",
    "hingbert": "l3cube-pune/hing-bert",
}


@dataclass
class TrainConfig:
    lr: float = 2e-5
    batch_size: int = 32
    eval_batch_size: int = 128
    epochs: int = 4
    max_len: int = 128
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    fp16: bool = True  # only used on CUDA
    limit_train: int | None = None  # for smoke tests
    limit_eval: int | None = None
    extra: dict = field(default_factory=dict)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _Enc(torch.utils.data.Dataset):
    def __init__(self, enc, labels):
        self.enc, self.labels = enc, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = int(self.labels[i])
        return item


def make_loader(tok, texts, labels, max_len, batch_size, shuffle, seed=0):
    enc = tok(list(texts), truncation=True, max_length=max_len)
    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        _Enc(enc, np.asarray(labels)),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=g if shuffle else None,
        collate_fn=DataCollatorWithPadding(tok),
    )


@torch.no_grad()
def predict_logits(model, loader, device) -> np.ndarray:
    model.eval()
    out = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            out.append(model(**batch).logits.float().cpu())
    return torch.cat(out).numpy()


def scores(y, logits) -> dict:
    pred = logits.argmax(1)
    return {
        "weighted_f1": float(f1_score(y, pred, average="weighted")),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "accuracy": float(accuracy_score(y, pred)),
        "ece": float(ece_score(softmax(logits, axis=1), y)),
    }


def run_finetune(
    model_key: str,
    seed: int,
    splits: dict,
    out_dir: str | Path,
    cfg: TrainConfig = TrainConfig(),
    save_model: bool = True,
    text_col: str = "text",
    verbose: bool = True,
) -> dict:
    """Fine-tune one model with one seed. Skips if result.json already exists (Colab resume)."""
    run_dir = Path(out_dir) / model_key / f"seed{seed}"
    if (run_dir / "result.json").exists():
        if verbose:
            print(f"[skip] {model_key} seed{seed} already done")
        return json.loads((run_dir / "result.json").read_text())
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    device = get_device()
    hf_name = MODELS.get(model_key, model_key)
    tok = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_name,
        num_labels=len(LABELS),
        id2label=dict(enumerate(LABELS)),
        label2id={l: i for i, l in enumerate(LABELS)},
    ).to(device)

    tr, va, te = splits["train"], splits["val"], splits["test"]
    if cfg.limit_train:
        tr = tr.sample(cfg.limit_train, random_state=seed)
    if cfg.limit_eval:
        va, te = va.head(cfg.limit_eval), te.head(cfg.limit_eval)

    dl_tr = make_loader(tok, tr[text_col], tr.label_id, cfg.max_len, cfg.batch_size, True, seed)
    dl_va = make_loader(tok, va[text_col], va.label_id, cfg.max_len, cfg.eval_batch_size, False)
    dl_te = make_loader(tok, te[text_col], te.label_id, cfg.max_len, cfg.eval_batch_size, False)

    no_decay = ["bias", "LayerNorm.weight", "layer_norm", "LayerNorm.bias"]
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=cfg.lr)
    total = len(dl_tr) * cfg.epochs
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup_ratio * total), total)
    use_amp = cfg.fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history, best_f1, best_state = [], -1.0, None
    t0 = time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(dl_tr, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = model(**batch).loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += loss.item()
        val = scores(va.label_id.values, predict_logits(model, dl_va, device))
        val.update(epoch=epoch, train_loss=running / len(dl_tr), minutes=(time.time() - t0) / 60)
        history.append(val)
        if verbose:
            print(f"  {model_key} seed{seed} ep{epoch}: loss {val['train_loss']:.4f} | "
                  f"val wF1 {val['weighted_f1']:.4f} | val ECE {val['ece']:.4f} | {val['minutes']:.1f} min")
        if val["weighted_f1"] > best_f1:
            best_f1 = val["weighted_f1"]
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    lv = predict_logits(model, dl_va, device)
    lt = predict_logits(model, dl_te, device)
    yv, yt = va.label_id.values, te.label_id.values
    np.savez_compressed(run_dir / "logits.npz", val=lv, test=lt, y_val=yv, y_test=yt,
                        uid_val=va.uid.values, uid_test=te.uid.values)

    T = fit_temperature(lv, yv)
    result = {
        "model_key": model_key,
        "hf_name": hf_name,
        "seed": seed,
        "best_epoch": int(max(history, key=lambda h: h["weighted_f1"])["epoch"]),
        "n_params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "train_minutes": round((time.time() - t0) / 60, 1),
        "device": str(device),
        "val": scores(yv, lv),
        "test": scores(yt, lt),
        "temperature": T,
        "test_ece_after_T": float(ece_score(apply_temperature(lt, T), yt)),
        "history": history,
        "config": asdict(cfg),
    }
    if save_model:
        model.save_pretrained(run_dir / "model")
        tok.save_pretrained(run_dir / "model")
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    del model, best_state
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def keep_best_checkpoint_only(out_dir: str | Path, model_key: str) -> int | None:
    """Delete model/ folders of all seeds except the one with best val weighted F1 (saves Drive space)."""
    runs = sorted((Path(out_dir) / model_key).glob("seed*/result.json"))
    if not runs:
        return None
    res = [json.loads(p.read_text()) for p in runs]
    best = max(res, key=lambda r: r["val"]["weighted_f1"])["seed"]
    for r in res:
        d = Path(out_dir) / model_key / f"seed{r['seed']}" / "model"
        if r["seed"] != best and d.exists():
            shutil.rmtree(d)
    return best
