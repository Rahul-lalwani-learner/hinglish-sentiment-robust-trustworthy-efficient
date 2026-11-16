"""Efficiency study: make the teacher (spelling-augmented HingBERT) small and fast.

Three steps, each measured on its own:
1. Vocabulary trimming  - keep only word pieces our tweets use (plus every single
   character, so any unseen spelling can still be written out). Drops most of the
   30,522-row embedding table; nothing else changes.
2. Distillation 12 -> 6 layers - the student starts from every second teacher layer
   and learns from the teacher's soft predictions plus the true labels
   (Hinton et al., 2015; Sanh et al., 2019).
3. Dynamic int8 quantisation of all Linear layers (CPU inference).

Sizes are measured as the bytes of the saved weights; speed as CPU milliseconds per
tweet at batch size 1, which is how a single live prediction is served.
"""
from __future__ import annotations

import copy
import io
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertTokenizerFast,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

from .calibration import apply_temperature, ece_score, fit_temperature
from .train import get_device, predict_logits, make_loader, scores, set_seed

SPECIAL = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


# --- 1. vocabulary trimming ----------------------------------------------------
def trimmed_vocab(tok, text_lists: list[list[str]], max_len: int = 128) -> list[str]:
    """Tokens to keep, in the teacher's original id order."""
    vocab = tok.get_vocab()
    keep = {vocab[t] for t in SPECIAL if t in vocab}
    for texts in text_lists:
        for ids in tok(list(texts), truncation=True, max_length=max_len)["input_ids"]:
            keep.update(ids)
    for t, i in vocab.items():  # every single character, plain and as a ##continuation
        if len(t) == 1 or (t.startswith("##") and len(t) == 3):
            keep.add(i)
    inv = {i: t for t, i in vocab.items()}
    return [inv[i] for i in sorted(keep)]


def trim_model(model, tok, keep_tokens: list[str], out_dir: str | Path):
    """Slice the embedding table to `keep_tokens`; save model + matching tokenizer."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    old = tok.get_vocab()
    ids = torch.tensor([old[t] for t in keep_tokens])
    m = copy.deepcopy(model).cpu()
    emb = m.get_input_embeddings()
    new = torch.nn.Embedding(len(keep_tokens), emb.weight.shape[1], padding_idx=0)
    new.weight.data = emb.weight.data[ids].clone()
    m.set_input_embeddings(new)
    m.config.vocab_size = len(keep_tokens)
    # Same normaliser / pre-tokeniser / WordPiece settings as the teacher; only the
    # vocabulary and the special-token ids change.
    spec = json.loads(tok.backend_tokenizer.to_str())
    new_ids = {t: i for i, t in enumerate(keep_tokens)}
    spec["model"]["vocab"] = new_ids
    for a in spec.get("added_tokens", []):
        a["id"] = new_ids[a["content"]]
    pp = spec.get("post_processor") or {}
    for name, st in (pp.get("special_tokens") or {}).items():
        st["ids"] = [new_ids[t] for t in st["tokens"]]
    tok.save_pretrained(out_dir)
    (out_dir / "tokenizer.json").write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    (out_dir / "vocab.txt").write_text("\n".join(keep_tokens) + "\n", encoding="utf-8")
    m.save_pretrained(out_dir)
    new_tok = AutoTokenizer.from_pretrained(out_dir)
    assert len(new_tok) == len(keep_tokens), (len(new_tok), len(keep_tokens))
    return m, new_tok


# --- 2. distillation ------------------------------------------------------------
def make_student(teacher, layers=(0, 2, 4, 6, 8, 10)):
    """6-layer copy of the teacher that keeps embeddings, pooler, classifier and
    the chosen encoder layers (DistilBERT-style initialisation)."""
    s = copy.deepcopy(teacher).cpu()
    s.bert.encoder.layer = torch.nn.ModuleList([copy.deepcopy(teacher.bert.encoder.layer[i]) for i in layers])
    s.config.num_hidden_layers = len(layers)
    return s


@dataclass
class DistillConfig:
    lr: float = 5e-5
    batch_size: int = 32
    eval_batch_size: int = 128
    epochs: int = 6
    max_len: int = 128
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    kd_temperature: float = 2.0
    alpha: float = 0.5      # weight on the teacher (soft) loss; 0 = plain fine-tuning
    limit_train: int | None = None


class _KDData(torch.utils.data.Dataset):
    def __init__(self, enc, labels):
        self.enc, self.labels = enc, np.asarray(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = int(self.labels[i])
        item["idx"] = i
        return item


def run_distill(name: str, seed: int, student_init_dir: str | Path, splits: dict,
                teacher_train_logits: np.ndarray | None, out_dir: str | Path,
                cfg: DistillConfig = DistillConfig(), verbose: bool = True) -> dict:
    """Train a student; skip if result.json exists. teacher_train_logits[i] must
    match splits['train'].iloc[i]."""
    run_dir = Path(out_dir) / name / f"seed{seed}"
    if (run_dir / "result.json").exists():
        if verbose:
            print(f"[skip] {name} seed{seed} already done")
        return json.loads((run_dir / "result.json").read_text())
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    device = get_device()
    tok = AutoTokenizer.from_pretrained(student_init_dir)
    model = AutoModelForSequenceClassification.from_pretrained(student_init_dir).to(device)

    tr, va, te = splits["train"], splits["val"], splits["test"]
    t_logits = None if teacher_train_logits is None else torch.tensor(teacher_train_logits, dtype=torch.float32)
    if cfg.limit_train:
        tr = tr.iloc[: cfg.limit_train]
    enc = tok(list(tr.text), truncation=True, max_length=cfg.max_len)
    g = torch.Generator().manual_seed(seed)
    dl_tr = DataLoader(_KDData(enc, tr.label_id), batch_size=cfg.batch_size, shuffle=True, generator=g,
                       collate_fn=DataCollatorWithPadding(tok))
    dl_va = make_loader(tok, va.text, va.label_id, cfg.max_len, cfg.eval_batch_size, False)
    dl_te = make_loader(tok, te.text, te.label_id, cfg.max_len, cfg.eval_batch_size, False)

    no_decay = ["bias", "LayerNorm.weight"]
    params = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(params, lr=cfg.lr)
    total = len(dl_tr) * cfg.epochs
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup_ratio * total), total)
    T, a = cfg.kd_temperature, cfg.alpha if t_logits is not None else 0.0

    history, best_f1, best_state, t0 = [], -1.0, None, time.time()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for batch in dl_tr:
            idx = batch.pop("idx")
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**{k: v for k, v in batch.items() if k != "labels"})
            ce = F.cross_entropy(out.logits, batch["labels"])
            if a > 0:
                tl = t_logits[idx].to(device)
                kd = F.kl_div(F.log_softmax(out.logits / T, -1), F.softmax(tl / T, -1), reduction="batchmean") * T * T
                loss = a * kd + (1 - a) * ce
            else:
                loss = ce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            running += loss.item()
        val = scores(va.label_id.values, predict_logits(model, dl_va, device))
        val.update(epoch=epoch, train_loss=running / len(dl_tr), minutes=(time.time() - t0) / 60)
        history.append(val)
        if verbose:
            print(f"  {name} seed{seed} ep{epoch}: loss {val['train_loss']:.4f} | val wF1 {val['weighted_f1']:.4f} | "
                  f"val ECE {val['ece']:.4f} | {val['minutes']:.1f} min", flush=True)
        if val["weighted_f1"] > best_f1:
            best_f1 = val["weighted_f1"]
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    lv, lt = predict_logits(model, dl_va, device), predict_logits(model, dl_te, device)
    yv, yt = va.label_id.values, te.label_id.values
    np.savez_compressed(run_dir / "logits.npz", val=lv, test=lt, y_val=yv, y_test=yt,
                        uid_val=va.uid.values, uid_test=te.uid.values)
    Tc = fit_temperature(lv, yv)
    result = {"name": name, "seed": seed, "best_epoch": int(max(history, key=lambda h: h["weighted_f1"])["epoch"]),
              "n_params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
              "train_minutes": round((time.time() - t0) / 60, 1), "device": str(device),
              "val": scores(yv, lv), "test": scores(yt, lt), "temperature": Tc,
              "test_ece_after_T": float(ece_score(apply_temperature(lt, Tc), yt)),
              "history": history, "config": asdict(cfg)}
    model.save_pretrained(run_dir / "model")
    tok.save_pretrained(run_dir / "model")
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


# --- 3. quantisation + measurement ------------------------------------------------
def quantize_int8(model):
    torch.backends.quantized.engine = "qnnpack" if "qnnpack" in torch.backends.quantized.supported_engines else "fbgemm"
    return torch.ao.quantization.quantize_dynamic(copy.deepcopy(model).cpu().eval(), {torch.nn.Linear}, dtype=torch.qint8)


def size_mb(model) -> float:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 2**20


@torch.no_grad()
def cpu_latency_ms(model, tok, texts, n: int = 300, threads: int = 4, max_len: int = 128) -> float:
    """Median CPU milliseconds per tweet at batch size 1."""
    torch.set_num_threads(threads)
    model = model.cpu().eval()
    texts = list(texts)[:n]
    for t in texts[:10]:  # warm-up
        model(**tok(t, return_tensors="pt", truncation=True, max_length=max_len))
    times = []
    for t in texts:
        x = tok(t, return_tensors="pt", truncation=True, max_length=max_len)
        t0 = time.perf_counter()
        model(**x)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


@torch.no_grad()
def cpu_logits(model, tok, texts, max_len: int = 128, batch_size: int = 64) -> np.ndarray:
    model = model.cpu().eval()
    out, texts = [], list(texts)
    for i in range(0, len(texts), batch_size):
        x = tok(texts[i:i + batch_size], return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        out.append(model(**x).logits.float())
    return torch.cat(out).numpy()
