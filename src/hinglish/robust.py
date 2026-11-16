"""Robustness study: respell romanised-Hindi words the way people actually type them.

Two kinds of noise, applied only to *eligible* Hindi-tagged words:

* natural  - swap a word for another spelling of it that real users wrote in the
             TRAIN set (bahut -> bohot / bhot, nahi -> nhi). Variants are found by
             grouping words with the same consonant skeleton. Words with no mined
             variant fall back to a rule edit, so every chosen word changes.
* rule     - SMS-style edits only: drop inner vowels, stretch / shorten a vowel,
             w<->v, z<->j, ph<->f, ee<->i, oo<->u, q->k, drop a final nasal n.

Level p = share of eligible Hindi words that get respelled (0.25 / 0.5 / 1.0).
The sentiment label never changes; only the spelling does.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

from .data import normalise

VOWELS = set("aeiou")
_ALPHA = re.compile(r"^[a-z]+$")


# --- which words may be respelled ---------------------------------------------
def english_vocab(df: pd.DataFrame) -> set[str]:
    """Words tagged Eng more often than Hin. The tagger labels many English words
    (and @handles) as Hin, so these are excluded from respelling."""
    hin, eng = Counter(), Counter()
    for toks, langs in zip(df.tokens, df.langs):
        for t, l in zip(toks, langs):
            (hin if l == "Hin" else eng if l == "Eng" else Counter())[t.lower()] += 1
    return {w for w in set(hin) | set(eng) if eng[w] >= hin[w]}


def eligible(tokens: list[str], langs: list[str], eng: set[str]) -> list[int]:
    out = []
    for i, (t, l) in enumerate(zip(tokens, langs)):
        w = t.lower()
        if l != "Hin" or len(w) < 3 or not _ALPHA.match(w) or w in eng:
            continue
        if i > 0 and tokens[i - 1] in {"@", "#"}:  # mention / hashtag pieces
            continue
        if t[0].isupper() and i > 0:              # likely a name (Kashmir, Modi)
            continue
        out.append(i)
    return out


# --- natural variants mined from real tweets -----------------------------------
# Distinct words whose spellings differ only by a vowel; never swap these.
CONFUSABLE = {frozenset(p) for p in [
    ("bhai", "bhi"), ("hai", "hi"), ("hain", "hun"), ("main", "man"), ("mein", "man"),
    ("tum", "tm"), ("kam", "kaam"), ("sab", "sabse"), ("par", "pyar"), ("pyar", "pir"),
    ("din", "dino"), ("koi", "kai"), ("kya", "kyu"), ("sir", "sar"), ("dil", "dal"),
    ("mat", "mit"), ("bat", "but"), ("log", "lag"), ("ladki", "ladka"), ("tera", "teri"),
    ("bhi", "bahi"), ("kya", "kiya"), ("kiya", "kyaa"), ("kiya", "kyaaa"), ("hum", "hmm"),
    ("hum", "hmmm"), ("karne", "karane"), ("karte", "karate"), ("karta", "karata"),
]}
def skeleton(w: str) -> str:
    w = re.sub(r"(.)\1+", r"\1", w.lower())
    for a, b in [("ph", "f"), ("w", "v"), ("z", "j"), ("q", "k"), ("ck", "k")]:
        w = w.replace(a, b)
    return w[0] + "".join(c for c in w[1:] if c not in VOWELS)


_VOWEL_SWAPS = {frozenset(p) for p in [("a", "o"), ("u", "o"), ("a", "e"), ("i", "e"), ("ai", "e"),
                                         ("au", "o"), ("i", "y"), ("e", "y"), ("ee", "i")]}
_CONS_SWAPS = {frozenset(p) for p in [("w", "v"), ("z", "j"), ("ph", "f"), ("q", "k"), ("ck", "k")]}


def _collapse(w: str) -> str:
    return re.sub(r"(.)\1+", r"\1", w)


def plausible_variant(w: str, v: str) -> bool:
    """True if v differs from w only in the ways people respell Hinglish:
    repeated letters (haii), dropped inner vowels (bahut -> bht, nahi -> nhi),
    a few vowel swaps in words of 5+ letters (bahut -> bohot), w/v, z/j, ph/f."""
    a, b = _collapse(w), _collapse(v)
    if a == b:
        return True
    ops = [o for o in SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if o[0] != "equal"]
    for tag, i1, i2, j1, j2 in ops:
        x, y = a[i1:i2], b[j1:j2]
        if tag in ("delete", "insert"):
            seg, src, k1, k2 = (x, a, i1, i2) if tag == "delete" else (y, b, j1, j2)
            inner = 0 < k1 and k2 < len(src)
            between_consonants = inner and src[k1 - 1] not in VOWELS and src[k2] not in VOWELS
            if not (set(seg) <= VOWELS and between_consonants):
                return False
        else:  # replace
            pair = frozenset((x, y))
            if pair in _CONS_SWAPS:
                continue
            if pair in _VOWEL_SWAPS and min(len(w), len(v)) >= 5:
                continue
            return False
    return True


def mine_variants(df: pd.DataFrame, eng: set[str], min_count: int = 2) -> dict[str, list[str]]:
    """word -> other spellings of it seen at least `min_count` times in df."""
    freq = Counter()
    for toks, langs in zip(df.tokens, df.langs):
        for i in eligible(toks, langs, eng):
            freq[toks[i].lower()] += 1
    groups = defaultdict(list)
    for w, n in freq.items():
        if n >= min_count and len(w) >= 3:
            groups[skeleton(w)].append(w)
    variants = {}
    for words in groups.values():
        if len(words) < 2:
            continue
        for w in words:
            cands = [v for v in words if v != w and v[0] == w[0] and v[-1] == w[-1]
                     and abs(len(v) - len(w)) <= 2
                     and frozenset((w, v)) not in CONFUSABLE
                     and plausible_variant(w, v)]
            if cands:
                variants[w] = sorted(cands, key=lambda v: -freq[v])
    return variants


# --- rule-based SMS edits -------------------------------------------------------
_SWAPS = [("w", "v"), ("v", "w"), ("z", "j"), ("ph", "f"), ("f", "ph"), ("ee", "i"),
          ("oo", "u"), ("q", "k"), ("ai", "e"), ("au", "o")]


def rule_respell(w: str, rng: np.random.Generator) -> str:
    def drop_vowels(s):
        if len(s) < 4:                          # hai -> hi would change the word
            return s
        inner = [i for i in range(1, len(s) - 1) if s[i] in VOWELS]
        if not inner:
            return s
        k = max(1, len(inner) // 2)
        drop = set(rng.choice(inner, size=k, replace=False).tolist())
        return "".join(c for i, c in enumerate(s) if i not in drop)

    def stretch(s):
        idx = [i for i, c in enumerate(s) if c in VOWELS]
        if not idx:
            return s
        i = int(rng.choice(idx))
        if i + 1 < len(s) and s[i + 1] == s[i]:
            return s[:i] + s[i + 1:]             # shorten aa -> a
        return s[:i + 1] + s[i] + s[i + 1:]      # stretch a -> aa

    def swap(s):
        opts = [(a, b) for a, b in _SWAPS if a in s]
        if not opts:
            return s
        a, b = opts[int(rng.integers(len(opts)))]
        return s.replace(a, b, 1)

    def drop_final_n(s):
        return s[:-1] if len(s) > 3 and s.endswith(("in", "en", "on")) else s

    ops = [drop_vowels, stretch, swap, drop_final_n]
    for k in rng.permutation(len(ops)):
        out = ops[k](w)
        if out != w and len(out) >= 2:
            return out
    return w


# --- apply to a whole split -----------------------------------------------------
def _match_case(src: str, new: str) -> str:
    return new[0].upper() + new[1:] if src[:1].isupper() else new


def respell_tokens(tokens, langs, p, mode, variants, eng, rng):
    toks = list(tokens)
    idx = eligible(toks, langs, eng)
    n_changed = 0
    for i in idx:
        if rng.random() >= p:
            continue
        w = toks[i].lower()
        if mode in ("natural", "mined") and w in variants:
            cands = variants[w]
            new = cands[int(rng.integers(len(cands)))]
        elif mode == "mined":                   # augmentation: real variants only
            continue
        else:
            new = rule_respell(w, rng)
        if new != w:
            toks[i] = _match_case(toks[i], new)
            n_changed += 1
    return toks, n_changed, len(idx)


def perturb(df: pd.DataFrame, p: float, mode: str, variants: dict, eng: set, seed: int = 0) -> pd.DataFrame:
    """Return a copy of df with Hindi words respelled; text is rebuilt exactly like
    data.to_frame (join tokens, then normalise)."""
    assert mode in {"natural", "rule", "mined"}
    rng = np.random.default_rng(seed)
    out = df.copy()
    rows = [respell_tokens(t, l, p, mode, variants, eng, rng) for t, l in zip(df.tokens, df.langs)]
    out["tokens"] = [r[0] for r in rows]
    out["n_respelled"] = [r[1] for r in rows]
    out["n_eligible"] = [r[2] for r in rows]
    out["text_raw"] = out.tokens.apply(" ".join)
    out["text"] = out.text_raw.apply(normalise)
    return out


def augment_train(train: pd.DataFrame, variants: dict, eng: set, share: float = 0.5,
                  levels=(0.25, 0.5, 1.0), seed: int = 0) -> pd.DataFrame:
    """Same size as train: a `share` of tweets is replaced by a respelled copy
    (natural variants mined from train only, random level). Equal compute to plain training."""
    rng = np.random.default_rng(seed)
    pick = rng.random(len(train)) < share
    level_of = rng.integers(len(levels), size=len(train))
    out = train.copy()
    for k, lvl in enumerate(levels):
        m = pick & (level_of == k)
        if m.any():
            pert = perturb(train[m], lvl, "mined", variants, eng, seed=seed * 100 + int(lvl * 100))
            out.loc[m, ["tokens", "text_raw", "text"]] = pert[["tokens", "text_raw", "text"]].values
    return out
