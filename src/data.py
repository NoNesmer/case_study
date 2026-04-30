"""CoNLL-2003 NER data loading + IOB1 → IOB2 conversion.

Ports the synalp .txt mirror loader from the fast version.
"""
from __future__ import annotations

import os
import urllib.request
from typing import List, Tuple

Sentence = List[Tuple[str, str]]

URL_BASE = "https://raw.githubusercontent.com/synalp/NER/master/corpus/CoNLL-2003"
FILES = {"train": "eng.train", "valid": "eng.testa", "test": "eng.testb"}


def fetch(name: str, data_dir: str = "data") -> str:
    os.makedirs(data_dir, exist_ok=True)
    fname = FILES[name]
    path = os.path.join(data_dir, fname)
    if not os.path.exists(path):
        url = f"{URL_BASE}/{fname}"
        print(f"Downloading {name} from {url}")
        urllib.request.urlretrieve(url, path)
    else:
        print(f"Cached: {path}")
    return path


def parse_conll(path: str) -> List[Sentence]:
    """Parse CoNLL-2003 token-per-line format → list[list[(token, ner_tag)]]."""
    sents: List[Sentence] = []
    cur: Sentence = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r").strip()
            if not line:
                if cur:
                    sents.append(cur)
                    cur = []
                continue
            if line.startswith("-DOCSTART-"):
                if cur:
                    sents.append(cur)
                    cur = []
                continue
            parts = line.split()
            tok, ner = parts[0], parts[-1]
            cur.append((tok, ner))
    if cur:
        sents.append(cur)
    return sents


def iob1_to_iob2(sents: List[Sentence]) -> List[Sentence]:
    """CoNLL-2003 ships in IOB1; seqeval IOB2 strict mode wants IOB2."""
    out = []
    for s in sents:
        new_s = []
        prev_type = None
        for tok, tag in s:
            if tag.startswith("I-"):
                ent = tag[2:]
                if prev_type != ent:
                    tag = "B-" + ent
                prev_type = ent
            elif tag.startswith("B-"):
                prev_type = tag[2:]
            else:
                prev_type = None
            new_s.append((tok, tag))
        out.append(new_s)
    return out


def load_conll(data_dir: str = "data"):
    """Returns (train_sents, valid_sents, test_sents, label_list)."""
    train = iob1_to_iob2(parse_conll(fetch("train", data_dir)))
    valid = iob1_to_iob2(parse_conll(fetch("valid", data_dir)))
    test = iob1_to_iob2(parse_conll(fetch("test", data_dir)))
    tags = {t for s in (train + valid + test) for _, t in s}
    label_list = ["O"] + sorted(t for t in tags if t != "O")
    return train, valid, test, label_list
