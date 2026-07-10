"""09 — Sentence-transformer embeddings of scrubbed company descriptions.

Model: sentence-transformers/all-MiniLM-L6-v2 (384-dim). Text is scrubbed of
outcome-revealing sentences (same scrubber as 05/07) BEFORE embedding.
Note: a pretrained encoder is frozen — no fitting on our data — so unlike
TF-IDF/LSA/w2v there is nothing to refit per fold; scrubbing remains the
only leakage control needed on the text side.

Output: data/processed/st_minilm_scrubbed.npy (5999 x 384, row-aligned with
features.parquet sorted by [batch_date, id] — same order as 07_simulation).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

SCRUB_PAT = re.compile(
    r"(?i)\b("
    r"acquired?|acquisition|acquires|acqui-?hired?|merged?|merger|"
    r"went\s+public|going\s+public|ipo'?d?|public\s+offering|"
    r"shut(ting)?\s+down|shutdown|closed\s+(down|its|in)|closing\s+down|"
    r"no\s+longer\s+(operating|active|in\s+operation|in\s+business)|"
    r"ceased\s+operations?|wound\s+down|winding\s+down|out\s+of\s+business|"
    r"defunct|sold\s+to|sold\s+the\s+company|exited\s+to|now\s+part\s+of|"
    r"joined\s+forces\s+with|discontinued"
    r")\b"
)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def scrub_text(t) -> str:
    if not isinstance(t, str) or not t:
        return ""
    return " ".join(s for s in SENT_SPLIT.split(t) if not SCRUB_PAT.search(s))


df = pd.read_parquet(PROC / "features.parquet")
df = df.sort_values(["batch_date", "id"]).reset_index(drop=True)
texts = df["text_clean"].fillna("").map(scrub_text).tolist()

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb = model.encode(texts, batch_size=64, show_progress_bar=True,
                   normalize_embeddings=True)
np.save(PROC / "st_minilm_scrubbed.npy", emb.astype(np.float32))
print("saved", emb.shape, "->", PROC / "st_minilm_scrubbed.npy")
