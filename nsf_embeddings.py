"""
nsf_embeddings.py — Pre-compute and save NSF award embeddings.

One-time script: encodes all award texts from the DB with a SentenceTransformer
and writes L2-normalized float32 embeddings + matching award_ids to disk.

Usage:
    python nsf_embeddings.py \
        --db ./output/nsf_awards.db \
        --output-dir ./output
"""
from __future__ import annotations

import os
import sqlite3
import argparse
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_award_texts(db_path: str) -> tuple[list[str], list[str]]:
    """
    Returns (award_ids, texts) ordered by award.id (stable for chunked SELECT).
    Text fallback: COALESCE(abstract_narration, por_text, title).
    Only rows with non-NULL text are included.
    """
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    rows = conn.execute(
        """
        SELECT
            a.award_id,
            COALESCE(a.abstract_narration, a.por_text, a.title) AS text
        FROM award a
        WHERE COALESCE(a.abstract_narration, a.por_text, a.title) IS NOT NULL
        ORDER BY a.id
        """
    ).fetchall()
    conn.close()
    award_ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    return award_ids, texts


# ---------------------------------------------------------------------------
# SPECTER2 loader
# ---------------------------------------------------------------------------

# Models that use the AdapterHub two-part load pattern:
#   base model (allenai/specter2_base) + adapter (allenai/specter2 or aug2023refresh)
_SPECTER2_ADAPTERS: dict[str, tuple[str, str]] = {
    "allenai/specter2":                        ("allenai/specter2_base", "allenai/specter2"),
    "allenai/specter2_aug2023refresh":         ("allenai/specter2_aug2023refresh_base", "allenai/specter2_aug2023refresh"),
}


class SPECTER2Encoder:
    """
    SentenceTransformer-compatible wrapper for SPECTER2.

    Loads allenai/specter2_base directly via AutoModel (no adapters library
    required — avoids transformers/adapters version conflicts). Uses CLS-token
    pooling + L2 normalisation, matching the SPECTER2 paper's inference recipe.
    """

    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModel, AutoTokenizer

        base_id = _SPECTER2_ADAPTERS[model_name][0]   # e.g. "allenai/specter2_base"
        print(f"Loading SPECTER2 base: {base_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_id)
        self._model    = AutoModel.from_pretrained(base_id)
        self._model.eval()
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._model.to(self.device)
        self._torch = torch

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 64,
        normalize_embeddings: bool = True,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
        **kwargs,
    ) -> np.ndarray:
        import torch.nn.functional as F
        all_embs = []
        it = range(0, len(sentences), batch_size)
        if show_progress_bar:
            it = tqdm(it, unit="batch")
        for start in it:
            batch = sentences[start : start + batch_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=512, return_tensors="pt",
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with self._torch.no_grad():
                out = self._model(**enc)
            emb = out.last_hidden_state[:, 0, :]      # CLS token
            if normalize_embeddings:
                emb = F.normalize(emb, dim=-1)
            all_embs.append(emb.cpu().float().numpy())
        return np.concatenate(all_embs, axis=0)


def _encode_specter2(
    texts: list[str],
    base_model_id: str,
    adapter_id: str,
    batch_size: int,
) -> np.ndarray:
    """
    Encode with SPECTER2 using the AdapterHub two-step load:
      1. AutoAdapterModel.from_pretrained(base_model_id)
      2. model.load_adapter(adapter_id, set_active=True)
    Returns L2-normalized float32 embeddings.
    """
    import torch
    from adapters import AutoAdapterModel
    from transformers import AutoTokenizer

    print(f"Loading tokenizer: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    print(f"Loading base model: {base_model_id}")
    model = AutoAdapterModel.from_pretrained(base_model_id)

    print(f"Loading adapter: {adapter_id}")
    model.load_adapter(adapter_id, source="hf", load_as="proximity", set_active=True)
    model.eval()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    print(f"Device: {device}")

    all_embeddings = []
    print(f"Encoding {len(texts):,} texts (batch_size={batch_size}) …")
    for start in tqdm(range(0, len(texts), batch_size), unit="batch"):
        batch = texts[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            out = model(**encoded)
        # CLS token as document embedding
        emb = out.last_hidden_state[:, 0, :]
        emb = torch.nn.functional.normalize(emb, dim=-1)
        all_embeddings.append(emb.cpu().float().numpy())

    return np.concatenate(all_embeddings, axis=0)


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def compute_embeddings(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 512,
) -> np.ndarray:
    """
    Encodes texts, returns float32 L2-normalized embeddings of shape (N, D).
    Supports both SentenceTransformer models and SPECTER2 adapter models.
    """
    if model_name in _SPECTER2_ADAPTERS:
        base_id, adapter_id = _SPECTER2_ADAPTERS[model_name]
        return _encode_specter2(texts, base_id, adapter_id, batch_size)

    from sentence_transformers import SentenceTransformer

    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    print(f"Encoding {len(texts):,} texts (batch_size={batch_size}) …")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return embeddings.astype(np.float32)


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_outputs(output_dir: str, award_ids: list[str], embeddings: np.ndarray,
                 suffix: str = "") -> None:
    os.makedirs(output_dir, exist_ok=True)
    tag = f"_{suffix}" if suffix else ""
    emb_path = os.path.join(output_dir, f"embeddings{tag}.npy")
    ids_path = os.path.join(output_dir, f"award_ids{tag}.npy")
    np.save(emb_path, embeddings)
    np.save(ids_path, np.array(award_ids, dtype=object))
    print(f"Saved embeddings : {emb_path}  shape={embeddings.shape}  dtype={embeddings.dtype}")
    print(f"Saved award_ids  : {ids_path}  count={len(award_ids)}")
    size_mb = os.path.getsize(emb_path) / 1024 / 1024
    print(f"Embeddings file   : {size_mb:.1f} MB")


def load_precomputed(
    embeddings_path: str, award_ids_path: str
) -> tuple[np.ndarray, list[str]]:
    """Load previously saved embeddings (already L2-normalized float32)."""
    embeddings = np.load(embeddings_path)
    award_ids = np.load(award_ids_path, allow_pickle=True).tolist()
    return embeddings, award_ids


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify(award_ids: list[str], embeddings: np.ndarray) -> None:
    assert len(award_ids) == embeddings.shape[0], (
        f"Mismatch: {len(award_ids)} ids vs {embeddings.shape[0]} rows"
    )
    norms = np.linalg.norm(embeddings[:5], axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), f"Embeddings not normalized: {norms}"
    print(f"Verification OK — {len(award_ids):,} awards, shape={embeddings.shape}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Pre-compute NSF award embeddings")
    p.add_argument("--db", default="./output/nsf_awards.db")
    p.add_argument("--output-dir", default="./output")
    p.add_argument("--model", default="all-MiniLM-L6-v2")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--suffix", default="",
                   help="Suffix for output filenames, e.g. 'specter' → embeddings_specter.npy")
    return p.parse_args()


def main():
    args = parse_args()
    print(f"DB: {args.db}")

    award_ids, texts = load_award_texts(args.db)
    print(f"Loaded {len(award_ids):,} award texts from DB.")

    embeddings = compute_embeddings(texts, model_name=args.model, batch_size=args.batch_size)

    verify(award_ids, embeddings)
    save_outputs(args.output_dir, award_ids, embeddings, suffix=args.suffix)
    print("\nDone.")


if __name__ == "__main__":
    main()
