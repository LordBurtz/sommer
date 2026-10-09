"""train 05 on fixed split

  checkpoints/<tag>.pt            best checkpoint (by val spectral angle)
  results/runs.jsonl              one JSON record appended per run
  results/<tag>_curve.png         train loss / val SA per epoch
Usage:
  uv run python train_deep.py                 # full run
  uv run python train_deep.py --smoke         # ~2k spectra, 1 epoch
  uv run python train_deep.py --data-dir /content/drive/MyDrive/sommer
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import common as c


@dataclass
class Config:
    #paths (relative paths:data_dir / out_dir)
    data_dir: str = "refac"
    data_file: str = "clean_spectra_split.parquet"
    out_dir: str = "."
    cache_file: str = "cache/deep_arrays_refac_trainval.npz"
    tag: str = "deep_baseline"
    #model
    emb: int = 64
    hidden: int = 128
    head: str = "pooled"
    use_frag: bool = False
    activation: str = "softplus"
    #optimisation (early stopping)
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 512
    max_epochs: int = 50
    patience: int = 5
    lr_factor: float = 0.5
    lr_patience: int = 1
    seed: int = 1312
    smoke: bool = False
    smoke_n: int = 2000


FRAGS = ("2xIT_2xHCD-1h", "DDA-1h")
CACHE_KEYS = ("tokens", "charge", "frag", "Y", "mask", "is_val")
ACTIVATIONS = {"softplus": torch.nn.functional.softplus, "sigmoid": torch.sigmoid}


def place_bonds(bond_out: torch.Tensor, n_res: torch.Tensor) -> torch.Tensor:
    n = torch.arange(1, bond_out.size(1) + 1, device=bond_out.device).unsqueeze(0)
    n_res = n_res.unsqueeze(1)
    valid = n < n_res
    b = bond_out[..., 0]
    y = bond_out[..., 1].gather(1, (n_res - 1 - n).clamp(min=0))
    return (torch.stack([b, y], dim=-1) * valid.unsqueeze(-1)).flatten(1)


class PrositLike(nn.Module):
    def __init__(self, vocab=c.VOCAB_SIZE, n_charge=6, emb=64, hidden=128, out=c.VECTOR_DIM, head="pooled",
                 n_frag=0, activation="softplus"):
        super().__init__()
        if head not in ("pooled", "bond"):
            raise ValueError(f"unknown head {head!r}, expected 'pooled' or 'bond'")
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}, expected one of {sorted(ACTIVATIONS)}")
        self.head_type = head
        self.final_act = ACTIVATIONS[activation]
        self.tok_emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.charge_emb = nn.Embedding(n_charge + 1, 16)
        self.frag_emb = nn.Embedding(n_frag, 16) if n_frag else None
        self.gru = nn.GRU(emb + 16 + (16 if n_frag else 0), hidden, batch_first=True,
                          bidirectional=True, num_layers=2, dropout=0.2)
        if head == "pooled":
            self.attn = nn.Linear(2 * hidden, 1)
            self.head = nn.Sequential(
                nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(hidden, out),
            )
        else:
            self.bond_head = nn.Sequential(
                nn.Linear(4 * hidden, hidden), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(hidden, c.N_ION_TYPES),
            )

    def forward(self, tokens, charge, frag=None):
        pad_mask = (tokens != 0).float().unsqueeze(-1)        # (B, L, 1)
        x = self.tok_emb(tokens)                              # (B, L, emb)
        cz = self.charge_emb(charge).unsqueeze(1).expand(-1, x.size(1), -1)
        x = torch.cat([x, cz], dim=-1)
        if self.frag_emb is not None:
            if frag is None:
                raise ValueError("model was built with the acquisition method as input, but frag is None")
            fz = self.frag_emb(frag).unsqueeze(1).expand(-1, x.size(1), -1)
            x = torch.cat([x, fz], dim=-1)
        h, _ = self.gru(x)                                    # (B, L, 2H)
        if self.head_type == "bond":
            bonds = torch.cat([h[:, :-1], h[:, 1:]], dim=-1)
            out = self.final_act(self.bond_head(bonds))
            return place_bonds(out, (tokens != 0).sum(dim=1))
        # Attention pooling over valid (non-pad) positions.
        score = self.attn(h).masked_fill(pad_mask == 0, -1e9)
        w = torch.softmax(score, dim=1)
        pooled = (w * h).sum(dim=1)                           # (B, 2H)
        return self.final_act(self.head(pooled))              # (B, out) in [0,1]


def masked_spectral_angle_loss(pred, target, mask, eps=1e-8):
    pred = pred * mask
    target = target * mask
    dot = (pred * target).sum(dim=1)
    pn = pred.norm(dim=1)
    tn = target.norm(dim=1)
    cos = (dot / (pn * tn + eps)).clamp(-1 + 1e-7, 1 - 1e-7)
    return (2.0 * torch.arccos(cos) / np.pi).mean()


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                      stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                        text=True, stderr=subprocess.DEVNULL).strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def resolve(base: str, p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else Path(base) / p


def load_arrays(cfg: Config) -> dict[str, np.ndarray]:
    """Encoded train+val arrays (tokens, charge, Y, mask, split) cached to .npz."""
    cache = resolve(cfg.out_dir, cfg.cache_file)
    if cache.exists():
        with np.load(cache) as z:
            missing = [k for k in CACHE_KEYS if k not in z.files]
            if not missing:
                print(f"loading cached arrays from {cache}")
                return {k: z[k] for k in CACHE_KEYS}
        print(f"cache {cache} lacks {missing}, rebuilding")

    t0 = time.time()
    df = pd.read_parquet(resolve(cfg.data_dir, cfg.data_file),
                         columns=["raw_file", "scan_number", "peptide_sequence", "precursor_charge",
                                  "frag", "matched_ions", "intensities_raw", "split"],
                         filters=[("split", "in", ["train", "val"])])          # test is never loaded
    assert set(df["split"]) == {"train", "val"}
    df = df.sort_values(["raw_file", "scan_number"]).reset_index(drop=True)
    unknown = set(df["frag"]) - set(FRAGS)
    if unknown:
        raise ValueError(f"unknown acquisition methods {sorted(unknown)}, expected {FRAGS}")

    arrays = {
        "tokens": np.stack([c.encode_tokens(s) for s in df["peptide_sequence"]]).astype(np.int64),
        "charge": df["precursor_charge"].to_numpy(dtype=np.int64),
        "frag": df["frag"].map({f: i for i, f in enumerate(FRAGS)}).to_numpy(dtype=np.int64),
        "Y": np.stack([c.ions_to_vector(i, v, normalize=True)
                       for i, v in zip(df["matched_ions"], df["intensities_raw"])]).astype(np.float32),
        "mask": np.stack([c.valid_slot_mask(s) for s in df["peptide_sequence"]]).astype(np.float32),
        "is_val": (df["split"] == "val").to_numpy(),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, **arrays)
    print(f"encoded {len(df):,} spectra in {time.time() - t0:.0f}s -> {cache}")
    return arrays


def batches(idx: np.ndarray, tensors, batch_size: int, shuffle: bool, gen: torch.Generator | None = None):
    if shuffle:
        idx = idx[torch.randperm(len(idx), generator=gen).numpy()]
    for s in range(0, len(idx), batch_size):
        b = torch.from_numpy(idx[s:s + batch_size]).to(tensors[0].device)
        yield tuple(t[b] for t in tensors)


@torch.inference_mode()
def predict(model, idx, tensors, batch_size) -> np.ndarray:
    model.eval()
    return np.concatenate([model(tok, z, f).float().cpu().numpy()
                           for tok, z, _, _, f in batches(idx, tensors, batch_size, shuffle=False)])


def evaluate(pred: np.ndarray, Y: np.ndarray, mask: np.ndarray, charge: np.ndarray) -> dict:
    out = c.evaluate_vectors(Y, pred, mask.astype(bool))
    res = {"val_spectral_angle": out["spectral_angle"], "val_pearson": out["pearson"], "by_charge": {}}
    for z in (2, 3, 4):
        sel = charge == z
        if sel.any():
            r = c.evaluate_vectors(Y[sel], pred[sel], mask[sel].astype(bool))
            res["by_charge"][str(z)] = {"spectral_angle": r["spectral_angle"],
                                        "pearson": r["pearson"], "n": r["n"]}
        else:
            res["by_charge"][str(z)] = {"spectral_angle": None, "pearson": None, "n": 0}
    return res


def val_fingerprint(A: dict[str, np.ndarray], va_idx: np.ndarray) -> str:
    h = hashlib.sha1()
    for k in ("tokens", "charge", "Y"):
        h.update(np.ascontiguousarray(A[k][va_idx]).tobytes())
    return h.hexdigest()[:12]


def plot_curve(history: list[dict], best_epoch: int, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ep = [h["epoch"] for h in history]
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(ep, [h["train_loss"] for h in history], color="#4c72b0", marker="o", ms=3, label="train loss (1 - SA)")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss", color="#4c72b0")
    ax2 = ax1.twinx()
    ax2.plot(ep, [h["val_sa"] for h in history], color="#dd8452", marker="o", ms=3, label="val SA")
    ax2.set_ylabel("val spectral angle", color="#dd8452")
    ax2.axvline(best_epoch, color="grey", ls="--", lw=1)
    ax2.annotate(f"best: epoch {best_epoch}", xy=(best_epoch, ax2.get_ylim()[0]),
                 xytext=(4, 6), textcoords="offset points", color="grey", fontsize=8)
    ax1.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run(cfg: Config) -> dict:
    if cfg.head != "pooled":
        cfg.tag = f"{cfg.tag}_{cfg.head}"
    if cfg.use_frag:
        cfg.tag = f"{cfg.tag}_frag"
    if cfg.activation != "softplus":
        cfg.tag = f"{cfg.tag}_{cfg.activation}"
    if cfg.smoke:
        cfg.max_epochs, cfg.tag = 1, cfg.tag + "_smoke"
    set_seed(cfg.seed)
    device = pick_device()
    if device.type == "cpu":
        torch.set_num_threads(os.cpu_count())
    print(f"device: {device}")

    A = load_arrays(cfg)
    tr_idx, va_idx = np.where(~A["is_val"])[0], np.where(A["is_val"])[0]
    if cfg.smoke:
        rng = np.random.default_rng(cfg.seed)
        tr_idx = np.sort(rng.choice(tr_idx, cfg.smoke_n, replace=False))
        va_idx = np.sort(rng.choice(va_idx, cfg.smoke_n // 4, replace=False))
    print(f"n_train={len(tr_idx):,}  n_val={len(va_idx):,}")

    tensors = tuple(torch.from_numpy(np.array(A[k])).to(device) for k in ("tokens", "charge", "Y", "mask", "frag"))
    Yv, Mv, Zv = A["Y"][va_idx], A["mask"][va_idx], A["charge"][va_idx]

    model = PrositLike(emb=cfg.emb, hidden=cfg.hidden, head=cfg.head,
                       n_frag=len(FRAGS) if cfg.use_frag else 0, activation=cfg.activation).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {cfg.tag}  params={n_params:,}")
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=cfg.lr_factor, patience=cfg.lr_patience)
    gen = torch.Generator().manual_seed(cfg.seed)

    ckpt_path = resolve(cfg.out_dir, f"checkpoints/{cfg.tag}.pt")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    history, best_sa, best_epoch, bad = [], -1.0, 0, 0
    t_start = time.time()
    for ep in range(1, cfg.max_epochs + 1):
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for tok, z, y, m, f in batches(tr_idx, tensors, cfg.batch_size, shuffle=True, gen=gen):
            opt.zero_grad(set_to_none=True)
            loss = masked_spectral_angle_loss(model(tok, z, f), y, m)
            loss.backward()
            opt.step()
            tot += loss.item() * len(tok); n += len(tok)
        train_loss = tot / n
        val_sa = c.evaluate_vectors(Yv, predict(model, va_idx, tensors, 4096), Mv.astype(bool))["spectral_angle"]
        sched.step(1 - val_sa)
        lr_now = opt.param_groups[0]["lr"]
        history.append({"epoch": ep, "train_loss": train_loss, "val_sa": val_sa, "lr": lr_now,
                        "seconds": time.time() - t0})
        improved = val_sa > best_sa
        if improved:
            best_sa, best_epoch, bad = val_sa, ep, 0
            torch.save({"state_dict": model.state_dict(), "config": asdict(cfg), "epoch": ep,
                        "val_sa": val_sa}, ckpt_path)
        else:
            bad += 1
        print(f"epoch {ep:2d}  train_loss={train_loss:.4f}  val SA={val_sa:.4f}  lr={lr_now:.1e}  "
              f"{time.time() - t0:.0f}s{'  *' if improved else ''}", flush=True)
        if bad >= cfg.patience:
            print(f"early stop: no val SA improvement for {cfg.patience} epochs")
            break

    #metrics from best checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["state_dict"])
    metrics = evaluate(predict(model, va_idx, tensors, 4096), Yv, Mv, Zv)

    record = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "model": "PrositLike",
        "tag": cfg.tag,
        "hyperparameters": {k: getattr(cfg, k) for k in
                            ("emb", "hidden", "head", "use_frag", "activation", "lr", "weight_decay", "batch_size", "max_epochs",
                             "patience", "lr_factor", "lr_patience", "seed")}
                           | {"loss": "masked_spectral_angle", "scheduler": "ReduceLROnPlateau(val SA)"},
        "n_params": n_params,
        "device": str(device),
        "n_train": int(len(tr_idx)),
        "n_val": int(len(va_idx)),
        "val_fingerprint": val_fingerprint(A, va_idx),
        "epochs_trained": len(history),
        "best_epoch": best_epoch,
        **metrics,
        "train_minutes": round((time.time() - t_start) / 60, 2),
        "checkpoint": str(ckpt_path),
        "history": history,
    }
    runs = resolve(cfg.out_dir, "results/runs.jsonl" if not cfg.smoke else "results/smoke_runs.jsonl")
    runs.parent.mkdir(parents=True, exist_ok=True)
    with open(runs, "a") as f:
        f.write(json.dumps(record) + "\n")
    plot_curve(history, best_epoch, resolve(cfg.out_dir, f"results/{cfg.tag}_curve.png"),
               f"PrositLike baseline: best val SA {metrics['val_spectral_angle']:.4f} @ epoch {best_epoch}")

    print(f"\nbest epoch {best_epoch}: val SA={metrics['val_spectral_angle']:.4f}  "
          f"Pearson={metrics['val_pearson']:.4f}")
    for z, r in metrics["by_charge"].items():
        print(f"  z={z}: SA={r['spectral_angle']:.4f}  (n={r['n']:,})")
    print(f"logged to {runs}")
    return record


def main(argv=None):
    ap = argparse.ArgumentParser()
    for name, default in asdict(Config()).items():
        flag = "--" + name.replace("_", "-")
        if isinstance(default, bool):
            ap.add_argument(flag, action="store_true")
        else:
            ap.add_argument(flag, type=type(default), default=default)
    return run(Config(**vars(ap.parse_args(argv))))


if __name__ == "__main__":
    main()
