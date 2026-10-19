import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from refac.models import PeptideEncoder
from train_deep import Config, batches, load_arrays, masked_spectral_angle_loss, pick_device, set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1312)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--data-dir", default="refac")
    parser.add_argument("--output", type=Path, default=Path("results/peptide_encoder_search.json"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if min(args.trials, args.epochs, args.patience) < 1:
        parser.error("trials, epochs and patience must be positive")
    device = pick_device() if args.device == "auto" else torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS is not available in this PyTorch installation or on this Mac")

    arrays = load_arrays(Config(data_dir=args.data_dir))
    allowed = np.isin(arrays["charge"], [2, 3, 4])
    train_idx = np.flatnonzero(allowed & ~arrays["is_val"])
    val_idx = np.flatnonzero(allowed & arrays["is_val"])
    if args.smoke:
        rng = np.random.default_rng(args.seed)
        train_idx = rng.permutation(train_idx)[:128]
        val_idx = rng.permutation(val_idx)[:32]
    if not len(train_idx) or not len(val_idx):
        raise ValueError("Both train and validation must contain precursor charges 2, 3 or 4")
    tensors = tuple(torch.from_numpy(arrays[key]) for key in ("tokens", "charge", "Y", "mask", "frag"))
    print(f"device={device}, train={len(train_idx)}, validation={len(val_idx)}", flush=True)
    rng = random.Random(args.seed)
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    for trial in range(1, args.trials + 1):
        params = {
            "hidden_dimensions": rng.choice([64, 128, 256]),
            "transformer_layers": rng.choice([2, 4, 6]),
            "attention_heads": rng.choice([4, 8]),
            "lr": 10 ** rng.uniform(-4, -2.5),
            "weight_decay": 10 ** rng.uniform(-6, -2),
            "batch_size": rng.choice([64, 128, 256]),
        }
        set_seed(args.seed)
        generator = torch.Generator().manual_seed(args.seed)
        model = PeptideEncoder(
            aa_vocab_size=21, mod_vocab_size=2, precursor_vocab_size=3,
            **{key: params[key] for key in ("hidden_dimensions", "transformer_layers", "attention_heads")},
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
        best_loss, best_epoch, stale = float("inf"), 0, 0
        print(f"trial {trial}: {params}", flush=True)
        for epoch in range(1, (1 if args.smoke else args.epochs) + 1):
            model.train()
            for batch in batches(train_idx, tensors, params["batch_size"], True, generator):
                tokens, charge, target, mask, frag = (tensor.to(device) for tensor in batch)
                optimizer.zero_grad(set_to_none=True)
                loss = masked_spectral_angle_loss(model(tokens, charge, frag), target, mask)
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss = 0.0
            with torch.inference_mode():
                for batch in batches(val_idx, tensors, params["batch_size"], False):
                    tokens, charge, target, mask, frag = (tensor.to(device) for tensor in batch)
                    loss = masked_spectral_angle_loss(model(tokens, charge, frag), target, mask)
                    val_loss += loss.item() * len(tokens)
            val_loss /= len(val_idx)
            if not np.isfinite(val_loss):
                raise RuntimeError(f"Non-finite validation loss in trial {trial}")
            if val_loss < best_loss:
                best_loss, best_epoch, stale = val_loss, epoch, 0
            else:
                stale += 1
            print(f"  epoch {epoch}: validation SA={1 - val_loss:.4f}", flush=True)
            if stale >= args.patience:
                break

        results.append({"params": params, "val_spectral_angle": 1 - best_loss, "best_epoch": best_epoch})
        best = max(results, key=lambda result: result["val_spectral_angle"])
        args.output.write_text(json.dumps({"seed": args.seed, "smoke": args.smoke, "best": best, "trials": results}, indent=2) + "\n")
    print(f"Best: {best}\nResults: {args.output}")


if __name__ == "__main__":
    main()
