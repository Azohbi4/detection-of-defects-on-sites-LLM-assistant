"""Entraînement par transfer learning en deux temps : tête seule, puis dégel partiel.

Usage :
    python -m chantier_defects.models.train data/processed/splits.csv --arch resnet18 --loss weighted
    python -m chantier_defects.models.train data/processed/splits.csv --limit 200 --epochs-head 1 --epochs-ft 1  # essai
"""

import argparse
import copy
import random
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..data.dataset import DefectDataset, eval_transforms, train_transforms
from ..data.split import class_columns
from ..eval.metrics import (
    compute_metrics,
    sample_confidence,
    save_metrics,
    save_predictions,
    tune_thresholds,
)
from ..tracking import Tracker, flat_metrics
from .cnn import ARCHS, build_model, make_criterion, set_trainable


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.inference_mode()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, targets = [], []
    for x, y in loader:
        probs.append(torch.sigmoid(model(x.to(device))).cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(probs), np.concatenate(targets)


def train_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(x)
        n += len(x)
    return total / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("splits", type=Path)
    parser.add_argument("--arch", choices=ARCHS, default="resnet18")
    parser.add_argument("--loss", choices=("bce", "weighted", "focal"), default="weighted")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs-head", type=int, default=3)
    parser.add_argument("--epochs-ft", type=int, default=12)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-ft", type=float, default=1e-4)
    parser.add_argument(
        "--patience", type=int, default=4, help="Arrêt anticipé sur la PR-AUC macro de validation"
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Nombre max d'images par split (débogage)")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("models/model.pt"))
    parser.add_argument("--reports", type=Path, default=None)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = pick_device()
    run_name = f"{args.arch}-{args.loss}"
    reports = args.reports or Path("reports") / run_name

    df = pd.read_csv(args.splits)
    classes = class_columns(df)
    parts = {s: df[df["split"] == s] for s in ("train", "val", "test")}
    if args.limit:
        parts = {s: p.sample(min(args.limit, len(p)), random_state=args.seed) for s, p in parts.items()}

    loaders = {
        s: DataLoader(
            DefectDataset(
                p,
                classes,
                train_transforms(args.img_size) if s == "train" else eval_transforms(args.img_size),
            ),
            batch_size=args.batch_size,
            shuffle=s == "train",
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        for s, p in parts.items()
    }

    model = build_model(args.arch, len(classes), pretrained=not args.no_pretrained).to(device)
    criterion = make_criterion(
        args.loss, torch.tensor(parts["train"][classes].to_numpy(), dtype=torch.float32)
    )
    criterion = criterion.to(device)

    tracker = Tracker(enabled=not args.no_mlflow)
    with tracker.run("chantier-defects", run_name):
        tracker.params(
            {k: str(v) for k, v in vars(args).items()} | {"classes": ",".join(classes), "device": device.type}
        )

        best = {"score": -1.0, "state": None, "epoch": 0}
        epoch, stale = 0, 0
        stages = [("head", args.epochs_head, args.lr_head), ("finetune", args.epochs_ft, args.lr_ft)]
        for stage, n_epochs, lr in stages:
            params = set_trainable(model, args.arch, stage)
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(n_epochs, 1))
            for _ in range(n_epochs):
                epoch += 1
                t0 = time.time()
                loss = train_epoch(model, loaders["train"], criterion, optimizer, device)
                scheduler.step()
                val_prob, val_true = predict(model, loaders["val"], device)
                val = compute_metrics(val_true, val_prob, np.full(len(classes), 0.5), classes)
                score = val["macro_pr_auc"] or 0.0
                tracker.metrics({"train_loss": loss, "val_macro_pr_auc": score}, step=epoch)
                print(
                    f"[{stage}] epoch {epoch} loss {loss:.4f} val PR-AUC {score:.4f} ({time.time() - t0:.0f} s)"
                )
                if score > best["score"]:
                    best = {"score": score, "state": copy.deepcopy(model.state_dict()), "epoch": epoch}
                    stale = 0
                else:
                    stale += 1
                    if stage == "finetune" and stale >= args.patience:
                        print("Arrêt anticipé.")
                        break

        model.load_state_dict(best["state"])
        val_prob, val_true = predict(model, loaders["val"], device)
        thresholds = tune_thresholds(val_true, val_prob)
        test_prob, test_true = predict(model, loaders["test"], device)
        for split, (y_true, y_prob) in {"val": (val_true, val_prob), "test": (test_true, test_prob)}.items():
            metrics = compute_metrics(y_true, y_prob, thresholds, classes)
            save_metrics(metrics, reports / f"metrics_{split}.json")
            tracker.metrics(flat_metrics(metrics, split))
        save_predictions(
            parts["test"]["path"].tolist(),
            test_true,
            test_prob,
            thresholds,
            classes,
            reports / "predictions_test.csv",
        )
        print(
            f"Meilleure époque {best['epoch']} — test macro-F1 {metrics['macro_f1']:.3f}, "
            f"macro-rappel {metrics['macro_recall']:.3f}"
        )

        args.out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "kind": "cnn",
                "arch": args.arch,
                "classes": classes,
                "img_size": args.img_size,
                "thresholds": thresholds.tolist(),
                "reference_confidence": float(sample_confidence(val_prob).mean()),
                "state_dict": model.state_dict(),
                "run_name": run_name,
                "created": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            args.out,
        )
        tracker.artifact(reports / "metrics_test.json")
        print(f"Modèle -> {args.out}, métriques -> {reports}")


if __name__ == "__main__":
    main()
