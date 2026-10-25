"""Split train/val/test reproductible, groupé par image source et stratifié.

Usage :
    python -m chantier_defects.data.split data/processed/index.csv data/processed/splits.csv --seed 42
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from ..labels import CODEBRIM_CLASSES, SDNET_CLASSES


def class_columns(df: pd.DataFrame) -> list[str]:
    known = dict.fromkeys(CODEBRIM_CLASSES + SDNET_CLASSES)  # ordre conservé, sans doublon
    return [c for c in known if c in df.columns]


def stratification_key(df: pd.DataFrame, classes: list[str]) -> np.ndarray:
    """Clé de stratification pour du multi-label : la classe positive la plus rare du patch, sinon « fond »."""
    rarity = df[classes].sum().sort_values().index.tolist()
    key = np.full(len(df), "background", dtype=object)
    for cls in reversed(rarity):  # la plus rare écrit en dernier et l'emporte
        key[df[cls].to_numpy() == 1] = cls
    return key


def _hold_out(df: pd.DataFrame, classes: list[str], fraction: float, seed: int) -> np.ndarray:
    """Masque booléen d'environ `fraction` des lignes, sans couper de groupe."""
    n_splits = max(2, round(1 / fraction))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, held = next(sgkf.split(df, stratification_key(df, classes), groups=df["group"]))
    mask = np.zeros(len(df), dtype=bool)
    mask[held] = True
    return mask


def grouped_split(df: pd.DataFrame, val: float = 0.15, test: float = 0.15, seed: int = 42) -> pd.DataFrame:
    classes = class_columns(df)
    df = df.sort_values("path").reset_index(drop=True)  # indépendant de l'ordre du fichier d'entrée
    out = df.copy()
    out["split"] = "train"

    test_mask = _hold_out(df, classes, test, seed)
    out.loc[test_mask, "split"] = "test"

    rest = df[~test_mask]
    val_mask = _hold_out(rest.reset_index(drop=True), classes, val / (1 - test), seed)
    out.loc[rest.index[val_mask], "split"] = "val"

    check_no_leakage(out)
    return out


def check_no_leakage(df: pd.DataFrame) -> None:
    splits_per_group = df.groupby("group")["split"].nunique()
    leaking = splits_per_group[splits_per_group > 1]
    if len(leaking):
        raise ValueError(
            f"{len(leaking)} groupes présents dans plusieurs splits, ex. {leaking.index[:3].tolist()}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("index", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = grouped_split(pd.read_csv(args.index), args.val, args.test, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    classes = class_columns(df)
    summary = df.groupby("split")[classes].agg(["sum"]).droplevel(1, axis=1)
    summary.insert(0, "n_images", df["split"].value_counts())
    summary.insert(1, "n_groups", df.groupby("split")["group"].nunique())
    print(summary.loc[["train", "val", "test"]].to_string())


if __name__ == "__main__":
    main()
