"""Planches d'erreurs pour l'analyse qualitative (faux négatifs / faux positifs d'une classe).

Usage :
    python -m chantier_defects.eval.errors reports/resnet18-weighted/predictions_test.csv crack --kind fn
Les faux négatifs sont triés du plus « sûr de lui » au moins sûr (probabilité la plus basse d'abord).
"""

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402


def select_errors(df: pd.DataFrame, cls: str, kind: str) -> pd.DataFrame:
    true, pred, prob = df[f"true_{cls}"], df[f"pred_{cls}"], df[f"prob_{cls}"]
    if kind == "fn":
        return df[(true == 1) & (pred == 0)].assign(_p=prob).sort_values("_p")
    return df[(true == 0) & (pred == 1)].assign(_p=prob).sort_values("_p", ascending=False)


def contact_sheet(errors: pd.DataFrame, cls: str, kind: str, n: int, out: Path) -> None:
    errors = errors.head(n)
    cols = 6
    rows = max(1, math.ceil(len(errors) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.4), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, row) in zip(axes.flat, errors.iterrows(), strict=False):
        with Image.open(row["path"]) as img:
            ax.imshow(img.convert("RGB"))
        ax.set_title(f"p={row['_p']:.2f}\n{Path(row['path']).name}", fontsize=7)
    label = "Faux négatifs" if kind == "fn" else "Faux positifs"
    fig.suptitle(f"{label} — {cls} ({len(errors)} affichés)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("predictions", type=Path)
    parser.add_argument("cls")
    parser.add_argument("--kind", choices=("fn", "fp"), default="fn")
    parser.add_argument("--n", type=int, default=24)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.predictions)
    errors = select_errors(df, args.cls, args.kind)
    out = args.out or Path("docs/figures") / f"errors_{args.cls}_{args.kind}.png"
    print(f"{len(errors)} erreurs ({args.kind}) pour « {args.cls} » sur {len(df)} images de test")
    if len(errors):
        contact_sheet(errors, args.cls, args.kind, args.n, out)
        print(f"Planche -> {out}")


if __name__ == "__main__":
    main()
