"""Tableau comparatif (Markdown) des modèles à partir de leurs metrics_test.json.

Usage :
    python -m chantier_defects.eval.compare reports/baseline reports/resnet18-weighted reports/resnet18-focal
"""

import argparse
import json
from pathlib import Path


def fmt(value) -> str:
    return "—" if value is None else f"{value:.3f}"


def comparison_table(runs: dict[str, dict]) -> str:
    classes = list(next(iter(runs.values()))["per_class"])
    header = ["Modèle", "F1 macro", "Rappel macro", "PR-AUC macro", "Rappel « au moins un défaut »"]
    header += [f"Rappel {c}" for c in classes]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name, m in runs.items():
        cells = [
            name,
            fmt(m["macro_f1"]),
            fmt(m["macro_recall"]),
            fmt(m["macro_pr_auc"]),
            fmt(m["any_defect"]["recall"]),
        ]
        cells += [fmt(m["per_class"][c]["recall"]) for c in classes]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    runs = {
        d.name: json.loads((d / f"metrics_{args.split}.json").read_text(encoding="utf-8"))
        for d in args.run_dirs
    }
    print(comparison_table(runs))


if __name__ == "__main__":
    main()
