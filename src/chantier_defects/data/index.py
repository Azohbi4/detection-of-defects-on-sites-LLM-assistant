"""Construit un index CSV unique (chemin, groupe, labels multi-hot) à partir des datasets bruts.

La colonne `group` identifie l'image source d'un patch : elle sert au split pour que deux patches
découpés dans la même photo ne se retrouvent jamais l'un en train, l'autre en test (fuite de données).

Usage :
    python -m chantier_defects.data.index sdnet2018 data/raw/SDNET2018 data/processed/index.csv
    python -m chantier_defects.data.index codebrim data/raw/CODEBRIM_classification_dataset data/processed/index.csv
"""

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from ..labels import CODEBRIM_CLASSES, DATASET_CLASSES

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# SDNET2018 : <surface>/<C|U><surface>/<image source>-<n° de patch>.jpg, ex. D/CD/7001-115.jpg
SDNET_SURFACES = {"D": "deck", "W": "wall", "P": "pavement"}
SDNET_NAME = re.compile(r"^(?P<source>\d+)-\d+$")

# CODEBRIM : image_0000005_crop_0000001.png -> photo source image_0000005
CODEBRIM_NAME = re.compile(r"^(?P<source>image_\d+)")
CODEBRIM_XML_TAGS = {
    "crack": "crack",
    "spallation": "spallation",
    "efflorescence": "efflorescence",
    "exposedbars": "exposed_bars",
    "corrosionstain": "corrosion_stain",
}


def _images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


def index_sdnet2018(root: Path) -> pd.DataFrame:
    rows = []
    for path in _images(root):
        label_dir = path.parent.name.upper()  # CD, UD, CW, UW, CP, UP
        if len(label_dir) != 2 or label_dir[0] not in "CU" or label_dir[1] not in SDNET_SURFACES:
            continue
        surface = SDNET_SURFACES[label_dir[1]]
        match = SDNET_NAME.match(path.stem)
        source = match["source"] if match else path.stem
        rows.append(
            {
                "path": path.as_posix(),
                "dataset": "sdnet2018",
                "surface": surface,
                "group": f"{surface}_{source}",
                "crack": int(label_dir[0] == "C"),
            }
        )
    return pd.DataFrame(rows)


def _parse_codebrim_xml(xml_path: Path) -> dict[str, dict[str, int]]:
    """Lit metadata/*.xml : un élément par image (attribut name), un enfant 0/1 par défaut."""
    labels = {}
    for elem in ET.parse(xml_path).getroot().iter():
        name = elem.get("name")
        if not name:
            continue
        values = {}
        for child in elem:
            key = CODEBRIM_XML_TAGS.get(child.tag.lower().replace("_", ""))
            if key:
                values[key] = int(float((child.text or "0").strip()))
        labels[name] = values
    return labels


def index_codebrim(root: Path) -> pd.DataFrame:
    labels: dict[str, dict[str, int]] = {}
    for xml_path in root.rglob("*.xml"):
        labels.update(_parse_codebrim_xml(xml_path))
    if not labels:
        raise FileNotFoundError(f"Aucun fichier de métadonnées XML trouvé sous {root}")

    rows = []
    for path in _images(root):
        if path.name not in labels:
            continue
        match = CODEBRIM_NAME.match(path.stem)
        row = {
            "path": path.as_posix(),
            "dataset": "codebrim",
            "surface": "bridge",
            "group": match["source"] if match else path.stem,
            # Le split officiel est conservé à titre indicatif ; on refait un split groupé.
            "official_split": next((p for p in ("train", "val", "test") if p in path.parts), ""),
        }
        row.update({c: labels[path.name].get(c, 0) for c in CODEBRIM_CLASSES})
        rows.append(row)
    # Une même image peut apparaître dans plusieurs sous-dossiers (version équilibrée / complète)
    return pd.DataFrame(rows).drop_duplicates(subset="path")


INDEXERS = {"sdnet2018": index_sdnet2018, "codebrim": index_codebrim}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset", choices=sorted(INDEXERS))
    parser.add_argument("root", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()

    df = INDEXERS[args.dataset](args.root)
    if df.empty:
        raise SystemExit(f"Aucune image indexée sous {args.root} : vérifie l'arborescence attendue.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    classes = DATASET_CLASSES[args.dataset]
    print(f"{len(df)} images, {df['group'].nunique()} images sources -> {args.out}")
    print("Positifs par classe :", df[classes].sum().to_dict())
    print("Fond (aucun défaut) :", int((df[classes].sum(axis=1) == 0).sum()))


if __name__ == "__main__":
    main()
