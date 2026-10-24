"""Classes de défauts et libellés métier.

Le problème est traité en multi-label : un patch peut contenir plusieurs défauts
(ex. armatures apparentes + traces de corrosion). « Fond » = aucun label actif.
"""

CODEBRIM_CLASSES = ["crack", "spallation", "efflorescence", "exposed_bars", "corrosion_stain"]
SDNET_CLASSES = ["crack"]

DATASET_CLASSES = {"codebrim": CODEBRIM_CLASSES, "sdnet2018": SDNET_CLASSES}

LABELS_FR = {
    "crack": "fissure",
    "spallation": "éclatement du béton",
    "efflorescence": "efflorescence",
    "exposed_bars": "armatures apparentes",
    "corrosion_stain": "traces de corrosion",
}

# Requêtes de recherche documentaire par type de défaut (FR + EN pour couvrir des sources des deux langues)
RAG_QUERIES = {
    "crack": "fissure béton causes diagnostic réparation injection crack concrete repair",
    "spallation": "éclatement épaufrure béton réparation ragréage spalling concrete repair",
    "efflorescence": "efflorescence béton infiltration eau étanchéité efflorescence leaching",
    "exposed_bars": "armatures apparentes corrosion acier enrobage réparation exposed reinforcement",
    "corrosion_stain": "traces de rouille corrosion armatures carbonatation chlorures corrosion staining",
}


def label_fr(defect_type: str) -> str:
    return LABELS_FR.get(defect_type, defect_type)
