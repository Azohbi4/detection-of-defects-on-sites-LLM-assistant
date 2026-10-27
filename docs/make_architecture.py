"""Rend docs/architecture.drawio.png : vue logique et déploiement GCP.

Usage : python docs/make_architecture.py
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = sys.argv[1] if len(sys.argv) > 1 else "docs/architecture.drawio.png"

C = {
    "client": ("#e8f1fb", "#2f6db5"),
    "api": ("#eef7ee", "#2e7d32"),
    "guard": ("#fdf0e6", "#c4621a"),
    "ext": ("#f3eefb", "#6a3fb5"),
    "gcp": ("#fbf7e6", "#a07a00"),
}

fig, ax = plt.subplots(figsize=(16, 10), dpi=150)
ax.set_xlim(0, 160)
ax.set_ylim(0, 104)
ax.axis("off")
fig.patch.set_facecolor("white")


def group(x, y, w, h, label, color):
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5",
                       fc=color, ec="#9aa3ad", lw=1.2, ls="--", zorder=0)
    )
    ax.text(x + 1.2, y + h - 1.5, label, fontsize=11, fontweight="bold", color="#3c4650", va="top", zorder=1)


def box(x, y, w, h, title, sub, kind):
    fc, ec = C[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1",
                                fc=fc, ec=ec, lw=1.8, zorder=2))
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=9.6,
            fontweight="bold", color="#1d2733", zorder=3)
    ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=8.3, color="#46525e", zorder=3)
    return (x, y, w, h)


def anchor(b, side):
    x, y, w, h = b
    return {"l": (x - 0.3, y + h / 2), "r": (x + w + 0.3, y + h / 2),
            "t": (x + w / 2, y + h + 0.3), "b": (x + w / 2, y - 0.3)}[side]


def route(points, label="", li=1):
    """Flèche orthogonale qui passe par les points donnés ; étiquette au milieu du segment li."""
    xs, ys = zip(*points, strict=True)
    ax.plot(xs, ys, color="#56606b", lw=1.4, zorder=4)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>", mutation_scale=14, lw=1.4,
                                 color="#56606b", zorder=4))
    if label:
        (x1, y1), (x2, y2) = points[li], points[li + 1]
        ax.text((x1 + x2) / 2, (y1 + y2) / 2, label, fontsize=8, color="#3c4650", ha="center", va="center",
                zorder=5, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))


def arrow(b1, s1, b2, s2, label="", dashed=False, rad=0.0, lx=0, ly=0):
    p1, p2 = anchor(b1, s1), anchor(b2, s2)
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=14, lw=1.4,
                                 color="#56606b", ls="--" if dashed else "-",
                                 connectionstyle=f"arc3,rad={rad}", zorder=4))
    if label:
        mx, my = (p1[0] + p2[0]) / 2 + lx, (p1[1] + p2[1]) / 2 + ly
        ax.text(mx, my, label, fontsize=8, color="#3c4650", ha="center", va="center", zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))


ax.text(80, 103, "Détection de défauts sur béton + assistant LLM/RAG — architecture",
        ha="center", va="top", fontsize=15, fontweight="bold", color="#1d2733")

# --- Vue logique ---
group(2, 40, 22, 54, "Client", "#f7f9fc")
group(28, 40, 98, 54, "API FastAPI — conteneur Cloud Run", "#f8fbf8")
group(130, 40, 28, 54, "Services", "#faf8fd")

ui = box(4, 60, 18, 12, "UI Streamlit", "upload, image annotée,\nrapport", "client")

an = box(31, 72, 18, 10, "POST /analyze", "images JSON/PNG", "api")
tl = box(53, 72, 20, 10, "Découpage en tuiles", "256 px, bords couverts", "api")
cnn = box(77, 72, 20, 10, "CNN multi-label", "ResNet18, BCE pondérée\nseuils F2", "api")
zn = box(101, 72, 22, 10, "Fusion en zones", "8-connexité → boîtes,\nscore, étendue", "api")

rp = box(31, 50, 18, 10, "POST /report", "détections + contexte", "api")
sev = box(53, 57, 20, 10, "Règle de gravité", "déterministe, recalculée\ncôté serveur", "api")
rag = box(53, 41, 20, 10, "Recherche RAG", "requêtes FR/EN\npar type de défaut", "api")
wr = box(77, 50, 20, 10, "Rédaction", "Claude (JSON Pydantic)\nou gabarit de secours", "api")
gf = box(101, 50, 22, 10, "Garde-fous", "sources réelles, défauts\nvérifiés, mention humaine", "guard")

qd = box(133, 40.5, 22, 11, "Qdrant", "guides & fiches\nembeddings fastembed", "ext")
cl = box(133, 66, 22, 12, "API Claude", "sortie structurée\nrepli serveur", "ext")

arrow(ui, "r", an, "l", "images", rad=-0.15, lx=-1, ly=3)
arrow(an, "r", tl, "l")
arrow(tl, "r", cnn, "l")
arrow(cnn, "r", zn, "l")
route([(112, 82.3), (112, 88), (13, 88), (13, 72.3)], "détections + gravité par image")
arrow(ui, "r", rp, "l", "détections", rad=0.15, lx=-1, ly=-3)
arrow(rp, "r", sev, "l")
arrow(rp, "r", rag, "l")
arrow(rag, "r", qd, "l", "passages", lx=26)
arrow(sev, "r", wr, "l")
arrow(rag, "r", wr, "l")
route([(87, 60.3), (87, 64), (128, 64), (128, 72), (132.7, 72)], "détections + passages", li=1)
arrow(wr, "r", gf, "l")
route([(112, 49.7), (112, 37.6), (13, 37.6), (13, 59.7)], "rapport sourcé")

# --- Vue déploiement ---
group(2, 3, 156, 32, "Déploiement Google Cloud Platform", "#fdfcf6")
gh = box(5, 12, 24, 13, "GitHub Actions", "ruff + pytest,\nbuild de l'image", "gcp")
ar = box(37, 12, 24, 13, "Artifact Registry", "image Docker\n(modèle + base RAG)", "gcp")
cr = box(69, 12, 26, 13, "Cloud Run", "API FastAPI, scale-to-zero\nQdrant embarqué", "gcp")
sm = box(103, 20, 24, 11, "Secret Manager", "ANTHROPIC_API_KEY", "gcp")
lg = box(103, 6, 24, 11, "Cloud Logging", "logs JSON, alerte\ndrift_alert", "gcp")
mo = box(133, 12, 22, 13, "Monitoring", "/metrics : latence p50/p95,\nscores, dérive", "gcp")

arrow(gh, "r", ar, "l", "push (WIF)", ly=2)
arrow(ar, "r", cr, "l", "déploiement", ly=2)
arrow(sm, "l", cr, "r", "secret", dashed=True, ly=2)
arrow(cr, "r", lg, "l", "stdout", ly=-2)
arrow(lg, "r", mo, "l", "alertes", dashed=True)

fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
print("ok", OUT)
