"""Interface de démonstration : upload, image annotée, rapport.

Lancement (API démarrée au préalable) :
    streamlit run src/chantier_defects/ui/streamlit_app.py
"""

import os

import httpx
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

API_URL = os.environ.get("CHANTIER_API_URL", "http://localhost:8000")
COLORS = {
    "crack": "#e63946",
    "spallation": "#f4a261",
    "efflorescence": "#2a9d8f",
    "exposed_bars": "#9b5de5",
    "corrosion_stain": "#b5651d",
}
# Copie de chantier_defects.labels.LABELS_FR : l'image de l'interface ne contient pas le paquet.
LABELS_FR = {
    "crack": "fissure",
    "spallation": "éclatement du béton",
    "efflorescence": "efflorescence",
    "exposed_bars": "armatures apparentes",
    "corrosion_stain": "traces de corrosion",
}
SEVERITY_BADGE = {"faible": "🟢 faible", "moyen": "🟠 moyen", "élevé": "🔴 élevé"}


def label_fr(defect_type: str) -> str:
    return LABELS_FR.get(defect_type, defect_type)


def load_font(size: int) -> ImageFont.FreeTypeFont:
    # La police intégrée à Pillow n'a pas les accents (« éclatement du béton ») : police système d'abord.
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def annotate(img: Image.Image, analysis: dict) -> Image.Image:
    img = img.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    # Trait et texte proportionnels à l'image : lisibles une fois l'image réduite à la largeur de la page.
    width = max(3, img.width // 200)
    font = load_font(max(14, img.width // 40))
    taken: list[tuple[int, int, int, int]] = []
    for d in analysis["detections"]:
        b, color = d["bbox"], COLORS.get(d["defect_type"], "#ffffff")
        draw.rectangle([b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]], outline=color, width=width)
        # Étiquette blanche sur fond de la couleur du défaut, décalée vers le bas si elle en chevauche une autre.
        text = label_fr(d["defect_type"])
        x, y = b["x"] + width, b["y"] + width
        box = draw.textbbox((x, y), text, font=font)
        while any(box[0] < t[2] and t[0] < box[2] and box[1] < t[3] and t[1] < box[3] for t in taken):
            y += box[3] - box[1] + width
            box = draw.textbbox((x, y), text, font=font)
        taken.append(box)
        draw.rectangle(box, fill=color)
        draw.text((x, y), text, fill="white", font=font)
    return img


def defects_table(defects: list[dict]) -> str:
    """Une ligne par type de défaut (zones regroupées), au lieu d'une ligne par zone."""
    rows: dict[str, list[str]] = {}
    for d in defects:
        rows.setdefault(d["defect_type"], []).append(d["location"])
    lines = ["| Défaut | Zones | Emplacement |", "|---|---|---|"]
    for defect_type, locations in rows.items():
        lines.append(f"| {label_fr(defect_type)} | {len(locations)} | {' ; '.join(locations)} |")
    return "\n".join(lines)


def render_report(report: dict) -> None:
    st.divider()
    st.subheader(f"Rapport — gravité {SEVERITY_BADGE[report['severity']['level']]}")

    by_image: dict[str, list[dict]] = {}
    for d in report["defects"]:
        by_image.setdefault(d["image_id"], []).append(d)
    if not by_image:
        st.markdown("Aucun défaut détecté.")
    for image_id, defects in by_image.items():
        if len(by_image) > 1:
            st.markdown(f"**{image_id}** — gravité {SEVERITY_BADGE[report['severity_by_image'][image_id]['level']]}")
        st.markdown(defects_table(defects))

    st.markdown("**À faire**")
    # En mode gabarit, les recommandations ne font que renvoyer vers une source : une seule consigne suffit.
    actions = [] if report["generated_by"] == "template" else [r["action"] for r in report["recommendations"]]
    for action in dict.fromkeys(actions):
        st.markdown(f"- {action}")
    st.markdown("- Vérifier ces défauts sur site et faire valider toute réparation par un ingénieur structure.")

    sources = {s["title"]: s["url"] for s in report["sources"]}
    if sources:
        links = [f"[{title}]({url})" if url else title for title, url in sources.items()]
        st.markdown("**Documentation** : " + " · ".join(links))
    st.caption(report["disclaimer"])


st.set_page_config(page_title="Défauts béton — inspection", layout="centered")
st.title("Détection de défauts sur béton")
st.caption("Fissures, éclatements, armatures apparentes… Aide à la décision, validation humaine requise.")

try:
    # Délai large : sur Cloud Run, l'API peut démarrer à froid (chargement du modèle et de la base RAG).
    httpx.get(f"{API_URL}/health", timeout=60).raise_for_status()
except httpx.HTTPError:
    st.error("Le service d'analyse est injoignable. Réessayer dans une minute.")
    st.stop()

uploads = st.file_uploader("Photos de béton (JPEG ou PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
context = st.text_input("Contexte (facultatif) : ouvrage, localisation, date", max_chars=2000)

if uploads and st.button("Analyser", type="primary"):
    files = [("files", (u.name, u.getvalue(), u.type)) for u in uploads]
    with st.spinner("Analyse des images…"):
        resp = httpx.post(f"{API_URL}/analyze", files=files, timeout=300)
    if resp.status_code != 200:
        st.error(resp.json().get("detail", resp.text))
        st.stop()
    st.session_state["analyses"] = resp.json()["images"]
    st.session_state.pop("report", None)

if analyses := st.session_state.get("analyses"):
    by_name = {u.name: u for u in uploads or []}
    cols = st.columns(min(2, len(analyses)))
    for i, a in enumerate(analyses):
        with cols[i % len(cols)]:
            caption = f"{SEVERITY_BADGE[a['severity']['level']]} · {len(a['detections'])} zone(s)"
            if a["image_id"] in by_name:
                st.image(annotate(Image.open(by_name[a["image_id"]]), a), caption=caption, width="stretch")
            else:
                st.markdown(caption)

    if st.button("Générer le rapport"):
        with st.spinner("Recherche documentaire et rédaction…"):
            resp = httpx.post(
                f"{API_URL}/report", json={"analyses": analyses, "context": context or None}, timeout=300
            )
        if resp.status_code == 200:
            st.session_state["report"] = resp.json()
        else:
            st.error(resp.text)

if report := st.session_state.get("report"):
    render_report(report)
