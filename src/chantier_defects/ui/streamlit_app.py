"""Interface de démonstration : upload, image annotée, rapport.

Lancement (API démarrée au préalable) :
    streamlit run src/chantier_defects/ui/streamlit_app.py
"""

import os

import httpx
import streamlit as st
from PIL import Image, ImageDraw

API_URL = os.environ.get("CHANTIER_API_URL", "http://localhost:8000")
COLORS = {
    "crack": "#e63946",
    "spallation": "#f4a261",
    "efflorescence": "#2a9d8f",
    "exposed_bars": "#9b5de5",
    "corrosion_stain": "#b5651d",
}
SEVERITY_BADGE = {"faible": "🟢 faible", "moyen": "🟠 moyen", "élevé": "🔴 élevé"}


def annotate(img: Image.Image, analysis: dict) -> Image.Image:
    img = img.convert("RGB").copy()
    draw = ImageDraw.Draw(img)
    width = max(2, img.width // 300)
    for d in analysis["detections"]:
        b, color = d["bbox"], COLORS.get(d["defect_type"], "#ffffff")
        draw.rectangle([b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]], outline=color, width=width)
        draw.text((b["x"] + 4, b["y"] + 4), f"{d['defect_type']} {d['score']:.2f}", fill=color)
    return img


def render_report(report: dict) -> None:
    st.subheader(f"Rapport — gravité {SEVERITY_BADGE[report['severity']['level']]}")
    st.warning(report["disclaimer"])
    st.markdown(f"**Résumé.** {report['summary']}")
    with st.expander("Justification de la gravité (règle déterministe)"):
        for reason in report["severity"]["reasons"]:
            st.markdown(f"- {reason}")
    st.markdown("**Défauts**")
    for d in report["defects"]:
        st.markdown(f"- `{d['image_id']}` — {d['defect_type']} : {d['location']}")
    st.markdown("**Actions recommandées**")
    if not report["recommendations"]:
        st.markdown("_Aucune recommandation sourcée disponible._")
    for r in report["recommendations"]:
        st.markdown(f"- {r['action']} [{', '.join(r['source_ids'])}]  \n  _{r['rationale']}_")
    st.markdown(f"**Incertitude.** {report['uncertainty']}")
    if report["sources"]:
        st.markdown("**Sources**")
        for s in report["sources"]:
            link = f" — {s['url']}" if s["url"] else ""
            st.markdown(f"- **{s['id']}** {s['title']}{link}")
    if report["guardrail_notes"]:
        with st.expander("Notes des garde-fous"):
            for note in report["guardrail_notes"]:
                st.markdown(f"- {note}")
    st.caption(f"Rédigé par : {report['generated_by']}")


st.set_page_config(page_title="Défauts béton — inspection", layout="wide")
st.title("Détection de défauts sur béton")
st.caption("Fissures, éclatements, armatures apparentes… Aide à la décision, validation humaine requise.")

try:
    # Délai large : sur Cloud Run, l'API peut démarrer à froid (chargement du modèle et de la base RAG).
    health = httpx.get(f"{API_URL}/health", timeout=60).json()
    st.sidebar.success(f"API OK — modèle : {health['model_version']}")
except httpx.HTTPError:
    st.sidebar.error(f"API injoignable sur {API_URL}")
    st.stop()

context = st.sidebar.text_area("Contexte (ouvrage, localisation, date)", max_chars=2000)
uploads = st.file_uploader("Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploads and st.button("Analyser", type="primary"):
    files = [("files", (u.name, u.getvalue(), u.type)) for u in uploads]
    resp = httpx.post(f"{API_URL}/analyze", files=files, timeout=300)
    if resp.status_code != 200:
        st.error(resp.text)
        st.stop()
    st.session_state["analyses"] = resp.json()["images"]
    st.session_state.pop("report", None)

if analyses := st.session_state.get("analyses"):
    by_name = {u.name: u for u in uploads or []}
    cols = st.columns(min(3, len(analyses)))
    for i, a in enumerate(analyses):
        with cols[i % len(cols)]:
            if a["image_id"] in by_name:
                st.image(annotate(Image.open(by_name[a["image_id"]]), a), width="stretch")
            st.markdown(
                f"**{a['image_id']}** — {SEVERITY_BADGE[a['severity']['level']]}, "
                f"{len(a['detections'])} zone(s), confiance moyenne {a['mean_confidence']:.2f}"
            )

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
