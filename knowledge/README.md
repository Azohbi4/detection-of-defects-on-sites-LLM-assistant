# Base documentaire du RAG

Ce dossier contient les documents **publics** indexés dans Qdrant, ainsi que `sources.yaml` qui les
décrit. Chaque recommandation du rapport doit citer l'un de ces passages.

## État actuel

- `snbi_concrete_extract.txt` est un **extrait officiel** : FHWA, *Specifications for the National Bridge
  Inventory* (mars 2022, FHWA-HIF-22-017), domaine public. Il reprend mot pour mot les définitions, les codes de
  notation de l'état des composants (tableau 20), un exemple de tablier en béton armé et la grille de gravité des
  défauts du béton (annexe C, tableau 47). Seule la mise en forme change (une ligne de tableau par paragraphe),
  pour que chaque passage indexé reste lisible. Les numéros de page sont indiqués. Le PDF source
  (`snbi_march_2022.pdf`, 6,7 Mo) n'est pas versionné : on le retélécharge depuis l'URL de `sources.yaml`.
- `fhwa_birm_concrete_defects.md` et `cerema_pathologies_beton_iqoa.md` sont des **synthèses rédigées pour la
  démonstration**, pas des extraits des guides cités. Leurs valeurs (ouvertures de fissure, pertes de section,
  normes) doivent être vérifiées dans les documents originaux. Il faut encore les remplacer par des extraits
  officiels dont la licence a été confirmée.

## Ajouter un document

1. Vérifie que la licence autorise la réutilisation (domaine public, licence ouverte, ou usage
   autorisé par l'éditeur).
2. Dépose le fichier ici (`.pdf`, `.md` ou `.txt`).
3. Ajoute une entrée dans `sources.yaml` :

```yaml
- file: nom_du_fichier.pdf
  title: Titre lisible qui sera cité dans le rapport
  url: https://… (page officielle du document)
  licence: ex. « domaine public (publication du gouvernement fédéral US) »
```

4. Réindexe : `python -m chantier_defects.llm.rag ingest knowledge/`

## Pistes de sources (à vérifier avant usage)

- **FHWA — Bridge Inspector's Reference Manual** (États-Unis) : chapitres sur les défauts du béton
  (fissuration, épaufrures, efflorescence, corrosion des armatures). Les publications du gouvernement
  fédéral américain relèvent en général du domaine public : à confirmer sur le document.
- **Cerema / ex-LCPC — guides IQOA et fascicules sur les pathologies des ouvrages en béton** : vérifier
  les conditions de réutilisation de chaque publication.
- **Fiches de pathologies du béton** publiées par des organismes professionnels : réutilisation à
  vérifier au cas par cas.

> Ne pas inventer de contenu technique pour remplir la base. Un rapport sans source vaut mieux qu'une
> recommandation appuyée sur une fiche fictive.
