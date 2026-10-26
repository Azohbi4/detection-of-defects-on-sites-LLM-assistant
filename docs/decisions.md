# Choix de conception

Chaque choix est noté avec son alternative et sa justification, pour pouvoir le défendre en revue ou
en entretien.

## 1. Le LLM rédige, il ne décide pas de la gravité

- **Choix** : gravité calculée par une règle déterministe (`severity.py`), recalculée côté serveur à
  chaque `/report`. Le LLM reçoit la gravité en entrée et ne peut pas la modifier.
- **Alternative écartée** : demander au LLM d'évaluer la gravité.
- **Pourquoi** : une règle explicite est reproductible (même entrée, même sortie), testable
  unitairement et auditable par un expert qui peut contester un seuil. Un LLM peut varier d'un appel à
  l'autre et ne sait pas justifier un seuil. Recalculer côté serveur empêche aussi un client de
  soumettre une gravité falsifiée.

## 2. Classification par tuiles plutôt que détection d'objets

- **Choix** : CNN multi-label sur des tuiles de 256 px, puis fusion des tuiles voisines en zones.
- **Alternative** : détection par boîtes englobantes (YOLO).
- **Pourquoi** : CODEBRIM et SDNET2018 fournissent surtout des labels au niveau du patch. Annoter des
  boîtes coûte cher, et une fissure, longue et fine, se prête mal à une boîte. La fusion par tuiles
  donne une localisation suffisante pour le terrain (« zone haut-gauche, 12 % de l'image »).
  ultralytics est de plus sous AGPL-3.0, une contrainte forte pour un service déployé.

## 3. Seuils choisis pour maximiser le F2

- **Choix** : seuil par classe qui maximise le F2 sur la validation, puis appliqué tel quel au test.
- **Pourquoi** : un défaut manqué (faux négatif) peut laisser se dégrader un ouvrage, alors qu'une fausse
  alerte coûte un contrôle visuel. Le F2 pondère le rappel deux fois plus que la précision. Fixer les
  seuils sur la validation évite de sur-ajuster sur le test.

## 4. Split groupé par photo source

- **Choix** : `StratifiedGroupKFold` sur l'identifiant de photo source.
- **Pourquoi** : des patches voisins issus d'une même photo partagent éclairage, texture et parfois le
  même défaut. Les répartir entre train et test gonflerait artificiellement les scores.

## 5. Garde-fous sur la sortie du LLM

- Sortie JSON contrainte (structured outputs) puis validée par Pydantic (`ReportDraft`).
- Une recommandation doit citer au moins un passage effectivement récupéré : les identifiants
  inventés sont retirés, et une recommandation sans source valide est supprimée.
- Un défaut que le modèle de vision n'a pas détecté est retiré du rapport. Si le résumé cite un type de
  défaut non détecté (hors mentions niées comme « aucune fissure »), il est remplacé par le résumé
  déterministe.
- Les passages dont la similarité est inférieure à `rag_min_score` (0,40, calibré sur `knowledge/`) sont
  écartés : mieux vaut aucune source qu'une source hors sujet.
- Les passages et le contexte saisi par l'utilisateur sont présentés au LLM comme des données
  (balises dédiées, consigne explicite, contenu échappé pour qu'un document ne puisse pas fermer une
  balise), ce qui limite l'injection de consignes.
- Appel borné : 120 s par tentative et une seule relance, pour rester sous le délai de l'interface.
  Repli serveur en cas de refus (`fallbacks: "default"`, beta `server-side-fallback-2026-07-01`).
- Chaque suppression est tracée dans `guardrail_notes`, ce qui rend les garde-fous visibles et non
  silencieux.
- Mode dégradé : sans clé API, ou si l'appel échoue, un gabarit déterministe produit le rapport.
  L'API reste ainsi disponible.

## 6. Embeddings locaux (fastembed) et Qdrant

- Embeddings multilingues calculés en local (ONNX) : pas de second fournisseur d'API, pas de coût par
  requête, et les documents ne quittent pas l'infrastructure au moment de l'indexation.
- Qdrant tourne en conteneur avec docker-compose en local. Les tests et le développement sans Docker
  peuvent utiliser le mode embarqué (`CHANTIER_QDRANT_PATH`).

## 7. Indicateur de dérive

- **Choix** : surveiller la confiance moyenne `max(p, 1-p)` de la classe la plus incertaine, comparée à
  sa valeur sur la validation (stockée dans le checkpoint).
- **Pourquoi** : sans vérité terrain en production, une baisse durable de confiance est le signal le
  plus simple de photos qui s'éloignent de la distribution d'entraînement (autre type de béton, nuit,
  appareil différent).
