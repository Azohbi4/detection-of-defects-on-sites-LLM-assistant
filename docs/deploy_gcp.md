# Déploiement Google Cloud Platform (GCP) : Artifact Registry + Cloud Run

Une voie de déploiement moderne, serverless et économique : **Google Cloud Run**, avec l'image stockée dans **Artifact Registry**. Les commandes utilisent Google Cloud SDK (`gcloud`).

## 1. Initialisation et ressources GCP (une fois)

```bash
PROJECT_ID="chantier-defects-$(gcloud config get-value project 2>/dev/null || echo $RANDOM)"
REGION="europe-west9"                     # Paris (ou europe-west1 : Belgique)
REPO="chantier-images"
SERVICE="api-chantier"

# Authentification et sélection du projet
gcloud auth login
gcloud config set project $PROJECT_ID

# Activation des API nécessaires
gcloud services enable \
  artifactregistry.googleapis.com \
  run.googleapis.com \
  secretmanager.googleapis.com \
  cloudbuild.googleapis.com \
  logging.googleapis.com \
  monitoring.googleapis.com

# Création du registre Docker dans Artifact Registry
gcloud artifacts repositories create $REPO \
  --repository-format=docker \
  --location=$REGION \
  --description="Images Docker pour l'API de détection de défauts"
```

## 2. Construction de l'image Docker

Le modèle entraîné (`models/model.pt`) est intégré à l'image et chargé au démarrage.

Construction et push direct dans le cloud avec Cloud Build :

```bash
IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/api:v1"

gcloud builds submit --tag $IMAGE -f docker/Dockerfile .
```

## 3. Gestion des Secrets (Secret Manager)

Les clés d'API (comme `ANTHROPIC_API_KEY`) ne sont **jamais** écrites dans le code ou l'image Docker :

```bash
# Création du secret
echo -n "sk-ant-api03-..." | gcloud secrets create anthropic-api-key --data-file=-

# Attribution des droits de lecture à Cloud Run
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)")
gcloud secrets add-iam-policy-binding anthropic-api-key \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

## 4. Déploiement du Service sur Google Cloud Run

```bash
gcloud run deploy $SERVICE \
  --image $IMAGE \
  --region $REGION \
  --platform managed \
  --allow-unauthenticated \
  --port 8000 \
  --cpu 2 \
  --memory 4Gi \
  --min-instances 0 \
  --max-instances 4 \
  --set-secrets ANTHROPIC_API_KEY=anthropic-api-key:latest \
  --set-env-vars CHANTIER_QDRANT_PATH=qdrant_storage

# Récupération de l'URL publique HTTPS
gcloud run services describe $SERVICE --region $REGION --format="value(status.url)"
```

> **Avantage économique majeur :** `--min-instances 0` active le *scale-to-zero* : aucune ressource n'est facturée lorsque l'API n'est pas sollicitée (démarrage à froid de quelques secondes à la première requête).

## 5. CI/CD avec GitHub Actions

Le workflow [`.github/workflows/ci.yml`](../.github/workflows/ci.yml) automatise le cycle de vie :
- À chaque push / PR : exécution des linters (`ruff`) et des tests (`pytest`).
- À chaque push sur `main` : authentification sans clé longue durée via **Workload Identity Federation (WIF)**, build de l'image et mise à jour transparente du service Cloud Run.

Variables requises côté GitHub (`Settings` > `Secrets and variables` > `Actions`) :
- `GCP_PROJECT_ID` : identifiant du projet GCP
- `GCP_REGION` : région choisie (ex. `europe-west9`)
- `GCP_WORKLOAD_IDENTITY_PROVIDER` : provider WIF
- `GCP_SERVICE_ACCOUNT` : compte de service avec les rôles `roles/run.admin` et `roles/artifactregistry.writer`.

## 6. Monitoring & Dérive (Cloud Logging & Cloud Monitoring)

- Les logs JSON structurés émis par l'API sont automatiquement ingérés par **Google Cloud Logging**.
- Visualisation des requêtes et latences :
  ```text
  resource.type="cloud_run_revision"
  jsonPayload.event="request"
  ```
- Alerte sur dérive statistique :
  ```text
  resource.type="cloud_run_revision"
  jsonPayload.event="drift_alert"
  ```
- Endpoint de santé et métriques instantanées : `GET /metrics` (latences p50/p95, distribution des scores de détection, statut de dérive).
