# Déploiement Azure : Container Registry + Container Apps

Une seule voie de déploiement, pour tenir dans le temps imparti : **Azure Container Apps**, avec
l'image stockée dans **Azure Container Registry (ACR)**. Les commandes utilisent Azure CLI (`az`).

## 1. Ressources (une fois)

```bash
RG=rg-chantier-defects
LOC=francecentral
ACR=chantierdefects$RANDOM          # nom globalement unique, minuscules
ENV=env-chantier
APP=api-chantier

az login
az group create -n $RG -l $LOC
az acr create -g $RG -n $ACR --sku Basic
az containerapp env create -g $RG -n $ENV -l $LOC   # crée aussi l'espace Log Analytics
```

## 2. Image

Le modèle (`models/model.pt`) est intégré à l'image et chargé au démarrage. Il n'est pas versionné
dans Git : la CI le récupère depuis un stockage (voir § 5) ou on construit l'image en local.

```bash
az acr build -r $ACR -t chantier-api:v1 -f docker/Dockerfile .
```

## 3. Qdrant

Pour une démo, Qdrant tourne comme seconde Container App interne, avec un volume Azure Files pour que
l'index survive aux redémarrages. Variante plus simple : réindexer au démarrage, la base étant petite.

```bash
az containerapp create -g $RG -n qdrant --environment $ENV \
  --image qdrant/qdrant:v1.20.0 --target-port 6333 --ingress internal \
  --min-replicas 1 --max-replicas 1
```

## 4. API

Les secrets ne sont **jamais** dans le dépôt : ils sont stockés comme secrets de la Container App
(ou référencés depuis Azure Key Vault avec une identité managée).

```bash
az containerapp create -g $RG -n $APP --environment $ENV \
  --image $ACR.azurecr.io/chantier-api:v1 --registry-server $ACR.azurecr.io \
  --target-port 8000 --ingress external --min-replicas 0 --max-replicas 2 \
  --cpu 2 --memory 4Gi \
  --secrets anthropic-key=<CLE> \
  --env-vars ANTHROPIC_API_KEY=secretref:anthropic-key CHANTIER_QDRANT_URL=http://qdrant

az containerapp show -g $RG -n $APP --query properties.configuration.ingress.fqdn -o tsv
```

Variante Key Vault : `--secrets anthropic-key=keyvaultref:https://<kv>.vault.azure.net/secrets/anthropic-key,identityref:<id>`.

`--min-replicas 0` permet de ne rien payer au repos (démarrage à froid de quelques secondes).
Pour économiser les crédits après la démo : `az group delete -n $RG`, en gardant des captures.

## 5. CI/CD

`.github/workflows/ci.yml` :

- à chaque push ou PR : lint (ruff) et tests (pytest) ;
- sur `main`, si la variable `ACR_NAME` est définie : connexion à Azure par OIDC (sans mot de passe
  stocké), build, push vers ACR, puis mise à jour de la Container App.

Configuration côté GitHub :

| Type | Nom | Valeur |
|---|---|---|
| Secret | `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID` | identité fédérée (`az ad app federated-credential create`) avec le rôle `AcrPush` et `Contributor` sur le groupe |
| Variable | `ACR_NAME`, `RESOURCE_GROUP`, `CONTAINER_APP` | noms des ressources |
| Variable | `MODEL_URL` | URL (SAS en lecture, stockée en secret si besoin) du `model.pt` à intégrer à l'image |

## 6. Monitoring

- Les logs JSON de l'API arrivent dans Log Analytics (table `ContainerAppConsoleLogs_CL`).
- Latence et confiance par requête : événements `"event": "request"`.
- Alerte de dérive : règle d'alerte Azure Monitor sur la requête KQL

```kql
ContainerAppConsoleLogs_CL
| where Log_s has '"event": "drift_alert"'
```

- Vue instantanée : `GET /metrics` (p50/p95 de latence, histogramme des scores, état de dérive).
