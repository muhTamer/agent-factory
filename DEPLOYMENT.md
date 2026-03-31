# Deployment Guide — Azure Container Apps

This guide explains how to deploy Agent Factory so people can test it online via a public URL.

## Architecture

```
┌─────────────┐        ┌──────────────────┐       ┌───────────────────┐
│   Browser    │──────▶ │  Frontend (Next)  │─────▶│  Backend (FastAPI) │
│              │  :443  │  Container App    │ :8080 │  Container App     │──▶ Azure OpenAI
└─────────────┘        └──────────────────┘       └───────────────────┘
```

Both services run as **Azure Container Apps** with auto-scaling (0–3 replicas), built from Docker images stored in **Azure Container Registry (ACR)**.

## Prerequisites

- [Azure CLI](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli) installed
- [Docker](https://docs.docker.com/get-docker/) installed
- An Azure subscription with an Azure OpenAI resource
- A GitHub repository (for CI/CD)

## Quick Start — Local (Docker Compose)

Test the full stack locally before deploying:

```bash
# 1. Copy and fill in your environment variables
cp .env.example .env
# Edit .env with your Azure OpenAI credentials

# 2. Build and run both services
docker compose up --build

# 3. Open the app
#    Frontend: http://localhost:3000
#    Backend:  http://localhost:8080/health
```

## Deploy to Azure

### Step 1: Create Azure Infrastructure (one-time)

```bash
chmod +x scripts/azure-setup.sh
./scripts/azure-setup.sh
```

This creates:
- Resource group (`agent-factory-rg`)
- Azure Container Registry (`agentfactoryacr`)
- Container Apps environment
- Backend & frontend container apps
- GitHub Actions service principal

### Step 2: Configure Secrets

**Azure OpenAI secrets** (on the backend container app):

```bash
az containerapp secret set \
  --name agent-factory-backend \
  --resource-group agent-factory-rg \
  --secrets \
    azure-openai-endpoint=https://YOUR_RESOURCE.openai.azure.com/ \
    azure-openai-api-key=YOUR_KEY \
    azure-openai-deployment=gpt-4o-mini
```

**GitHub repository secret** — add the JSON output from the setup script as `AZURE_CREDENTIALS` in your GitHub repo settings under *Settings → Secrets and variables → Actions*.

### Step 3: Deploy

Push to `main` to trigger the CI/CD pipeline:

```bash
git push origin main
```

Or trigger manually: *GitHub → Actions → Build & Deploy to Azure Container Apps → Run workflow*.

## URLs

After deployment, your apps will be available at:

| Service  | URL |
|----------|-----|
| Frontend | `https://agent-factory-frontend.<region>.azurecontainerapps.io` |
| Backend  | `https://agent-factory-backend.<region>.azurecontainerapps.io` |

## Custom Domain (optional)

To use a custom domain like `demo.yourdomain.com`:

```bash
# Add custom domain to the frontend app
az containerapp hostname add \
  --name agent-factory-frontend \
  --resource-group agent-factory-rg \
  --hostname demo.yourdomain.com

# Bind a managed certificate (free SSL)
az containerapp hostname bind \
  --name agent-factory-frontend \
  --resource-group agent-factory-rg \
  --hostname demo.yourdomain.com \
  --environment agent-factory-env \
  --validation-method CNAME
```

## Scaling & Cost

| Setting | Backend | Frontend |
|---------|---------|----------|
| Min replicas | 0 (scale to zero) | 0 |
| Max replicas | 3 | 3 |
| CPU | 1.0 vCPU | 0.5 vCPU |
| Memory | 2 GB | 1 GB |

**Estimated cost**: ~$0 when idle (scale-to-zero), ~$30–50/month under moderate use. Azure Container Apps charges per vCPU-second and GB-second of active usage.

To adjust scaling:

```bash
az containerapp update \
  --name agent-factory-backend \
  --resource-group agent-factory-rg \
  --min-replicas 1 \
  --max-replicas 5
```

## Troubleshooting

```bash
# View backend logs
az containerapp logs show --name agent-factory-backend --resource-group agent-factory-rg --follow

# View frontend logs
az containerapp logs show --name agent-factory-frontend --resource-group agent-factory-rg --follow

# Check container app status
az containerapp show --name agent-factory-backend --resource-group agent-factory-rg --query "properties.runningStatus"
```
