#!/usr/bin/env bash
# ── Azure Container Apps — One-time infrastructure setup ───────
# Run this once to create the Azure resources needed for deployment.
#
# Prerequisites:
#   - Azure CLI installed (https://learn.microsoft.com/en-us/cli/azure/install-azure-cli)
#   - Logged in: az login
#
# Usage:
#   chmod +x scripts/azure-setup.sh
#   ./scripts/azure-setup.sh

set -euo pipefail

# ── Configuration (edit these to match your environment) ───────
RESOURCE_GROUP="MetaAgentFactory"
LOCATION="westeurope"                # Co-locate with your Azure OpenAI resource
ACR_NAME="metaagentfactoryacr"
CONTAINER_ENV="agent-factory-env"
BACKEND_APP="agent-factory-backend"
FRONTEND_APP="agent-factory-frontend"

echo "==> Creating resource group: $RESOURCE_GROUP"
az group create --name "$RESOURCE_GROUP" --location "$LOCATION"

echo "==> Creating Azure Container Registry: $ACR_NAME"
az acr create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$ACR_NAME" \
  --sku Basic \
  --admin-enabled true

echo "==> Creating Container Apps environment: $CONTAINER_ENV"
az containerapp env create \
  --name "$CONTAINER_ENV" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION"

echo "==> Creating backend container app: $BACKEND_APP"
az containerapp create \
  --name "$BACKEND_APP" \
  --resource-group "$RESOURCE_GROUP" \
  --environment "$CONTAINER_ENV" \
  --image "mcr.microsoft.com/azuredocs/containerapps-helloworld:latest" \
  --target-port 8080 \
  --ingress external \
  --min-replicas 0 \
  --max-replicas 3 \
  --cpu 1.0 \
  --memory 2.0Gi

echo "==> Creating frontend container app: $FRONTEND_APP"
az containerapp create \
  --name "$FRONTEND_APP" \
  --resource-group "$RESOURCE_GROUP" \
  --environment "$CONTAINER_ENV" \
  --image "mcr.microsoft.com/azuredocs/containerapps-helloworld:latest" \
  --target-port 3000 \
  --ingress external \
  --min-replicas 0 \
  --max-replicas 3 \
  --cpu 0.5 \
  --memory 1.0Gi

# ── Store secrets for the backend ──
echo ""
echo "==> Next: Add your Azure OpenAI secrets to the backend app."
echo "    Run these commands with your actual values:"
echo ""
echo "    az containerapp secret set \\"
echo "      --name $BACKEND_APP \\"
echo "      --resource-group $RESOURCE_GROUP \\"
echo "      --secrets \\"
echo "        azure-openai-endpoint=<YOUR_ENDPOINT> \\"
echo "        azure-openai-api-key=<YOUR_API_KEY> \\"
echo "        azure-openai-deployment=<YOUR_DEPLOYMENT>"
echo ""

# ── Create GitHub Actions service principal ──
echo "==> Creating service principal for GitHub Actions CI/CD..."
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
SP_OUTPUT=$(az ad sp create-for-rbac \
  --name "agent-factory-github" \
  --role contributor \
  --scopes "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP" \
  --sdk-auth)

echo ""
echo "==> Add this JSON as a GitHub repository secret named AZURE_CREDENTIALS:"
echo "$SP_OUTPUT"
echo ""

# Grant ACR push/pull to the service principal
SP_APP_ID=$(echo "$SP_OUTPUT" | python3 -c "import sys,json; print(json.load(sys.stdin)['clientId'])")
ACR_ID=$(az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" --query id -o tsv)
az role assignment create --assignee "$SP_APP_ID" --role AcrPush --scope "$ACR_ID"

BACKEND_URL=$(az containerapp show --name "$BACKEND_APP" --resource-group "$RESOURCE_GROUP" --query "properties.configuration.ingress.fqdn" -o tsv)
FRONTEND_URL=$(az containerapp show --name "$FRONTEND_APP" --resource-group "$RESOURCE_GROUP" --query "properties.configuration.ingress.fqdn" -o tsv)

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo "  Backend URL:  https://$BACKEND_URL"
echo "  Frontend URL: https://$FRONTEND_URL"
echo ""
echo "  Next steps:"
echo "  1. Add AZURE_CREDENTIALS secret to GitHub repo"
echo "  2. Add Azure OpenAI secrets (see command above)"
echo "  3. Push to main branch to trigger deployment"
echo "============================================"
