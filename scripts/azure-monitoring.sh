#!/usr/bin/env bash
# ── Azure Monitoring & Alerting — LLM cost protection ──────────
# Sets up budget alerts and monitoring for Azure OpenAI usage.
#
# Prerequisites:
#   - Azure infrastructure already created (run azure-setup.sh first)
#   - Azure CLI logged in: az login
#
# Usage:
#   chmod +x scripts/azure-monitoring.sh
#   ./scripts/azure-monitoring.sh <your-email@example.com>

set -euo pipefail

ALERT_EMAIL="${1:?Usage: $0 <alert-email-address>}"

# ── Configuration ───────────────────────────────────────────────
RESOURCE_GROUP="agent-factory-rg"
BACKEND_APP="agent-factory-backend"
OPENAI_RESOURCE_NAME="${AZURE_OPENAI_RESOURCE_NAME:-}"  # set if known
MONTHLY_BUDGET_USD="${MONTHLY_BUDGET_USD:-100}"
LOCATION="eastus"

echo "==> Setting up monitoring for Agent Factory"
echo "    Alert email: $ALERT_EMAIL"
echo "    Monthly budget: \$${MONTHLY_BUDGET_USD}"
echo ""

# ── 1. Create Action Group (email notifications) ───────────────
echo "==> Creating alert action group..."
az monitor action-group create \
  --resource-group "$RESOURCE_GROUP" \
  --name "agent-factory-alerts" \
  --short-name "af-alerts" \
  --action email admin-alert "$ALERT_EMAIL"

echo "    ✓ Action group created"

# ── 2. Budget Alert (monthly Azure spend) ──────────────────────
echo "==> Creating monthly budget alert (\$${MONTHLY_BUDGET_USD})..."
SUBSCRIPTION_ID=$(az account show --query id -o tsv)

az consumption budget create \
  --budget-name "agent-factory-monthly" \
  --amount "$MONTHLY_BUDGET_USD" \
  --category Cost \
  --time-grain Monthly \
  --start-date "$(date +%Y-%m-01)" \
  --end-date "$(date -d '+1 year' +%Y-%m-01 2>/dev/null || date -v+1y +%Y-%m-01)" \
  --resource-group "$RESOURCE_GROUP" \
  --notifications \
    '{
      "at50pct": {
        "enabled": true,
        "operator": "GreaterThan",
        "threshold": 50,
        "contactEmails": ["'"$ALERT_EMAIL"'"]
      },
      "at80pct": {
        "enabled": true,
        "operator": "GreaterThan",
        "threshold": 80,
        "contactEmails": ["'"$ALERT_EMAIL"'"]
      },
      "at100pct": {
        "enabled": true,
        "operator": "GreaterThan",
        "threshold": 100,
        "contactEmails": ["'"$ALERT_EMAIL"'"]
      }
    }'

echo "    ✓ Budget alerts: 50%, 80%, 100% of \$${MONTHLY_BUDGET_USD}/month"

# ── 3. Container App scaling alert (high replica count) ────────
echo "==> Creating high-load alert (replica count > 2)..."
BACKEND_ID=$(az containerapp show \
  --name "$BACKEND_APP" \
  --resource-group "$RESOURCE_GROUP" \
  --query id -o tsv)

az monitor metrics alert create \
  --resource-group "$RESOURCE_GROUP" \
  --name "af-backend-high-replicas" \
  --description "Agent Factory backend scaling beyond 2 replicas (high load)" \
  --scopes "$BACKEND_ID" \
  --condition "avg Replicas > 2" \
  --window-size 5m \
  --evaluation-frequency 1m \
  --severity 2 \
  --action "agent-factory-alerts"

echo "    ✓ High-load alert created"

# ── 4. Container App request rate alert ────────────────────────
echo "==> Creating high request rate alert (>1000 requests/5min)..."
az monitor metrics alert create \
  --resource-group "$RESOURCE_GROUP" \
  --name "af-backend-high-requests" \
  --description "Agent Factory receiving unusually high request volume" \
  --scopes "$BACKEND_ID" \
  --condition "total Requests > 1000" \
  --window-size 5m \
  --evaluation-frequency 1m \
  --severity 2 \
  --action "agent-factory-alerts"

echo "    ✓ High request rate alert created"

# ── 5. Azure OpenAI usage alert (if resource name provided) ───
if [ -n "$OPENAI_RESOURCE_NAME" ]; then
  echo "==> Creating Azure OpenAI token usage alert..."
  OPENAI_ID=$(az cognitiveservices account show \
    --name "$OPENAI_RESOURCE_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --query id -o tsv 2>/dev/null || echo "")

  if [ -n "$OPENAI_ID" ]; then
    az monitor metrics alert create \
      --resource-group "$RESOURCE_GROUP" \
      --name "af-openai-high-tokens" \
      --description "Azure OpenAI token usage spike — possible abuse" \
      --scopes "$OPENAI_ID" \
      --condition "total TokenTransaction > 100000" \
      --window-size 1h \
      --evaluation-frequency 5m \
      --severity 1 \
      --action "agent-factory-alerts"

    echo "    ✓ Azure OpenAI token alert created (>100k tokens/hour)"
  else
    echo "    ⚠ Could not find OpenAI resource '$OPENAI_RESOURCE_NAME' in $RESOURCE_GROUP"
    echo "      Skipping OpenAI-specific alert. You can set it up manually later."
  fi
else
  echo "    ℹ Skipping Azure OpenAI token alert (set AZURE_OPENAI_RESOURCE_NAME to enable)"
fi

# ── 6. Enable Log Analytics for the Container App ─────────────
echo "==> Enabling diagnostic logging..."
WORKSPACE_ID=$(az monitor log-analytics workspace list \
  --resource-group "$RESOURCE_GROUP" \
  --query "[0].id" -o tsv 2>/dev/null || echo "")

if [ -z "$WORKSPACE_ID" ]; then
  echo "    Creating Log Analytics workspace..."
  az monitor log-analytics workspace create \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "agent-factory-logs" \
    --location "$LOCATION"

  WORKSPACE_ID=$(az monitor log-analytics workspace show \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "agent-factory-logs" \
    --query id -o tsv)
fi

echo "    ✓ Log Analytics workspace ready"

echo ""
echo "============================================"
echo "  Monitoring setup complete!"
echo "============================================"
echo ""
echo "  Alerts configured:"
echo "  • Budget: 50%, 80%, 100% of \$${MONTHLY_BUDGET_USD}/month"
echo "  • High load: replica count > 2"
echo "  • Traffic spike: >1000 requests / 5 min"
if [ -n "$OPENAI_RESOURCE_NAME" ]; then
  echo "  • OpenAI tokens: >100k tokens / hour"
fi
echo ""
echo "  All alerts → $ALERT_EMAIL"
echo ""
echo "  Application-level limits (configurable via env vars):"
echo "  • RATE_LIMIT_REQUESTS=30       (per IP per 60s)"
echo "  • SESSION_MAX_LLM_CALLS=50     (per session)"
echo "  • DAILY_MAX_LLM_CALLS=5000     (global daily cap)"
echo ""
echo "  Monitor dashboard:"
echo "  az monitor metrics list --resource \$BACKEND_ID --metric Requests"
echo "============================================"
