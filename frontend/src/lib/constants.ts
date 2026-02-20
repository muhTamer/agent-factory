export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE || "http://127.0.0.1:808";

export const CONCIERGE_API =
  process.env.NEXT_PUBLIC_CONCIERGE_API || "http://127.0.0.1:8001";

export const AGENT_DISPLAY: Record<
  string,
  { icon: string; label: string }
> = {
  faq_rag: { icon: "MessageSquare", label: "FAQ Agent" },
  faq: { icon: "MessageSquare", label: "FAQ Agent" },
  knowledge_rag: { icon: "BookOpen", label: "Knowledge Agent" },
  workflow_runner: { icon: "GitBranch", label: "Workflow Agent" },
  tool_operator: { icon: "Wrench", label: "Tool Agent" },
  guardrails: { icon: "Shield", label: "Safety Guard" },
  complaint: { icon: "AlertTriangle", label: "Complaint Handler" },
  refund: { icon: "DollarSign", label: "Refund Specialist" },
  router: { icon: "Route", label: "Router" },
};

export function getAgentDisplay(
  agentId: string,
  agentType?: string
): { icon: string; label: string } {
  // Try exact id match
  for (const [key, val] of Object.entries(AGENT_DISPLAY)) {
    if (agentId.toLowerCase().includes(key)) return val;
  }
  // Try type match
  if (agentType) {
    for (const [key, val] of Object.entries(AGENT_DISPLAY)) {
      if (agentType.toLowerCase().includes(key)) return val;
    }
  }
  return { icon: "Bot", label: agentId.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()) };
}
