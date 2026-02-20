"use client";

import {
  MessageSquare,
  GitBranch,
  Wrench,
  Shield,
  AlertTriangle,
  DollarSign,
  BookOpen,
  Route,
  Bot,
  User,
} from "lucide-react";

const ICON_MAP: Record<string, React.ElementType> = {
  MessageSquare,
  GitBranch,
  Wrench,
  Shield,
  AlertTriangle,
  DollarSign,
  BookOpen,
  Route,
  Bot,
  User,
};

interface AgentAvatarProps {
  iconName: string;
  label: string;
  size?: number;
}

export function AgentAvatar({ iconName, label, size = 18 }: AgentAvatarProps) {
  const Icon = ICON_MAP[iconName] || Bot;
  return (
    <div className="flex items-center gap-1.5 mb-1">
      <div className="flex h-6 w-6 items-center justify-center rounded-full bg-slate-200">
        <Icon size={size - 4} className="text-slate-600" />
      </div>
      <span className="text-xs font-semibold text-slate-500">{label}</span>
    </div>
  );
}
