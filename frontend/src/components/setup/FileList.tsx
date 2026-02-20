"use client";

import { X, FileText } from "lucide-react";
import { Badge } from "@/components/ui/badge";

interface FileListProps {
  files: File[];
  onRemove: (name: string) => void;
}

function formatSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function FileList({ files, onRemove }: FileListProps) {
  if (!files.length) return null;

  return (
    <ul className="space-y-2">
      {files.map((f) => {
        const ext = f.name.split(".").pop()?.toUpperCase() || "FILE";
        return (
          <li
            key={f.name}
            className="flex items-center gap-3 rounded-lg border border-slate-200 bg-white px-3 py-2"
          >
            <FileText size={16} className="shrink-0 text-slate-400" />
            <span className="flex-1 truncate text-sm text-slate-700">
              {f.name}
            </span>
            <Badge variant="outline" className="text-[10px]">
              {ext}
            </Badge>
            <span className="text-xs text-slate-400">{formatSize(f.size)}</span>
            <button
              onClick={() => onRemove(f.name)}
              className="rounded p-1 text-slate-400 transition-colors hover:bg-slate-100 hover:text-slate-600"
            >
              <X size={14} />
            </button>
          </li>
        );
      })}
    </ul>
  );
}
