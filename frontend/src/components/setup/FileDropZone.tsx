"use client";

import { useCallback, useRef, useState } from "react";
import { Upload } from "lucide-react";
import { cn } from "@/lib/utils";

const ACCEPT = ".csv,.yaml,.yml,.md,.txt";

interface FileDropZoneProps {
  onFiles: (files: File[]) => void;
  disabled?: boolean;
}

export function FileDropZone({ onFiles, disabled }: FileDropZoneProps) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      if (disabled) return;
      const files = Array.from(e.dataTransfer.files);
      if (files.length) onFiles(files);
    },
    [onFiles, disabled]
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(e.target.files || []);
      if (files.length) onFiles(files);
      e.target.value = "";
    },
    [onFiles]
  );

  return (
    <div
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      onClick={() => !disabled && inputRef.current?.click()}
      className={cn(
        "flex cursor-pointer flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed px-6 py-10 transition-colors",
        dragOver
          ? "border-blue-400 bg-blue-50"
          : "border-slate-300 bg-slate-50 hover:border-slate-400",
        disabled && "pointer-events-none opacity-50"
      )}
    >
      <Upload size={28} className="text-slate-400" />
      <p className="text-sm font-medium text-slate-600">
        Drag & drop files here, or click to browse
      </p>
      <p className="text-xs text-slate-400">CSV, YAML, Markdown, or TXT</p>
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT}
        multiple
        onChange={handleChange}
        className="hidden"
      />
    </div>
  );
}
