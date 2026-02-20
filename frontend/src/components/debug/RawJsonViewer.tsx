"use client";

import { useState } from "react";
import { Copy, Check } from "lucide-react";

interface Props {
  data: unknown;
}

export function RawJsonViewer({ data }: Props) {
  const [copied, setCopied] = useState(false);
  const json = JSON.stringify(data, null, 2);

  function handleCopy() {
    navigator.clipboard.writeText(json);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <h3 className="text-xs font-semibold uppercase text-slate-500">
          Raw Response
        </h3>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 text-xs text-slate-400 hover:text-slate-600"
        >
          {copied ? <Check size={12} /> : <Copy size={12} />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <pre className="max-h-80 overflow-auto rounded-lg bg-slate-900 p-3 text-xs text-slate-300">
        {json}
      </pre>
    </div>
  );
}
