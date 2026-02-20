"use client";

interface QuickRepliesProps {
  replies: string[];
  onSelect: (reply: string) => void;
}

export function QuickReplies({ replies, onSelect }: QuickRepliesProps) {
  if (!replies.length) return null;
  return (
    <div className="flex flex-wrap gap-2 px-4 pb-2">
      {replies.map((r) => (
        <button
          key={r}
          onClick={() => onSelect(r)}
          className="rounded-full border border-blue-200 bg-blue-50 px-3 py-1 text-xs font-medium text-blue-700 transition-colors hover:bg-blue-100"
        >
          {r}
        </button>
      ))}
    </div>
  );
}
