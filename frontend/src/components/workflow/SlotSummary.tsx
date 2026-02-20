"use client";

interface Props {
  slots: Record<string, unknown>;
  missingSlots?: string[];
}

export function SlotSummary({ slots, missingSlots }: Props) {
  const entries = Object.entries(slots).filter(
    ([, v]) => v !== null && v !== undefined
  );
  if (!entries.length && !missingSlots?.length) return null;

  return (
    <div className="mt-2 rounded-lg border border-slate-200 bg-white p-2 text-xs">
      {entries.map(([k, v]) => (
        <div key={k} className="flex justify-between py-0.5">
          <span className="text-slate-500">{k.replace(/_/g, " ")}</span>
          <span className="font-medium text-slate-700">{String(v)}</span>
        </div>
      ))}
      {missingSlots?.map((s) => (
        <div key={s} className="flex justify-between py-0.5">
          <span className="text-slate-500">{s.replace(/_/g, " ")}</span>
          <span className="text-amber-600 font-medium">needed</span>
        </div>
      ))}
    </div>
  );
}
