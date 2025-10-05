import { Link } from "react-router-dom";

type Props = {
  id: string;
  title?: string;
  label?: string;
  summary?: string;
  imageURL?: string;
  timestamp?: number;
};

export default function NoteCard({ id, title, label, summary, imageURL, timestamp }: Props) {
  const when = timestamp ? new Date(timestamp * 1000).toLocaleString() : "N/A";
  return (
    <Link
      to={`/note/${id}`}
      className="group block rounded-xl border border-[var(--border)] bg-[rgba(16,22,40,.55)] hover:bg-[rgba(16,22,40,.7)] transition shadow-sm hover:shadow-lg overflow-hidden"
    >
      <div className="aspect-[16/9] bg-black/20 overflow-hidden">
        {imageURL ? (
          <img
            src={imageURL}
            alt={title || id}
            className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-[1.02]"
            loading="lazy"
          />
        ) : (
          <div className="h-full w-full grid place-items-center text-[var(--muted)] text-sm">No preview</div>
        )}
      </div>
      <div className="p-3.5 space-y-2">
        <div className="flex items-center gap-2">
          <div className="font-semibold truncate">{title || id}</div>
          {label ? (
            <span className="inline-flex items-center rounded-full px-2.5 py-0.5 text-[11px] bg-[var(--accent)]/18 text-[#cfe0ff] border border-[var(--accent)]/30">
              {label}
            </span>
          ) : (
            <span className="text-[var(--muted)] text-[11px]">unlabeled</span>
          )}
        </div>
        {summary && <div className="text-sm text-white/80 line-clamp-2">{summary}</div>}
        <div className="text-[11px] text-[var(--muted)]">{when}</div>
      </div>
    </Link>
  );
}