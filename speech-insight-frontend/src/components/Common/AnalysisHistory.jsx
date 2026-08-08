import React from 'react';
import { Trash2 } from 'lucide-react';

const formatDate = (iso) => {
  if (!iso) return '';
  const d = new Date(iso.endsWith('Z') ? iso : `${iso}Z`);
  return Number.isNaN(d.getTime())
    ? ''
    : d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
};

/**
 * Past analyses visible to the signed-in user. The backend already restricts
 * this list — a normal user only ever receives their own rows, an admin
 * receives everyone's — so there is no client-side filtering to do here.
 */
export const AnalysisHistory = ({ analyses, loading, activeJobId, scope, onOpen, onDelete }) => {
  if (loading) {
    return <p className="history-empty">Loading history…</p>;
  }

  if (!analyses.length) {
    return <p className="history-empty">No analyses yet. Upload a media file to get started.</p>;
  }

  return (
    <div className="history-list">
      {analyses.map((a) => (
        <div
          key={a.job_id}
          className={`history-item ${activeJobId === a.job_id ? 'active' : ''}`}
          onClick={() => onOpen(a.job_id)}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => { if (e.key === 'Enter') onOpen(a.job_id); }}
          title={a.filename}
        >
          <div className="history-item-main">
            <span className="history-filename">{a.filename}</span>
            <span className="history-meta">
              {formatDate(a.created_at)}
              {a.total_speakers != null && ` · ${a.total_speakers} spk`}
              {a.total_score != null && ` · ${Math.round(a.total_score)}/100`}
            </span>
            {/* Only meaningful for admins, who see other people's rows too. */}
            {scope === 'all' && a.owner_name && (
              <span className="history-owner">{a.owner_name}</span>
            )}
          </div>
          <button
            className="history-delete"
            title="Delete this analysis"
            onClick={(e) => { e.stopPropagation(); onDelete(a.job_id, a.filename); }}
          >
            <Trash2 size={13} />
          </button>
        </div>
      ))}
    </div>
  );
};
