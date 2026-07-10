import React from 'react';

export const EvidenceBlock = React.memo(({ evidence = [], title = "Evidence" }) => {
  if (!evidence || evidence.length === 0) return null;

  return (
    <div className="speaker-evidence-box">
      <span className="speaker-evidence-title">{title}</span>
      <ul className="speaker-evidence-list">
        {evidence.map((ev, idx) => (
          <li key={idx} className="speaker-evidence-item" style={{ wordBreak: 'break-word', fontSize: '0.85rem' }}>
            "{ev}"
          </li>
        ))}
      </ul>
    </div>
  );
});

EvidenceBlock.displayName = 'EvidenceBlock';
