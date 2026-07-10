import React from 'react';
import { Check } from 'lucide-react';

export const PredictionFactors = React.memo(({ factors = [] }) => {
  if (!factors || factors.length === 0) return null;

  return (
    <div className="speaker-behavior-box" style={{ borderTop: '1px solid var(--border-color)', paddingTop: '0.65rem' }}>
      <span className="speaker-behavior-title" style={{ fontSize: '0.72rem', color: 'var(--text-light)', fontWeight: '700', textTransform: 'uppercase', letterSpacing: '0.03em' }}>
        Behaviour Summary
      </span>
      <ul style={{ listStyle: 'none', paddingLeft: 0, margin: '0.2rem 0 0 0', display: 'flex', flexDirection: 'column', gap: '0.2rem' }}>
        {factors.map((factor, idx) => (
          <li
            key={idx}
            className="prediction-factor-item"
            style={{
              fontSize: '0.78rem',
              color: 'var(--text-secondary)',
              display: 'flex',
              alignItems: 'flex-start',
              gap: '0.3rem',
              lineHeight: 1.25
            }}
          >
            <Check size={14} style={{ color: 'var(--accent)', flexShrink: 0, marginTop: '2px' }} />
            <span>{factor}</span>
          </li>
        ))}
      </ul>
    </div>
  );
});

PredictionFactors.displayName = 'PredictionFactors';
