import React from 'react';

/**
 * EvaluationSummary — Executive report header.
 * No borders on the score header or tone row — they are subordinate to the accordion.
 * Visual hierarchy: Score → Tone → Strengths/Recommendations.
 */
export const EvaluationSummary = React.memo(({ totalScore, vibeStats, strengths, improvements }) => {
  const dashArray = `${Math.round(totalScore)}, 100`;

  return (
    <div className="report-tab-layout" style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
      {/* 1. Score header — no border, no background */}
      <div className="report-radial-header">
        <div className="report-intro-text" style={{ display: 'flex', flexDirection: 'column', gap: '0.1rem' }}>
          <h5 style={{ margin: 0 }}>Meeting Performance</h5>
          <span className="report-subtext">
            Compliance rating based on conversational scoring rules.
          </span>
        </div>

        <div className="radial-score-circle">
          <svg className="radial-svg" viewBox="0 0 36 36">
            <path
              d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
              style={{ stroke: 'rgba(255, 255, 255, 0.04)', strokeWidth: 3, fill: 'none' }}
            />
            <path
              strokeDasharray={dashArray}
              stroke="#8b5cf6"
              d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
              style={{ strokeWidth: 3, fill: 'none', strokeLinecap: 'round', transition: 'stroke-dasharray 0.5s ease' }}
            />
          </svg>
          <div className="radial-inner-label">
            <span className="score-num">{Math.round(totalScore)}</span>
            <span className="score-total">/100</span>
          </div>
        </div>
      </div>

      {/* 2. Conversation Tone — no border, just a divider */}
      {vibeStats && vibeStats.length > 0 && (
        <div className="meeting-vibe-summary">
          <span className="vibe-label">Tone profile</span>
          <div className="vibe-pills-row">
            {vibeStats.slice(0, 3).map((vibe, i) => (
              <span
                key={i}
                className="vibe-pill"
                style={{
                  backgroundColor: 'rgba(16, 185, 129, 0.08)',
                  border: '1px solid rgba(16, 185, 129, 0.15)',
                  color: '#10b981',
                }}
              >
                {vibe.name}: {vibe.percentage}%
              </span>
            ))}
          </div>
        </div>
      )}

      {/* 3 & 4. Strengths & Recommendations */}
      <div className="report-summaries-grid">
        {strengths && strengths.length > 0 && (
          <div className="summary-list-card strength">
            <h6>Key Strengths</h6>
            <ul>
              {strengths.slice(0, 2).map((str, i) => (
                <li key={i}>{str}</li>
              ))}
            </ul>
          </div>
        )}

        {improvements && improvements.length > 0 && (
          <div className="summary-list-card improvement">
            <h6>Recommendations</h6>
            <ul>
              {improvements.slice(0, 2).map((imp, i) => (
                <li key={i}>{imp}</li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
});

EvaluationSummary.displayName = 'EvaluationSummary';
