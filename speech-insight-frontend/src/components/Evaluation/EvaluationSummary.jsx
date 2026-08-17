import React from 'react';
import { CheckCircle2, AlertTriangle, Sparkles, AlertCircle, ShieldAlert } from 'lucide-react';

const renderFormattedText = (text) => {
  if (!text) return '';
  const parts = text.split('**');
  return parts.map((part, index) => {
    if (index % 2 === 1) {
      return <strong key={index} style={{ fontWeight: 800, color: 'var(--text-main)' }}>{part}</strong>;
    }
    return part;
  });
};

/**
 * EvaluationSummary — Redesigned Executive report header.
 * Visual hierarchy: Scorecard Card Widget (Score, Badge, Title, Description) -> Tone Profile -> Strengths & Recommendations Grid.
 */
export const EvaluationSummary = React.memo(({ totalScore, vibeStats, strengths, improvements }) => {
  const roundedScore = Math.round(totalScore);
  const dashArray = `${roundedScore}, 100`;

  // Determine the status classification
  let statusText = "Needs Attention";
  let statusColor = "var(--danger)";
  let statusBg = "rgba(255, 0, 85, 0.08)";
  let StatusIcon = ShieldAlert;

  if (roundedScore >= 80) {
    statusText = "Excellent Compliance";
    statusColor = "var(--success)";
    statusBg = "rgba(0, 255, 157, 0.08)";
    StatusIcon = Sparkles;
  } else if (roundedScore >= 60) {
    statusText = "Good Compliance";
    statusColor = "var(--accent)";
    statusBg = "rgba(0, 240, 255, 0.08)";
    StatusIcon = CheckCircle2;
  } else if (roundedScore >= 40) {
    statusText = "Fair Compliance";
    statusColor = "var(--orange)";
    statusBg = "rgba(255, 102, 0, 0.08)";
    StatusIcon = AlertCircle;
  }

  return (
    <div className="report-tab-layout">
      {/* 1. Large Overall Score Widget */}
      <div className="report-score-card">
        <div className="radial-score-container">
          <div className="radial-score-circle">
            <svg className="radial-svg" viewBox="0 0 36 36">
              <path
                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                className="radial-track"
              />
              <path
                strokeDasharray={dashArray}
                stroke="var(--accent)"
                d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                className="radial-progress"
              />
            </svg>
            <div className="radial-inner-label">
              <span className="score-num">{roundedScore}</span>
              <span className="score-total">/100</span>
            </div>
          </div>
        </div>

        <div className="report-score-info">
          <div className="status-badge" style={{ backgroundColor: statusBg, color: statusColor }}>
            <StatusIcon size={13} style={{ flexShrink: 0 }} />
            <span>{statusText}</span>
          </div>
          <h3 className="report-score-title">Meeting Performance</h3>
          <p className="report-score-description">
            Compliance rating based on conversational scoring rules, guideline similarity, and audio dynamics.
          </p>
        </div>
      </div>

      {/* 2. Conversation Tone Profile */}
      {vibeStats && vibeStats.length > 0 && (
        <div className="meeting-vibe-summary">
          <span className="vibe-label">Conversation Tone Profile</span>
          <div className="vibe-pills-row">
            {vibeStats.slice(0, 3).map((vibe, i) => (
              <span
                key={i}
                className="vibe-pill"
              >
                {vibe.name}: {vibe.percentage}%
              </span>
            ))}
          </div>
        </div>
      )}

      {/* 3 & 4. Key Strengths & Recommendations Grid */}
      <div className="report-summaries-grid">
        <div className="summary-list-card strength">
          <div className="card-header">
            <Sparkles size={16} />
            <h6>Key Strengths</h6>
          </div>
          <ul>
            {strengths && strengths.length > 0 ? (
              strengths.slice(0, 3).map((str, i) => (
                <li key={i}>{renderFormattedText(str)}</li>
              ))
            ) : (
              <li>Perfect structural layout and positive speaking tones.</li>
            )}
          </ul>
        </div>

        <div className="summary-list-card improvement">
          <div className="card-header">
            <AlertTriangle size={16} />
            <h6>Recommendations</h6>
          </div>
          <ul>
            {improvements && improvements.length > 0 ? (
              improvements.slice(0, 3).map((imp, i) => (
                <li key={i}>{renderFormattedText(imp)}</li>
              ))
            ) : (
              <li>Continue maintaining balanced dialogue ratios and guideline compliance.</li>
            )}
          </ul>
        </div>
      </div>
    </div>
  );
});

EvaluationSummary.displayName = 'EvaluationSummary';
