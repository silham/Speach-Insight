import React from 'react';
import { getGuidelineDetails, getScoreStatus, CATEGORY_COLORS } from '../../utils/evaluationUtils';
import { CheckCircle2, Star, XCircle, AlertCircle, Info, Circle, ChevronDown } from 'lucide-react';

const renderStatusIcon = (type) => {
  switch (type) {
    case 'excellent': return <CheckCircle2 size={13} />;
    case 'good': return <Info size={13} />;
    case 'warning': return <AlertCircle size={13} />;
    case 'danger': return <XCircle size={13} />;
    default: return <Circle size={13} />;
  }
};

/**
 * EvaluationAccordion — Standardized 44px collapsed height.
 * Clean layout: Name | Score | Progress | Status | Chevron.
 */
export const EvaluationAccordion = React.memo(({ cat, results, isExpanded, onToggle }) => {
  const details = getGuidelineDetails(cat, results);
  const status = getScoreStatus(cat.score, cat.max_score);
  const progressPercent = (cat.score / cat.max_score) * 100;
  const catColor = CATEGORY_COLORS[cat.name] || '#8b5cf6';

  return (
    <div className={`eval-row ${isExpanded ? 'open' : ''}`}>
      <div className="eval-summary" onClick={onToggle}>
        <div className="eval-info">
          <span className="eval-name">{details.title}</span>
          <span className="eval-score" style={{ color: catColor }}>
            {cat.score}/{cat.max_score}
          </span>
        </div>
        
        <div className="eval-progress-container">
          <div
            className="eval-progress-bar"
            style={{ width: `${progressPercent}%`, backgroundColor: catColor }}
          />
        </div>

        <div style={{ color: status.color, fontSize: '0.72rem', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '0.2rem', minWidth: '110px', justifyContent: 'flex-end' }}>
          {renderStatusIcon(status.type)}
          <span>{status.label}</span>
        </div>

        <div className="eval-dropdown-toggle" style={{ transform: isExpanded ? 'rotate(180deg)' : 'none' }}>
          <ChevronDown size={14} />
        </div>
      </div>

      <div className="eval-details-pane" style={{ display: isExpanded ? 'block' : 'none' }}>
        {/* Explanation */}
        {details.explanation && (
          <div className="eval-detail-section">
            <span className="eval-detail-title">Score Explanation</span>
            <p className="eval-detail-text">{details.explanation}</p>
          </div>
        )}

        {/* Evidence */}
        {details.evidence && details.evidence.length > 0 && (
          <div className="eval-detail-section">
            <span className="eval-detail-title">Evidence Found</span>
            <ul className="eval-detail-list">
              {details.evidence.map((ev, i) => (
                <li key={i} className="eval-detail-item" style={{ color: 'var(--text-secondary)', display: 'flex', gap: '0.3rem' }}>
                  <CheckCircle2 size={13} style={{ marginTop: '0.1rem', flexShrink: 0 }} />
                  <span>{ev}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Positive Examples */}
        {details.positiveExamples && details.positiveExamples.length > 0 && (
          <div className="eval-detail-section">
            <span className="eval-detail-title">Positive Examples</span>
            <ul className="eval-detail-list">
              {details.positiveExamples.map((ex, i) => (
                <li key={i} className="eval-detail-item" style={{ color: '#10b981', display: 'flex', gap: '0.3rem' }}>
                  <Star size={13} style={{ marginTop: '0.1rem', flexShrink: 0 }} />
                  <span>{ex}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Missed Opportunities */}
        {details.missedOpportunities && details.missedOpportunities.length > 0 && (
          <div className="eval-detail-section">
            <span className="eval-detail-title">Missed Opportunities</span>
            <ul className="eval-detail-list">
              {details.missedOpportunities.map((mo, i) => (
                <li key={i} className="eval-detail-item" style={{ color: '#ef4444', display: 'flex', gap: '0.3rem' }}>
                  <XCircle size={13} style={{ marginTop: '0.1rem', flexShrink: 0 }} />
                  <span>{mo}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Recommendation */}
        {details.recommendation && (
          <div className="eval-detail-section recommendation-section">
            <span className="eval-detail-title" style={{ color: 'var(--accent)' }}>Recommendation</span>
            <p className="eval-detail-text recommendation-text" style={{ fontWeight: 500 }}>{details.recommendation}</p>
          </div>
        )}
      </div>
    </div>
  );
});

EvaluationAccordion.displayName = 'EvaluationAccordion';
