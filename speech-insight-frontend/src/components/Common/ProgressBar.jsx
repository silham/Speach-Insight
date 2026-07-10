import React from 'react';
import { AlertTriangle } from 'lucide-react';

export const ProgressBar = React.memo(({ value, showWarning = false, colorOverride = null, height = '6px' }) => {
  const percent = Math.min(100, Math.max(0, Math.round(value)));

  // Thresholds color coding: Green (>=80), Yellow (60-79), Red (<60)
  let color = '#10b981';
  if (percent < 60) {
    color = '#ef4444';
  } else if (percent < 80) {
    color = '#f59e0b';
  }

  const finalColor = colorOverride || color;

  return (
    <div className="confidence-visualizer">
      <div className="confidence-bar-track" style={{ height }}>
        <div
          className="confidence-bar-fill"
          style={{
            width: `${percent}%`,
            backgroundColor: finalColor,
          }}
        />
      </div>
      <div className="confidence-bar-meta">
        <span className="confidence-percent" style={{ color: finalColor }}>{percent}%</span>
        {showWarning && percent < 60 && (
          <span className="confidence-low-alert">
            <AlertTriangle size={12} /> Low confidence
          </span>
        )}
      </div>
    </div>
  );
});

ProgressBar.displayName = 'ProgressBar';
