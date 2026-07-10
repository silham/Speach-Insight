import React, { useState } from 'react';

export const ExpandableText = React.memo(({ text, maxLength = 150 }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  const rawText = String(text || "");
  const shouldTrim = rawText.length > maxLength;
  const displayedText = (shouldTrim && !isExpanded) 
    ? rawText.substring(0, maxLength) + "..." 
    : rawText;

  return (
    <span className="expandable-text-wrapper" style={{ wordBreak: 'break-word' }}>
      "{displayedText}"
      {shouldTrim && (
        <button
          onClick={(e) => {
            e.stopPropagation();
            setIsExpanded(prev => !prev);
          }}
          style={{
            background: 'none',
            border: 'none',
            color: 'var(--accent)',
            cursor: 'pointer',
            fontSize: '0.78rem',
            textDecoration: 'underline',
            padding: '0 0 0 0.4rem',
            boxShadow: 'none',
            verticalAlign: 'baseline',
            display: 'inline'
          }}
        >
          {isExpanded ? "Show Less" : "View Full"}
        </button>
      )}
    </span>
  );
});

ExpandableText.displayName = 'ExpandableText';
