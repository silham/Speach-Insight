import React from 'react';
import { BadgeGroup } from '../Common/BadgeGroup';

export const TranscriptCard = React.memo(({
  row,
  isActive,
  isPlaying,
  speakerTheme,
  onCardClick,
  onPlayClick,
  activeSegmentId
}) => {
  // If row is grouped, row.textParts is a list of sub-utterance structures
  const parts = row.textParts || [{ text: row.text, segment_id: row.segment_id, end_time: row.end_time }];

  return (
    <div
      id={`bubble-${row.segment_id}`}
      className={`bubble-wrapper ${isActive ? 'active-bubble' : ''}`}
      onClick={(e) => {
        // Only select if not clicking buttons
        if (e.target.tagName !== 'BUTTON' && !e.target.closest('button')) {
          onCardClick(row.segment_id);
        }
      }}
    >
      {/* Header */}
      <div className="bubble-header">
        <span
          className="bubble-speaker font-bold"
          style={{
            color: speakerTheme.text,
            backgroundColor: speakerTheme.bg,
            borderColor: speakerTheme.border
          }}
        >
          {row.speaker}
        </span>
        <span className="bubble-time">
          {parseFloat(row.start_time).toFixed(1)}s - {parseFloat(row.end_time).toFixed(1)}s
        </span>
      </div>

      {/* Body paragraphs */}
      <div className="bubble-body">
        {parts.map((p, index) => {
          const isSubActive = activeSegmentId === p.segment_id;
          return (
            <div
              key={index}
              className={`bubble-text-part ${isSubActive ? 'sub-active-text' : ''}`}
              style={{
                borderRadius: '4px',
                padding: parts.length > 1 ? '0.2rem 0.4rem' : '0',
                backgroundColor: isSubActive ? 'rgba(0, 240, 255, 0.08)' : 'transparent',
                transition: 'background-color 0.2s'
              }}
            >
              <p className="bubble-text">{p.text}</p>
            </div>
          );
        })}
      </div>

      {/* Meta bottom row */}
      <div className="bubble-meta-row">
        <BadgeGroup
          emotion={row.emotion}
          emotionConf={row.confidence}
          templateLabel={row.template_label}
          role={row.role}
          roleConf={row.role_confidence}
        />

        <button
          className={`bubble-play-btn ${isPlaying ? 'playing' : ''}`}
          onClick={(e) => {
            e.stopPropagation();
            onPlayClick(row.segment_id);
          }}
        >
          {isPlaying ? (
            <>
              <div className="soundwave-anim">
                <span className="soundwave-bar" />
                <span className="soundwave-bar" />
                <span className="soundwave-bar" />
              </div>
              <span>Playing</span>
            </>
          ) : (
            <>
              <svg className="play-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
                <polygon points="5 3 19 12 5 21 5 3"></polygon>
              </svg>
              <span>Play</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
});

TranscriptCard.displayName = 'TranscriptCard';
