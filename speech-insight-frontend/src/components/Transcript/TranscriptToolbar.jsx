import React from 'react';

export const TranscriptToolbar = React.memo(({
  collapseRepeated,
  onToggleCollapse,
  selectedSegmentId,
  onClearSelection,
  onJumpSegment,
  totalResultsCount
}) => {
  return (
    <div
      className="transcript-toolbar"
      style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        background: 'rgba(255, 255, 255, 0.02)',
        border: '1px solid var(--border-color)',
        padding: '0.6rem 1rem',
        borderRadius: 'var(--radius-md)',
        marginTop: '1rem',
        flexWrap: 'wrap',
        gap: '0.75rem'
      }}
    >
      {/* 1. Turn Merge Toggle Switch */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <input
          type="checkbox"
          id="collapse-turns-toggle"
          checked={collapseRepeated}
          onChange={(e) => onToggleCollapse(e.target.checked)}
          style={{ width: '16px', height: '16px', cursor: 'pointer', accentColor: 'var(--accent)' }}
        />
        <label htmlFor="collapse-turns-toggle" style={{ fontSize: '0.82rem', fontWeight: '600', color: 'var(--text-secondary)', cursor: 'pointer' }}>
          Collapse consecutive turns
        </label>
      </div>

      {/* 2. Jump Buttons (Prev/Next Segment) */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        {selectedSegmentId !== null && (
          <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginRight: '0.5rem' }}>
            Selected Turn: #{selectedSegmentId}
          </span>
        )}

        <button
          className="btn-sidebar-reset"
          onClick={() => onJumpSegment('prev')}
          style={{
            margin: 0,
            padding: '0.35rem 0.65rem',
            borderRadius: 'var(--radius-sm)',
            fontSize: '0.78rem',
            fontWeight: 'bold',
            display: 'flex',
            alignItems: 'center',
            gap: '0.25rem'
          }}
          title="Jump to previous segment turn"
        >
          ▲ Prev Turn
        </button>

        <button
          className="btn-sidebar-reset"
          onClick={() => onJumpSegment('next')}
          style={{
            margin: 0,
            padding: '0.35rem 0.65rem',
            borderRadius: 'var(--radius-sm)',
            fontSize: '0.78rem',
            fontWeight: 'bold',
            display: 'flex',
            alignItems: 'center',
            gap: '0.25rem'
          }}
          title="Jump to next segment turn"
        >
          ▼ Next Turn
        </button>

        {selectedSegmentId !== null && (
          <button
            className="btn-sidebar-reset"
            onClick={onClearSelection}
            style={{
              margin: 0,
              padding: '0.35rem 0.65rem',
              borderRadius: 'var(--radius-sm)',
              fontSize: '0.78rem',
              fontWeight: 'bold',
              backgroundColor: 'rgba(255, 0, 85, 0.08)',
              borderColor: 'rgba(255, 0, 85, 0.2)',
              color: 'var(--danger)'
            }}
          >
            Clear Selected
          </button>
        )}
      </div>
    </div>
  );
});

TranscriptToolbar.displayName = 'TranscriptToolbar';
