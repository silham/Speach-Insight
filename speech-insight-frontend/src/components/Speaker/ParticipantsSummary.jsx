import React from 'react';
import { Users, User, Box, Award } from 'lucide-react';

export const ParticipantsSummary = React.memo(({ stats = {}, totalSpeakers, leadSpeaker }) => {
  return (
    <div
      className="composition-card"
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '0.65rem',
        padding: '1rem',
        height: 'fit-content',
        alignSelf: 'start'
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span className="composition-title" style={{ fontSize: '0.78rem' }}>Participants</span>
        <span
          className="badge-template font-bold"
          style={{
            borderColor: 'var(--accent)',
            color: 'var(--accent)',
            backgroundColor: 'rgba(0, 240, 255, 0.08)',
            fontSize: '0.72rem',
            padding: '0.1rem 0.45rem',
            borderRadius: '4px'
          }}
        >
          {totalSpeakers} Total
        </span>
      </div>

      <div
        className="composition-stats-grid"
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(2, 1fr)',
          gap: '0.45rem',
          marginTop: '0.2rem'
        }}
      >
        {/* Leader (Blue) */}
        <div
          className="composition-chip"
          style={{
            background: 'rgba(59, 130, 246, 0.08)',
            border: '1px solid rgba(59, 130, 246, 0.2)',
            borderRadius: '4px',
            padding: '0.35rem 0.5rem',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            fontSize: '0.78rem'
          }}
        >
          <span style={{ color: '#3b82f6', fontWeight: '600', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
            <Award size={14} /> Leader
          </span>
          <span style={{ color: '#3b82f6', fontWeight: '800' }}>{stats.Leader || 0}</span>
        </div>

        {/* HR (Teal) */}
        <div
          className="composition-chip"
          style={{
            background: 'rgba(20, 184, 166, 0.08)',
            border: '1px solid rgba(20, 184, 166, 0.2)',
            borderRadius: '4px',
            padding: '0.35rem 0.5rem',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            fontSize: '0.78rem'
          }}
        >
          <span style={{ color: '#14b8a6', fontWeight: '600', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
            <Users size={14} /> HR
          </span>
          <span style={{ color: '#14b8a6', fontWeight: '800' }}>{stats.HR || 0}</span>
        </div>

        {/* Junior (Purple) */}
        <div
          className="composition-chip"
          style={{
            background: 'rgba(168, 85, 247, 0.08)',
            border: '1px solid rgba(168, 85, 247, 0.2)',
            borderRadius: '4px',
            padding: '0.35rem 0.5rem',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            fontSize: '0.78rem'
          }}
        >
          <span style={{ color: '#a855f7', fontWeight: '600', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
            <User size={14} /> Junior
          </span>
          <span style={{ color: '#a855f7', fontWeight: '800' }}>{stats.Junior || 0}</span>
        </div>

        {/* Other (Gray) */}
        <div
          className="composition-chip"
          style={{
            background: 'rgba(156, 163, 175, 0.08)',
            border: '1px solid rgba(156, 163, 175, 0.2)',
            borderRadius: '4px',
            padding: '0.35rem 0.5rem',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            fontSize: '0.78rem'
          }}
        >
          <span style={{ color: '#9ca3af', fontWeight: '600', display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
            <Box size={14} /> Other
          </span>
          <span style={{ color: '#9ca3af', fontWeight: '800' }}>{stats.Other || 0}</span>
        </div>
      </div>

      {/* Selected Evaluation Target Section */}
      {leadSpeaker && (
        <div style={{
          marginTop: '0.5rem',
          paddingTop: '0.5rem',
          borderTop: '1px dashed var(--border-color)',
          fontSize: '0.75rem',
          color: 'var(--text-secondary)',
          display: 'flex',
          flexDirection: 'column',
          gap: '0.2rem'
        }}>
          <span style={{ fontSize: '0.65rem', textTransform: 'uppercase', color: 'var(--text-light)', fontWeight: 600 }}>
            Elected Evaluation Target
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: '0.25rem', color: '#f59e0b', fontWeight: 700 }}>
            👑 {leadSpeaker}
          </span>
        </div>
      )}
    </div>
  );
});

ParticipantsSummary.displayName = 'ParticipantsSummary';
