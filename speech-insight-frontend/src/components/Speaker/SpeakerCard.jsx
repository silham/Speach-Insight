import React from 'react';
import { Star } from 'lucide-react';
import { capitalize } from '../../utils/speakerUtils';

/**
 * SpeakerCard — Restructured Header Layout.
 * Features:
 * - Clean vertical flex layout (Name row -> Badge row -> Behaviour chips).
 * - circular speaker initials avatar.
 * - Flat badge row for Evaluation Leader (Gold), Role, and Emotion pills.
 * - Decoupled and flexible heights, auto-wrapping on small viewports.
 */
export const SpeakerCard = React.memo(({
  spk,
  isSelected,
  onClick,
  devMode = false,
}) => {
  const confPct = typeof spk.confidence === 'number' ? Math.round(spk.confidence * 100) : 0;

  // Role badge colors
  const roleBg = spk.theme.bg;
  const roleBorder = spk.theme.border;
  const roleColor = spk.theme.text;

  // Dynamic emotion badge coloring helper
  const getEmotionColors = (emo) => {
    const e = String(emo || '').toLowerCase();
    if (e.includes('pos') || e.includes('happy') || e.includes('joy') || e.includes('excit')) {
      return { bg: 'rgba(16, 185, 129, 0.1)', border: 'rgba(16, 185, 129, 0.3)', text: '#10b981' };
    }
    if (e.includes('neg') || e.includes('sad') || e.includes('angr') || e.includes('fear') || e.includes('frust')) {
      return { bg: 'rgba(239, 68, 68, 0.1)', border: 'rgba(239, 68, 68, 0.3)', text: '#ef4444' };
    }
    return { bg: 'rgba(156, 163, 175, 0.1)', border: 'rgba(156, 163, 175, 0.3)', text: '#9ca3af' };
  };
  const emoColors = getEmotionColors(spk.dominantEmotion);

  // Avatar initials getter (e.g. SPEAKER_01 -> S1)
  const avatarInitials = spk.name.includes('_') ? 'S' + spk.name.split('_').pop() : spk.name.substring(0, 2).toUpperCase();

  return (
    <div
      className={`speaker-row ${spk.isLead ? 'lead-row' : ''} ${isSelected ? 'selected-row' : ''}`}
      onClick={onClick}
      style={{
        display: 'flex',
        flexDirection: 'column',
        padding: '1.25rem',
        borderBottom: '1px solid var(--border-color)',
        cursor: 'pointer',
        transition: 'background-color 0.15s ease',
        gap: '0.75rem'
      }}
    >
      {/* Container: Left side details vs Right side metrics */}
      <div 
        className="speaker-card-content" 
        style={{
          display: 'flex',
          flexDirection: 'row',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          flexWrap: 'wrap',
          gap: '1rem',
          width: '100%'
        }}
      >
        {/* Left Side: Speaker Profile Info (Avatar + Name, Badges, Behavior Chips) */}
        <div 
          className="speaker-card-left" 
          style={{
            display: 'flex',
            flexDirection: 'column',
            flex: '1 1 300px',
            gap: '8px'
          }}
        >
          {/* Row 1: Avatar + Speaker Name */}
          <div 
            className="speaker-avatar-name" 
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '10px'
            }}
          >
            <div 
              className="speaker-avatar" 
              style={{
                width: '32px',
                height: '32px',
                borderRadius: '50%',
                backgroundColor: roleBg,
                border: `1px solid ${roleBorder}`,
                color: roleColor,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontWeight: '700',
                fontSize: '0.8rem',
                flexShrink: 0
              }}
            >
              {avatarInitials}
            </div>
            <span 
              className="name-text" 
              style={{
                fontSize: '0.95rem',
                fontWeight: '700',
                color: 'var(--text-main)'
              }}
            >
              {spk.name}
            </span>
          </div>

          {/* Row 2: Badges Container */}
          <div 
            className="speaker-badges-row" 
            style={{
              display: 'flex',
              flexWrap: 'wrap',
              alignItems: 'center',
              gap: '12px',
              marginTop: '4px',
              marginBottom: '4px'
            }}
          >
            {spk.isLead && (
              <span 
                className="evaluation-leader-badge" 
                style={{
                  backgroundColor: 'rgba(245, 158, 11, 0.15)',
                  borderColor: 'rgba(245, 158, 11, 0.4)',
                  color: '#f59e0b',
                  border: '1px solid',
                  fontSize: '13px',
                  padding: '10px 18px',
                  borderRadius: '12px',
                  fontWeight: '600',
                  textTransform: 'uppercase',
                  display: 'inline-flex',
                  alignItems: 'center',
                  gap: '0.3rem',
                  lineHeight: '1.2',
                  whiteSpace: 'nowrap'
                }}
              >
                👑 EVALUATION LEADER
              </span>
            )}

            <span
              className="role-badge-pill"
              style={{
                backgroundColor: roleBg,
                borderColor: roleBorder,
                color: roleColor,
                border: '1px solid',
                fontSize: '13px',
                padding: '10px 18px',
                borderRadius: '12px',
                fontWeight: '600',
                display: 'inline-flex',
                alignItems: 'center',
                lineHeight: '1.2',
                whiteSpace: 'nowrap'
              }}
            >
              {capitalize(spk.predictedRole)} ({confPct}%)
            </span>

            <span
              className="emotion-badge-pill"
              style={{
                backgroundColor: emoColors.bg,
                borderColor: emoColors.border,
                color: emoColors.text,
                border: '1px solid',
                fontSize: '13px',
                padding: '10px 18px',
                borderRadius: '12px',
                fontWeight: '600',
                display: 'inline-flex',
                alignItems: 'center',
                lineHeight: '1.2',
                whiteSpace: 'nowrap',
                textTransform: 'capitalize'
              }}
            >
              {spk.dominantEmotion}
            </span>
          </div>

          {/* Row 3: Behavior Chips */}
          <div 
            className="speaker-row-secondary" 
            style={{
              display: 'flex',
              flexWrap: 'wrap',
              gap: '10px',
              padding: 0,
              marginTop: '4px'
            }}
          >
            {spk.behavior.questions > 0 && (
              <span className="behaviour-chip" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem', fontSize: '0.68rem', color: 'var(--text-light)' }}>
                Questions <span className="chip-count" style={{ fontWeight: 700, color: 'var(--text-secondary)' }}>{spk.behavior.questions}</span>
              </span>
            )}
            {spk.behavior.suggestions > 0 && (
              <span className="behaviour-chip" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem', fontSize: '0.68rem', color: 'var(--text-light)' }}>
                Suggestions <span className="chip-count" style={{ fontWeight: 700, color: 'var(--text-secondary)' }}>{spk.behavior.suggestions}</span>
              </span>
            )}
            {spk.behavior.praise > 0 && (
              <span className="behaviour-chip" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem', fontSize: '0.68rem', color: 'var(--text-light)' }}>
                Praise <span className="chip-count" style={{ fontWeight: 700, color: 'var(--text-secondary)' }}>{spk.behavior.praise}</span>
              </span>
            )}
            {spk.behavior.listening > 0 && (
              <span className="behaviour-chip" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem', fontSize: '0.68rem', color: 'var(--text-light)' }}>
                Listening <span className="chip-count" style={{ fontWeight: 700, color: 'var(--text-secondary)' }}>{spk.behavior.listening}</span>
              </span>
            )}
            {spk.behavior.direct > 0 && (
              <span className="behaviour-chip" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem', fontSize: '0.68rem', color: 'var(--text-light)' }}>
                Directives <span className="chip-count" style={{ fontWeight: 700, color: 'var(--text-secondary)' }}>{spk.behavior.direct}</span>
              </span>
            )}
            {spk.behavior.warmup > 0 && (
              <span className="behaviour-chip" style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem', fontSize: '0.68rem', color: 'var(--text-light)' }}>
                Warmup <span className="chip-count" style={{ fontWeight: 700, color: 'var(--text-secondary)' }}>{spk.behavior.warmup}</span>
              </span>
            )}
            {/* If no behaviours at all, show a subtle fallback */}
            {Object.values(spk.behavior).every(v => v === 0) && (
              <span className="behaviour-chip" style={{ fontStyle: 'italic', fontSize: '0.68rem', color: 'var(--text-light)' }}>General participation</span>
            )}
          </div>
        </div>

        {/* Right Side: Metrics Grid */}
        <div 
          className="speaker-card-right" 
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '1.5rem',
            flex: '0 0 auto',
            minWidth: '240px',
            justifyContent: 'flex-end',
            alignSelf: 'center',
            flexWrap: 'wrap'
          }}
        >
          <div className="row-stat" style={{ display: 'flex', flexDirection: 'column', gap: '0.05rem', minWidth: '70px' }}>
            <span className="row-stat-label" style={{ fontSize: '0.6rem', fontWeight: 600, color: 'var(--text-light)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Time</span>
            <span className="row-stat-value" style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-main)' }}>
              {spk.talkTime}s <span style={{ fontSize: '0.68rem', color: 'var(--text-light)', fontWeight: 500 }}>({spk.percentage}%)</span>
            </span>
          </div>
          
          <div className="row-stat" style={{ display: 'flex', flexDirection: 'column', gap: '0.05rem', minWidth: '40px' }}>
            <span className="row-stat-label" style={{ fontSize: '0.6rem', fontWeight: 600, color: 'var(--text-light)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Turns</span>
            <span className="row-stat-value" style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-main)' }}>{spk.turns}</span>
          </div>
          
          <div className="row-stat" style={{ display: 'flex', flexDirection: 'column', gap: '0.05rem', minWidth: '50px' }}>
            <span className="row-stat-label" style={{ fontSize: '0.6rem', fontWeight: 600, color: 'var(--text-light)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Avg Turn</span>
            <span className="row-stat-value" style={{ fontSize: '0.82rem', fontWeight: 700, color: 'var(--text-main)' }}>{spk.avgDuration}s</span>
          </div>
          
          <div className="row-stat" style={{ display: 'flex', flexDirection: 'column', gap: '0.05rem', minWidth: '60px' }}>
            <span className="row-stat-label" style={{ fontSize: '0.6rem', fontWeight: 600, color: 'var(--text-light)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>Sentiment</span>
            <span 
              className="row-stat-value" 
              style={{
                fontSize: '0.82rem',
                fontWeight: 700,
                color: parseFloat(spk.avgSentiment) >= 0.05 ? '#10b981' : parseFloat(spk.avgSentiment) <= -0.05 ? '#ef4444' : 'var(--text-main)'
              }}
            >
              {spk.avgSentiment}
            </span>
          </div>
        </div>
      </div>

      {/* Row 4: Provenance Flow (if devMode is active) */}
      {devMode && spk.xgboost && (
        <div 
          className="speaker-row-provenance" 
          style={{
            marginTop: '0.45rem',
            paddingTop: '0.45rem',
            borderTop: '1px dashed var(--border-color)',
            fontSize: '0.7rem',
            color: 'var(--text-light)',
            display: 'flex',
            flexWrap: 'wrap',
            alignItems: 'center',
            gap: '0.35rem',
            width: '100%'
          }}
        >
          <span style={{ fontWeight: 600 }}>Classification Path:</span>
          <span style={{ padding: '0.05rem 0.25rem', backgroundColor: 'rgba(255,255,255,0.03)', borderRadius: '3px' }}>
            XGBoost: {capitalize(spk.xgboost.role)} ({Math.round(spk.xgboost.confidence * 100)}%)
          </span>
          <span style={{ color: 'var(--text-muted)' }}>→</span>
          {spk.gemini && spk.gemini.used ? (
            <>
              <span style={{ padding: '0.05rem 0.25rem', backgroundColor: 'rgba(20, 184, 166, 0.1)', color: '#14b8a6', borderRadius: '3px', border: '1px solid rgba(20, 184, 166, 0.2)' }}>
                Gemini Fallback: {capitalize(spk.gemini.role)}
              </span>
              <span style={{ color: 'var(--text-muted)' }}>→</span>
              <span style={{ padding: '0.05rem 0.25rem', backgroundColor: 'rgba(168, 85, 247, 0.1)', color: '#a855f7', borderRadius: '3px', fontWeight: 600 }}>
                Final Role: {capitalize(spk.finalRole)} (via Gemini)
              </span>
            </>
          ) : (
            <>
              <span style={{ padding: '0.05rem 0.25rem', backgroundColor: 'rgba(59, 130, 246, 0.1)', color: '#3b82f6', borderRadius: '3px', border: '1px solid rgba(59, 130, 246, 0.2)' }}>
                Direct (Conf ≥ 80%)
              </span>
              <span style={{ color: 'var(--text-muted)' }}>→</span>
              <span style={{ padding: '0.05rem 0.25rem', backgroundColor: 'rgba(59, 130, 246, 0.1)', color: '#3b82f6', borderRadius: '3px', fontWeight: 600 }}>
                Final Role: {capitalize(spk.finalRole)} (via XGBoost)
              </span>
            </>
          )}
        </div>
      )}
    </div>
  );
});

SpeakerCard.displayName = 'SpeakerCard';
