import React from 'react';
import { capitalize } from '../../utils/speakerUtils';

// Standardized to Green family according to Visual Design System (Priority 7)
const EMOTION_STYLES = {
  happy: { label: 'Happy', bg: 'rgba(16, 185, 129, 0.15)', border: 'rgba(16, 185, 129, 0.4)', text: '#10b981', dot: '#10b981' },
  sad: { label: 'Sad', bg: 'rgba(16, 185, 129, 0.1)', border: 'rgba(16, 185, 129, 0.3)', text: '#10b981', dot: '#6b7280' },
  angry: { label: 'Angry', bg: 'rgba(16, 185, 129, 0.08)', border: 'rgba(16, 185, 129, 0.35)', text: '#10b981', dot: '#ef4444' },
  fear: { label: 'Fear', bg: 'rgba(16, 185, 129, 0.08)', border: 'rgba(16, 185, 129, 0.35)', text: '#10b981', dot: '#a855f7' },
  surprise: { label: 'Surprise', bg: 'rgba(16, 185, 129, 0.12)', border: 'rgba(16, 185, 129, 0.4)', text: '#10b981', dot: '#f59e0b' },
  neutral: { label: 'Neutral', bg: 'rgba(16, 185, 129, 0.05)', border: 'rgba(16, 185, 129, 0.2)', text: '#10b981', dot: '#9ca3af' },
};

// Standardized to Orange family according to Visual Design System (Priority 7)
const TEMPLATE_STYLES = {
  WarmUp: { border: 'rgba(249, 115, 22, 0.4)', text: '#f97316', bg: 'rgba(249, 115, 22, 0.1)' },
  Praise: { border: 'rgba(249, 115, 22, 0.4)', text: '#f97316', bg: 'rgba(249, 115, 22, 0.1)' },
  PSuggest: { border: 'rgba(249, 115, 22, 0.4)', text: '#f97316', bg: 'rgba(249, 115, 22, 0.1)' },
  NSuggest: { border: 'rgba(249, 115, 22, 0.4)', text: '#f97316', bg: 'rgba(249, 115, 22, 0.1)' },
  Listen: { border: 'rgba(249, 115, 22, 0.4)', text: '#f97316', bg: 'rgba(249, 115, 22, 0.1)' },
  Direct: { border: 'rgba(249, 115, 22, 0.4)', text: '#f97316', bg: 'rgba(249, 115, 22, 0.1)' },
};

// Standardized to Role colors according to Visual Design System (Priority 7)
const ROLE_STYLES = {
  leader: { border: 'rgba(59, 130, 246, 0.4)', text: '#3b82f6', bg: 'rgba(59, 130, 246, 0.15)' }, // Blue
  hr: { border: 'rgba(20, 184, 166, 0.4)', text: '#14b8a6', bg: 'rgba(20, 184, 166, 0.15)' }, // Teal
  junior: { border: 'rgba(168, 85, 247, 0.4)', text: '#a855f7', bg: 'rgba(168, 85, 247, 0.15)' }, // Purple
  other: { border: 'rgba(156, 163, 175, 0.4)', text: '#9ca3af', bg: 'rgba(156, 163, 175, 0.15)' }, // Gray
};

export const EmotionBadge = React.memo(({ emotion, confidence }) => {
  if (!emotion) return null;
  const key = emotion.toLowerCase();
  const style = EMOTION_STYLES[key] || EMOTION_STYLES.neutral;
  const roundedConf = Math.round(confidence * 100);
  
  return (
    <span
      className="badge-emotion"
      style={{
        color: style.text,
        backgroundColor: style.bg,
        borderColor: style.border
      }}
    >
      <span className="dot" style={{ backgroundColor: style.dot }} />
      {style.label} {roundedConf > 0 && <span className="conf">{roundedConf}%</span>}
    </span>
  );
});
EmotionBadge.displayName = 'EmotionBadge';

export const TemplateBadge = React.memo(({ label }) => {
  if (!label || label === 'unknown') return null;
  const style = TEMPLATE_STYLES[label] || { border: 'rgba(249, 115, 22, 0.3)', text: '#f97316', bg: 'rgba(249, 115, 22, 0.05)' };
  
  return (
    <span
      className="badge-template"
      style={{
        borderColor: style.border,
        color: style.text,
        backgroundColor: style.bg
      }}
    >
      {label}
    </span>
  );
});
TemplateBadge.displayName = 'TemplateBadge';

export const RoleBadge = React.memo(({ role, confidence }) => {
  if (!role || role === 'unknown') return null;
  const key = role.toLowerCase();
  
  // Guard mapping "manager" or "Manager" to "leader" presentation layer key
  const finalKey = (key === 'manager') ? 'leader' : key;
  const style = ROLE_STYLES[finalKey] || ROLE_STYLES.other;
  const roundedConf = typeof confidence === 'number' ? Math.round(confidence * 100) : 0;

  return (
    <span
      className="badge-template font-bold"
      style={{
        borderColor: style.border,
        color: style.text,
        backgroundColor: style.bg,
        fontSize: '0.68rem',
        textTransform: 'uppercase',
        letterSpacing: '0.02em'
      }}
    >
      {capitalize(role)} {roundedConf > 0 ? `(${roundedConf}%)` : ''}
    </span>
  );
});
RoleBadge.displayName = 'RoleBadge';

export const BadgeGroup = React.memo(({ emotion, emotionConf, templateLabel, role, roleConf }) => {
  return (
    <div className="bubble-badges">
      {emotion && <EmotionBadge emotion={emotion} confidence={emotionConf} />}
      {templateLabel && <TemplateBadge label={templateLabel} />}
      {role && <RoleBadge role={role} confidence={roleConf} />}
    </div>
  );
});
BadgeGroup.displayName = 'BadgeGroup';
