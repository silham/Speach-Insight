import React from 'react';

export const Skeleton = React.memo(({ type = 'text', count = 1 }) => {
  const renderItem = (idx) => {
    if (type === 'card') {
      return (
        <div key={idx} className="card-panel skeleton-card">
          <div className="skeleton-line skeleton-title" style={{ width: '40%' }} />
          <div className="skeleton-line" style={{ width: '80%', height: '1.25rem' }} />
          <div className="skeleton-line" style={{ width: '60%' }} />
          <div className="skeleton-line" style={{ width: '90%' }} />
        </div>
      );
    }
    if (type === 'bubble') {
      return (
        <div key={idx} className="bubble-wrapper skeleton-bubble">
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
            <div className="skeleton-line" style={{ width: '25%', height: '1rem', borderRadius: '4px' }} />
            <div className="skeleton-line" style={{ width: '15%' }} />
          </div>
          <div className="skeleton-line" style={{ width: '95%' }} />
          <div className="skeleton-line" style={{ width: '80%' }} />
          <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.5rem' }}>
            <div className="skeleton-line" style={{ width: '60px', height: '14px', borderRadius: '999px' }} />
            <div className="skeleton-line" style={{ width: '80px', height: '14px', borderRadius: '999px' }} />
          </div>
        </div>
      );
    }
    if (type === 'accordion') {
      return (
        <div key={idx} className="eval-row skeleton-eval">
          <div className="eval-summary" style={{ height: '48px', padding: '0.75rem 0.95rem' }}>
            <div className="skeleton-line" style={{ width: '20%' }} />
            <div className="skeleton-line" style={{ flex: 1, margin: '0 1rem' }} />
            <div className="skeleton-line" style={{ width: '24px', height: '24px', borderRadius: '50%' }} />
          </div>
        </div>
      );
    }
    // Default text line skeleton
    return (
      <div key={idx} className="skeleton-line-container">
        <div className="skeleton-line" style={{ width: '100%' }} />
      </div>
    );
  };

  return (
    <div className="skeleton-container">
      {Array.from({ length: count }).map((_, i) => renderItem(i))}
    </div>
  );
});

Skeleton.displayName = 'Skeleton';
