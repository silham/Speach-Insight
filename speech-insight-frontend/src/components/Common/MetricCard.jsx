import React from 'react';

export const MetricCard = React.memo(({ title, value, icon, description, style = {}, valueStyle = {} }) => {
  return (
    <div className="card-panel metric-card" style={style}>
      <div className="metric-card-header">
        <span className="metric-card-title">{title}</span>
        {icon && <span className="metric-card-icon">{icon}</span>}
      </div>
      <div className="metric-card-value" style={valueStyle}>{value}</div>
      {description && <div className="metric-card-description">{description}</div>}
    </div>
  );
});

MetricCard.displayName = 'MetricCard';
