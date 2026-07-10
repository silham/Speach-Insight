import React from 'react';
import { CATEGORY_NAMES } from '../../utils/evaluationUtils';
import { Search, User, Briefcase, LayoutTemplate, Smile, ChevronUp, ChevronDown, X } from 'lucide-react';

/**
 * TranscriptFilters — Unified command bar.
 * Merges search, dropdown filters, collapse toggle, prev/next navigation, and clear.
 * Two rows: filter row + toolbar row.
 */
export const TranscriptFilters = React.memo(({
  searchTerm,
  onSearchChange,
  speakerFilter,
  onSpeakerChange,
  speakersList = [],
  roleFilter,
  onRoleChange,
  templateFilter,
  onTemplateChange,
  emotionFilter,
  onEmotionChange,
  // Toolbar props (merged from TranscriptToolbar)
  collapseRepeated,
  onToggleCollapse,
  selectedSegmentId,
  onClearSelection,
  onJumpSegment,
  totalResultsCount,
}) => {
  return (
    <div className="command-bar">
      {/* Row 1: Search + Filter Dropdowns */}
      <div className="command-bar-row">
        <div className="search-input-wrapper">
          <Search style={{ position: 'absolute', left: '8px', top: '50%', transform: 'translateY(-50%)', color: '#71717A', width: '15px', height: '15px' }} />
          <input
            type="text"
            className="filter-text-input"
            placeholder="Search transcript..."
            value={searchTerm}
            onChange={(e) => onSearchChange(e.target.value)}
          />
        </div>

        <div className="select-filter-wrapper">
          <User style={{ position: 'absolute', left: '8px', top: '50%', transform: 'translateY(-50%)', color: '#71717A', width: '13px', height: '13px', pointerEvents: 'none' }} />
          <select className="filter-select-input" value={speakerFilter} onChange={(e) => onSpeakerChange(e.target.value)}>
            <option value="all">All Speakers</option>
            {speakersList.map(spk => (
              <option key={spk} value={spk}>{spk}</option>
            ))}
          </select>
        </div>

        <div className="select-filter-wrapper">
          <Briefcase style={{ position: 'absolute', left: '8px', top: '50%', transform: 'translateY(-50%)', color: '#71717A', width: '13px', height: '13px', pointerEvents: 'none' }} />
          <select className="filter-select-input" value={roleFilter} onChange={(e) => onRoleChange(e.target.value)}>
            <option value="all">All Roles</option>
            <option value="leader">Leader</option>
            <option value="hr">HR</option>
            <option value="junior">Junior</option>
            <option value="other">Other</option>
          </select>
        </div>

        <div className="select-filter-wrapper">
          <LayoutTemplate style={{ position: 'absolute', left: '8px', top: '50%', transform: 'translateY(-50%)', color: '#71717A', width: '13px', height: '13px', pointerEvents: 'none' }} />
          <select className="filter-select-input" value={templateFilter} onChange={(e) => onTemplateChange(e.target.value)}>
            <option value="all">All Templates</option>
            {Object.entries(CATEGORY_NAMES).map(([key, label]) => (
              <option key={key} value={key}>{label}</option>
            ))}
          </select>
        </div>

        <div className="select-filter-wrapper">
          <Smile style={{ position: 'absolute', left: '8px', top: '50%', transform: 'translateY(-50%)', color: '#71717A', width: '13px', height: '13px', pointerEvents: 'none' }} />
          <select className="filter-select-input" value={emotionFilter} onChange={(e) => onEmotionChange(e.target.value)}>
            <option value="all">All Emotions</option>
            <option value="happy">Happy</option>
            <option value="neutral">Neutral</option>
            <option value="sad">Sad</option>
            <option value="angry">Angry</option>
            <option value="fear">Fear</option>
            <option value="surprise">Surprise</option>
          </select>
        </div>
      </div>

      {/* Row 2: Toolbar — Collapse toggle, navigation, count */}
      <div className="toolbar-row">
        <div className="toolbar-left">
          <label className="toolbar-toggle">
            <input
              type="checkbox"
              checked={collapseRepeated}
              onChange={(e) => onToggleCollapse(e.target.checked)}
            />
            Collapse turns
          </label>

          <button className="toolbar-btn" onClick={() => onJumpSegment('prev')} title="Previous turn">
            <ChevronUp size={13} /> Prev
          </button>
          <button className="toolbar-btn" onClick={() => onJumpSegment('next')} title="Next turn">
            <ChevronDown size={13} /> Next
          </button>

          {selectedSegmentId !== null && (
            <button className="toolbar-btn danger" onClick={onClearSelection}>
              <X size={12} /> Clear
            </button>
          )}
        </div>

        <div className="toolbar-right">
          {selectedSegmentId !== null && (
            <span className="toolbar-count">Turn #{selectedSegmentId}</span>
          )}
          <span className="toolbar-count">{totalResultsCount} turns</span>
        </div>
      </div>
    </div>
  );
});

TranscriptFilters.displayName = 'TranscriptFilters';
