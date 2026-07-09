import { useState, useMemo } from 'react';
import { mapRoleLabel } from '../utils/speakerUtils';

export const useTranscriptFilters = (results = []) => {
  const [searchTerm, setSearchTerm] = useState("");
  const [speakerFilter, setSpeakerFilter] = useState("all");
  const [roleFilter, setRoleFilter] = useState("all");
  const [templateFilter, setTemplateFilter] = useState("all");
  const [emotionFilter, setEmotionFilter] = useState("all");
  const [collapseRepeated, setCollapseRepeated] = useState(false);
  const [visibleCount, setVisibleCount] = useState(50);
  
  const [selectedSegmentId, setSelectedSegmentId] = useState(null);
  const [playingSegmentId, setPlayingSegmentId] = useState(null);

  // Memoize filtered results for performance
  const filteredResults = useMemo(() => {
    if (!results || !Array.isArray(results)) return [];
    
    return results.filter(row => {
      const matchesSearch = !searchTerm || row.text.toLowerCase().includes(searchTerm.toLowerCase());
      const matchesSpeaker = speakerFilter === "all" || row.speaker === speakerFilter;
      
      // Enforce Leader presentation mapping on role filters matching
      const rowRole = mapRoleLabel(row.role || "Other");
      const matchesRole = roleFilter === "all" || rowRole.toLowerCase() === roleFilter.toLowerCase();
      
      const matchesTemplate = templateFilter === "all" || row.template_label === templateFilter;
      
      const rowEmo = row.emotion ? row.emotion.split(' ')[0].toLowerCase() : 'neutral';
      const matchesEmotion = emotionFilter === "all" || rowEmo === emotionFilter.toLowerCase();
      
      return matchesSearch && matchesSpeaker && matchesRole && matchesTemplate && matchesEmotion;
    });
  }, [results, searchTerm, speakerFilter, roleFilter, templateFilter, emotionFilter]);

  // Memoize grouped results to prevent excessive re-renders when rendering 200+ cards
  const renderedSegments = useMemo(() => {
    if (!collapseRepeated) {
      return filteredResults.slice(0, visibleCount).map((row, idx) => ({
        ...row,
        originalIndex: idx,
        isGrouped: false,
        textParts: [{ text: row.text, segment_id: row.segment_id, end_time: row.end_time }]
      }));
    }

    const grouped = [];
    filteredResults.forEach((row, idx) => {
      if (grouped.length > 0 && grouped[grouped.length - 1].speaker === row.speaker) {
        const last = grouped[grouped.length - 1];
        last.textParts.push({ text: row.text, segment_id: row.segment_id, end_time: row.end_time });
        last.end_time = row.end_time;
      } else {
        grouped.push({
          ...row,
          textParts: [{ text: row.text, segment_id: row.segment_id, end_time: row.end_time }],
          isGrouped: true,
          originalIndex: idx
        });
      }
    });
    return grouped.slice(0, visibleCount);
  }, [filteredResults, collapseRepeated, visibleCount]);

  const loadMore = () => {
    setVisibleCount(prev => Math.min(prev + 50, filteredResults.length));
  };

  const jumpToSegment = (direction) => {
    if (filteredResults.length === 0) return;
    const currentIdx = filteredResults.findIndex(r => r.segment_id === selectedSegmentId);
    let nextIdx = 0;
    
    if (direction === 'next') {
      nextIdx = currentIdx + 1 < filteredResults.length ? currentIdx + 1 : 0;
    } else {
      nextIdx = currentIdx - 1 >= 0 ? currentIdx - 1 : filteredResults.length - 1;
    }
    
    const targetSeg = filteredResults[nextIdx];
    if (targetSeg) {
      setSelectedSegmentId(targetSeg.segment_id);
      const element = document.getElementById(`bubble-${targetSeg.segment_id}`);
      if (element) {
        element.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      }
    }
  };

  const clearFilters = () => {
    setSearchTerm("");
    setSpeakerFilter("all");
    setRoleFilter("all");
    setTemplateFilter("all");
    setEmotionFilter("all");
    setCollapseRepeated(false);
    setVisibleCount(50);
  };

  return {
    searchTerm, setSearchTerm,
    speakerFilter, setSpeakerFilter,
    roleFilter, setRoleFilter,
    templateFilter, setTemplateFilter,
    emotionFilter, setEmotionFilter,
    collapseRepeated, setCollapseRepeated,
    visibleCount, loadMore,
    selectedSegmentId, setSelectedSegmentId,
    playingSegmentId, setPlayingSegmentId,
    filteredResults,
    renderedSegments,
    jumpToSegment,
    clearFilters
  };
};
