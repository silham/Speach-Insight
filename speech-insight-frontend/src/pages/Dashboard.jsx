import React, { useState, useEffect, useRef, useMemo } from 'react';
import axios from 'axios';
import { Skeleton } from '../components/Common/Skeleton';
import { BadgeGroup, EmotionBadge, RoleBadge } from '../components/Common/BadgeGroup';
import { EvaluationAccordion } from '../components/Evaluation/EvaluationAccordion';
import { EvaluationSummary } from '../components/Evaluation/EvaluationSummary';
import { SpeakerCard } from '../components/Speaker/SpeakerCard';
import { TranscriptCard } from '../components/Transcript/TranscriptCard';
import { TranscriptFilters } from '../components/Transcript/TranscriptFilters';
import { useTranscriptFilters } from '../hooks/useTranscriptFilters';
import { useSpeakerAnalysis } from '../hooks/useSpeakerAnalysis';
import { getSpeakerTheme, mapRoleLabel } from '../utils/speakerUtils';
import { Clock, MessageSquare, Users, Target, AlertTriangle, Search } from 'lucide-react';

export const Dashboard = () => {
  const [activeTab, setActiveTab] = useState('pipeline');

  // Theme state — persisted in localStorage
  const [isDark, setIsDark] = useState(() => {
    const saved = localStorage.getItem('si-theme');
    return saved !== null ? saved === 'dark' : true;
  });

  const toggleTheme = () => {
    setIsDark(prev => {
      const next = !prev;
      localStorage.setItem('si-theme', next ? 'dark' : 'light');
      return next;
    });
  };

  // Pipeline execution state
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState("");
  const [results, setResults] = useState([]);
  const [report, setReport] = useState(null);
  const [metadata, setMetadata] = useState(null);

  // RAG upload state
  const [ragFile, setRagFile] = useState(null);
  const [ragLoading, setRagLoading] = useState(false);
  const [ragStatus, setRagStatus] = useState("");

  const [expandedCategory, setExpandedCategory] = useState(null);
  const [rightPanelTab, setRightPanelTab] = useState('report');
  const [devMode, setDevMode] = useState(false);
  const [isSpeakerProfilesOpen, setIsSpeakerProfilesOpen] = useState(true);
  const [isTranscriptsOpen, setIsTranscriptsOpen] = useState(true);

  const audioInstanceRef = useRef(null);

  // Hook for filtering transcript
  const {
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
  } = useTranscriptFilters(results);

  // Hook for speaker analysis
  const {
    speakersList,
    speakerAnalysisData,
    compositionStats,
    totalDuration
  } = useSpeakerAnalysis(metadata, results, report);

  useEffect(() => {
    return () => {
      if (audioInstanceRef.current) {
        audioInstanceRef.current.pause();
      }
    };
  }, []);

  const handleFileChange = (e) => {
    if (e.target.files[0]) {
      setFile(e.target.files[0]);
      setStatus("");
    }
  };

  const handleUpload = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    setLoading(true);
    setStatus("Initiating speech transcription, emotion mapping, and compliance scoring...");
    setResults([]);
    setReport(null);
    setMetadata(null);
    setSelectedSegmentId(null);
    setPlayingSegmentId(null);

    try {
      const response = await axios.post("http://127.0.0.1:8000/analyze", formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      const data = response.data;
      setResults(data.data);
      setMetadata({
        job_id: data.job_id,
        lead_speaker: data.lead_speaker,
        total_speakers: data.total_speakers,
        total_segments: data.total_segments,
        total_duration: data.total_duration,
        speaker_roles: data.speaker_roles || {},
        filename: file.name
      });
      
      if (data.data.length > 0) {
        setSelectedSegmentId(data.data[0].segment_id);
      }

      setStatus("Analysis completed successfully.");

      // Fetch performance report
      try {
        const reportRes = await axios.get(`http://127.0.0.1:8000/report/${data.job_id}`);
        setReport(reportRes.data);
      } catch (reportErr) {
        console.warn("Performance report not available:", reportErr);
      }
    } catch (error) {
      console.error(error);
      setStatus("Error: " + (error.response?.data?.detail || "Could not connect to service"));
    } finally {
      setLoading(false);
    }
  };

  // RAG functions
  const handleRagFileChange = (e) => {
    if (e.target.files[0]) {
      setRagFile(e.target.files[0]);
      setRagStatus("");
    }
  };

  const handleRagUpload = async () => {
    if (!ragFile) return;

    const formData = new FormData();
    formData.append("file", ragFile);

    setRagLoading(true);
    setRagStatus("Indexing reference guidelines document...");

    try {
      const response = await axios.post("http://127.0.0.1:8000/rag/upload", formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setRagStatus(`Success! Indexed ${response.data.chunks_added} segments into ChromaDB guidelines base.`);
    } catch (error) {
      console.error(error);
      setRagStatus("Error: " + (error.response?.data?.detail || "Upload failed"));
    } finally {
      setRagLoading(false);
      setRagFile(null);
    }
  };

  // Play a segment clip
  const togglePlaySegment = (url, segmentId) => {
    if (playingSegmentId === segmentId) {
      if (audioInstanceRef.current) {
        audioInstanceRef.current.pause();
      }
      setPlayingSegmentId(null);
      return;
    }

    if (audioInstanceRef.current) {
      audioInstanceRef.current.pause();
    }

    const fullUrl = url.startsWith('http') ? url : `http://127.0.0.1:8000${url}`;
    const audio = new Audio(fullUrl);
    audioInstanceRef.current = audio;
    setPlayingSegmentId(segmentId);

    audio.play().catch(e => {
      console.error("Audio playback error:", e);
      setPlayingSegmentId(null);
    });

    audio.onended = () => {
      setPlayingSegmentId(null);
    };
  };

  const handleSegmentPlayClick = (segmentId) => {
    const matched = results.find(r => r.segment_id === segmentId);
    if (matched && matched.audio_url) {
      togglePlaySegment(matched.audio_url, segmentId);
    }
  };

  const handleSpeakerCardClick = (spkName) => {
    if (speakerFilter === spkName) {
      setSpeakerFilter("all");
    } else {
      setSpeakerFilter(spkName);
      const firstUtterance = results.find(r => r.speaker === spkName);
      if (firstUtterance) {
        setSelectedSegmentId(firstUtterance.segment_id);
        setTimeout(() => {
          const element = document.getElementById(`bubble-${firstUtterance.segment_id}`);
          if (element) {
            element.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
          }
        }, 100);
      }
    }
  };

  const activeSegment = useMemo(() => {
    return results.find(r => r.segment_id === selectedSegmentId) || null;
  }, [results, selectedSegmentId]);

  const activeSegmentSpeakerTheme = useMemo(() => {
    if (!activeSegment) return null;
    return getSpeakerTheme(activeSegment.speaker, speakersList);
  }, [activeSegment, speakersList]);

  // Overall emotion tone calculations
  const emotionVibeStats = useMemo(() => {
    if (!results || results.length === 0) return [];
    const emotionSummary = {};
    results.forEach(r => {
      const emo = r.emotion ? r.emotion.split(' ')[0].toLowerCase() : 'neutral';
      emotionSummary[emo] = (emotionSummary[emo] || 0) + 1;
    });
    return Object.entries(emotionSummary)
      .map(([emo, count]) => ({
        name: emo,
        percentage: ((count / results.length) * 100).toFixed(0)
      }))
      .sort((a, b) => b.percentage - a.percentage);
  }, [results]);

  // Scroll listener for virtual rendering
  const handleTranscriptScroll = (e) => {
    const { scrollTop, scrollHeight, clientHeight } = e.currentTarget;
    if (scrollHeight - scrollTop - clientHeight < 150) {
      loadMore();
    }
  };

  // Build participants detail string for KPI ribbon
  const participantsDetail = useMemo(() => {
    const parts = [];
    if (compositionStats.Leader > 0) parts.push(`Leader ${compositionStats.Leader}`);
    if (compositionStats.HR > 0) parts.push(`HR ${compositionStats.HR}`);
    if (compositionStats.Junior > 0) parts.push(`Junior ${compositionStats.Junior}`);
    if (compositionStats.Other > 0) parts.push(`Other ${compositionStats.Other}`);
    return parts.join(' · ');
  }, [compositionStats]);

  return (
    <div className="saas-layout" data-theme={isDark ? 'dark' : 'light'}>
      {/* 1. Sidebar Navigation */}
      <aside className="sidebar-nav">
        <div className="sidebar-brand">
          <div className="brand-icon-wrapper">
            <svg className="brand-logo-icon" fill="none" stroke="currentColor" strokeWidth="2.5" viewBox="0 0 24 24" strokeLinecap="round" strokeLinejoin="round">
              <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"></path>
              <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
              <line x1="12" x2="12" y1="19" y2="22"></line>
            </svg>
          </div>
          <div className="brand-details">
            <h3>SpeechInSight</h3>
            <span>Analytics Engine</span>
          </div>
          <button
            className="theme-toggle-btn"
            onClick={toggleTheme}
            title={isDark ? 'Switch to Light Mode' : 'Switch to Dark Mode'}
            style={{ marginLeft: 'auto' }}
          >
            {isDark ? (
              /* Sun icon — click to go light */
              <svg className="theme-toggle-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="5" />
                <line x1="12" y1="1" x2="12" y2="3" />
                <line x1="12" y1="21" x2="12" y2="23" />
                <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
                <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
                <line x1="1" y1="12" x2="3" y2="12" />
                <line x1="21" y1="12" x2="23" y2="12" />
                <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
                <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
              </svg>
            ) : (
              /* Moon icon — click to go dark */
              <svg className="theme-toggle-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
              </svg>
            )}
          </button>
        </div>

        <nav className="nav-menu">
          <button
            className={`nav-item-btn ${activeTab === 'pipeline' ? 'active' : ''}`}
            onClick={() => setActiveTab('pipeline')}
          >
            <svg className="nav-icon" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
              <rect width="7" height="9" x="3" y="3" rx="1"></rect>
              <rect width="7" height="5" x="14" y="3" rx="1"></rect>
              <rect width="7" height="9" x="14" y="12" rx="1"></rect>
              <rect width="7" height="5" x="3" y="16" rx="1"></rect>
            </svg>
            Dialogue Browser
          </button>

          <button
            className={`nav-item-btn ${activeTab === 'report' ? 'active' : ''}`}
            onClick={() => setActiveTab('report')}
            disabled={!report}
            title={!report ? "Perform speech analysis first to view reports" : ""}
          >
            <svg className="nav-icon" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
              <path d="M21.21 15.89A10 10 0 1 1 8 2.83"></path>
              <path d="M22 12A10 10 0 0 0 12 2v10z"></path>
            </svg>
            Evaluation Base
          </button>

          <button
            className={`nav-item-btn ${activeTab === 'guidelines' ? 'active' : ''}`}
            onClick={() => setActiveTab('guidelines')}
          >
            <svg className="nav-icon" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
              <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path>
            </svg>
            Guideline Base
          </button>
        </nav>

        {metadata && (
          <div className="sidebar-job-badge">
            <span className="badge-title">Active Session</span>
            <span className="badge-filename" title={metadata.filename}>{metadata.filename}</span>
            <div className="badge-row">
              <span>{metadata.total_speakers} speakers</span>
              <span className="dot-sep" />
              <span>{metadata.total_duration.toFixed(0)}s</span>
            </div>
            <button
              className="btn-sidebar-reset"
              onClick={() => {
                setFile(null);
                setResults([]);
                setMetadata(null);
                setReport(null);
                setStatus("");
                clearFilters();
              }}
            >
              Reset Session
            </button>
          </div>
        )}
      </aside>

      {/* 2. Main Dashboard Content */}
      <main className="saas-content">
        {activeTab === 'pipeline' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
            {results.length === 0 ? (
              <div className="upload-container">
                <div className="card-panel upload-card">
                  <div className="upload-icon-wrapper">
                    <svg className="upload-hero-icon" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                      <polyline points="17 8 12 3 7 8"></polyline>
                      <line x1="12" x2="12" y1="3" y2="15"></line>
                    </svg>
                  </div>
                   <h3>Speech Media Upload</h3>
                   <p>Provide a meeting audio or video file (WAV, MP3, FLAC, M4A, AAC, OGG, OPUS, MP4, MOV, MKV, AVI, WebM, MPEG, MPG). The engine will process, diarize, transcribe, and audit compliance guidelines.</p>
 
                   <div className="upload-controls">
                     <label className="file-select-label">
                       Choose Media File
                       <input type="file" className="file-raw-input" accept="audio/*,video/*,.wav,.mp3,.flac,.m4a,.aac,.ogg,.opus,.mp4,.mov,.mkv,.avi,.webm,.mpeg,.mpg" onChange={handleFileChange} />
                     </label>
                     {file && <span className="selected-filename">{file.name}</span>}
                   </div>
 
                   <button
                     className="btn-primary"
                     disabled={!file || loading}
                     onClick={handleUpload}
                   >
                     {loading ? "Processing..." : "Analyze Media"}
                   </button>

                  {status && (
                    <p style={{ marginTop: '1.25rem', fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                      {status}
                    </p>
                  )}
                  {loading && (
                    <div style={{ marginTop: '0.75rem' }}>
                      <Skeleton type="text" count={2} />
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <>
                {/* ─── KPI Stats Ribbon ─── */}
                <div className="stats-ribbon">
                  <div className="stat-cell">
                    <Clock size={16} className="stat-icon" />
                    <div className="stat-content">
                      <span className="stat-label">Duration</span>
                      <span className="stat-value">{metadata.total_duration.toFixed(1)}s</span>
                    </div>
                  </div>
                  <div className="stat-cell">
                    <MessageSquare size={16} className="stat-icon" />
                    <div className="stat-content">
                      <span className="stat-label">Turns</span>
                      <span className="stat-value">{metadata.total_segments}</span>
                    </div>
                  </div>
                  <div className="stat-cell">
                    <Users size={16} className="stat-icon" />
                    <div className="stat-content">
                      <span className="stat-label">Participants</span>
                      <span className="stat-value">
                        {metadata.total_speakers}
                        {participantsDetail && <span className="stat-detail">({participantsDetail})</span>}
                      </span>
                    </div>
                  </div>
                  <div className="stat-cell">
                    <Target size={16} className="stat-icon" />
                    <div className="stat-content">
                      <span className="stat-label">Compliance</span>
                      <span className="stat-value" style={{ color: 'var(--accent)' }}>
                        {report ? `${Math.round(report.total_score)}/100` : "N/A"}
                      </span>
                    </div>
                  </div>
                </div>

                {/* ─── HIGHLIGHTED GUIDELINE SCORES ─── */}
                <div className="guideline-scores-highlight" style={{ marginTop: '1.25rem', padding: '1.5rem', backgroundColor: 'var(--panel-bg)', borderRadius: 'var(--border-radius)', border: '1px solid var(--border-color)', boxShadow: '0 4px 12px rgba(0,0,0,0.05)' }}>
                  <div className="section-header" style={{ marginBottom: '1.25rem' }}>
                    <div className="header-text-group">
                      <h4 style={{ fontSize: '1.1rem', color: 'var(--text-main)', fontWeight: 700 }}>Guideline Scores</h4>
                      <span className="section-subtitle">Overall compliance evaluation based on your RAG base.</span>
                    </div>
                  </div>
                  {report ? (
                    <div className="report-tab-layout">
                      <EvaluationSummary
                        totalScore={report.total_score}
                        vibeStats={emotionVibeStats}
                        strengths={report.strengths}
                        improvements={report.improvements}
                      />
                      <div className="report-categories-list" style={{ marginTop: '0.75rem' }}>
                        {report.categories && report.categories.map((cat, idx) => (
                          <EvaluationAccordion
                            key={idx}
                            cat={cat}
                            results={results}
                            isExpanded={expandedCategory === cat.name}
                            onToggle={() => setExpandedCategory(expandedCategory === cat.name ? null : cat.name)}
                          />
                        ))}
                      </div>
                    </div>
                  ) : (
                    <div className="report-empty-state">
                      <span className="placeholder-icon"><AlertTriangle size={28} /></span>
                      <p style={{ fontSize: '0.78rem', color: 'var(--text-light)', marginTop: '0.5rem' }}>Compliance report not loaded. Ensure Google API Key is set in your environment if using RAG evaluation modules.</p>
                    </div>
                  )}
                </div>

                {/* ─── Speaker Profiles — Horizontal Table ─── */}
                <div className="accordion-container" style={{ marginTop: '1.25rem', border: '1px solid var(--border-color)', borderRadius: 'var(--border-radius)', overflow: 'hidden' }}>
                  <div 
                    className="accordion-header" 
                    onClick={() => setIsSpeakerProfilesOpen(!isSpeakerProfilesOpen)}
                    style={{ padding: '1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', backgroundColor: 'rgba(255,255,255,0.02)', cursor: 'pointer', userSelect: 'none' }}
                  >
                    <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                      <Users size={18} />
                      <h4 style={{ margin: 0, fontSize: '1rem', fontWeight: 600 }}>Speaker Profiles</h4>
                    </div>
                    <span>{isSpeakerProfilesOpen ? "▼" : "▶"}</span>
                  </div>
                  {isSpeakerProfilesOpen && (
                    <div style={{ padding: '1rem', borderTop: '1px solid var(--border-color)' }}>
                      <section className="speaker-analysis-container">
                  <div className="section-header">
                    <div className="header-text-group">
                      <h4>Speaker Profiles</h4>
                      <span className="section-subtitle">Role classification, behavioural metrics, and conversational statistics per participant.</span>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                      <label style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.75rem', cursor: 'pointer', color: 'var(--text-secondary)' }}>
                        <input 
                          type="checkbox" 
                          checked={devMode} 
                          onChange={(e) => setDevMode(e.target.checked)} 
                          style={{ cursor: 'pointer' }}
                        />
                        Developer Mode
                      </label>
                      <span className="toolbar-count">{speakerAnalysisData.length} speakers</span>
                    </div>
                  </div>

                  <div className="speaker-table">
                    {speakerAnalysisData.map((spk) => (
                      <SpeakerCard
                        key={spk.name}
                        spk={spk}
                        isSelected={speakerFilter === spk.name}
                        onClick={() => handleSpeakerCardClick(spk.name)}
                        devMode={devMode}
                      />
                    ))}
                  </div>
                </section>
                    </div>
                  )}
                </div>

                {/* ─── Meeting-Level Leader Resolution Section ─── */}
                {devMode && report && report.leader_resolution && (
                  <section className="leader-resolution-container" style={{
                    marginTop: '1.25rem',
                    padding: '0.85rem 1rem',
                    borderRadius: 'var(--border-radius)',
                    backgroundColor: 'rgba(255,255,255,0.02)',
                    border: '1px solid var(--border-color)',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '0.65rem'
                  }}>
                    <div className="section-header" style={{ marginBottom: 0 }}>
                      <div className="header-text-group">
                        <h4 style={{ fontSize: '0.85rem', color: 'var(--text-main)', fontWeight: 700 }}>
                          Meeting-Level Leader Resolution
                        </h4>
                        <span className="section-subtitle" style={{ fontSize: '0.72rem', color: 'var(--text-light)' }}>
                          Deterministic post-inference target election analysis.
                        </span>
                      </div>
                    </div>
                    
                    <div className="resolution-grid" style={{
                      display: 'grid',
                      gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
                      gap: '0.75rem',
                      marginTop: '0.25rem'
                    }}>
                      <div className="resolution-card" style={{ padding: '0.5rem', backgroundColor: 'rgba(255,255,255,0.01)', borderRadius: '4px', border: '1px solid rgba(255,255,255,0.02)' }}>
                        <span style={{ fontSize: '0.65rem', textTransform: 'uppercase', color: 'var(--text-light)', display: 'block', fontWeight: 600 }}>Resolution Required</span>
                        <span style={{ fontSize: '0.85rem', fontWeight: 700, color: report.leader_resolution.required ? '#f59e0b' : '#10b981', display: 'block', marginTop: '0.15rem' }}>
                          {report.leader_resolution.required ? "Yes (Ambiguity Tied)" : "No (Single Leader)"}
                        </span>
                      </div>
                      
                      <div className="resolution-card" style={{ padding: '0.5rem', backgroundColor: 'rgba(255,255,255,0.01)', borderRadius: '4px', border: '1px solid rgba(255,255,255,0.02)' }}>
                        <span style={{ fontSize: '0.65rem', textTransform: 'uppercase', color: 'var(--text-light)', display: 'block', fontWeight: 600 }}>Candidate Count</span>
                        <span style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--text-main)', display: 'block', marginTop: '0.15rem' }}>
                          {report.leader_resolution.candidate_count} {report.leader_resolution.candidate_count === 1 ? "speaker" : "speakers"}
                        </span>
                      </div>

                      <div className="resolution-card" style={{ padding: '0.5rem', backgroundColor: 'rgba(255,255,255,0.01)', borderRadius: '4px', border: '1px solid rgba(255,255,255,0.02)' }}>
                        <span style={{ fontSize: '0.65rem', textTransform: 'uppercase', color: 'var(--text-light)', display: 'block', fontWeight: 600 }}>Resolution Method</span>
                        <span style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--accent)', display: 'block', marginTop: '0.2rem', textTransform: 'capitalize' }}>
                          {report.leader_resolution.method ? report.leader_resolution.method.replace(/_/g, ' ') : "N/A"}
                        </span>
                      </div>

                      <div className="resolution-card" style={{ padding: '0.5rem', backgroundColor: 'rgba(255,255,255,0.01)', borderRadius: '4px', border: '1px solid rgba(255,255,255,0.02)' }}>
                        <span style={{ fontSize: '0.65rem', textTransform: 'uppercase', color: 'var(--text-light)', display: 'block', fontWeight: 600 }}>Evaluation Target</span>
                        <span style={{ fontSize: '0.85rem', fontWeight: 700, color: '#3b82f6', display: 'block', marginTop: '0.15rem' }}>
                          👑 {report.leader_resolution.selected || "None"}
                        </span>
                      </div>
                    </div>

                    {report.leader_resolution.candidate_speakers && report.leader_resolution.candidate_speakers.length > 0 && (
                      <div className="resolution-candidates" style={{ fontSize: '0.72rem', color: 'var(--text-secondary)' }}>
                        <strong>Candidate Speakers:</strong> {report.leader_resolution.candidate_speakers.join(", ")}
                      </div>
                    )}

                    <div className="resolution-reason-box" style={{
                      padding: '0.5rem 0.75rem',
                      backgroundColor: 'rgba(255,255,255,0.01)',
                      borderRadius: '4px',
                      borderLeft: '3px solid var(--accent)',
                      fontSize: '0.75rem',
                      lineHeight: 1.35,
                      color: 'var(--text-secondary)'
                    }}>
                      <strong>Resolution Log:</strong> {report.leader_resolution.reason}
                    </div>
                  </section>
                )}

                {/* ─── Workspace / Dialogue Section ─── */}
                <div className="accordion-container" style={{ marginTop: '1.25rem', border: '1px solid var(--border-color)', borderRadius: 'var(--border-radius)', overflow: 'hidden' }}>
                  <div 
                    className="accordion-header" 
                    onClick={() => setIsTranscriptsOpen(!isTranscriptsOpen)}
                    style={{ padding: '1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', backgroundColor: 'rgba(255,255,255,0.02)', cursor: 'pointer', userSelect: 'none' }}
                  >
                    <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                      <MessageSquare size={18} />
                      <h4 style={{ margin: 0, fontSize: '1rem', fontWeight: 600 }}>Dialogue Transcripts & Inspector</h4>
                    </div>
                    <span>{isTranscriptsOpen ? "▼" : "▶"}</span>
                  </div>
                  {isTranscriptsOpen && (
                    <div style={{ padding: '1rem', borderTop: '1px solid var(--border-color)' }}>
                      <div className="workspace-split">
                  {/* Left Column: Transcript List */}
                  <div className="card-panel chat-stream-card">
                    <div className="transcript-control-header">
                      <div className="section-header" style={{ marginBottom: 0 }}>
                        <div className="header-text-group">
                          <h4>Dialogue Transcript</h4>
                          <span className="section-subtitle">Segmented turns with audio track timestamps.</span>
                        </div>
                      </div>

                      {/* Unified Command Bar */}
                      <TranscriptFilters
                        searchTerm={searchTerm}
                        onSearchChange={setSearchTerm}
                        speakerFilter={speakerFilter}
                        onSpeakerChange={setSpeakerFilter}
                        speakersList={speakersList}
                        roleFilter={roleFilter}
                        onRoleChange={setRoleFilter}
                        templateFilter={templateFilter}
                        onTemplateChange={setTemplateFilter}
                        emotionFilter={emotionFilter}
                        onEmotionChange={setEmotionFilter}
                        collapseRepeated={collapseRepeated}
                        onToggleCollapse={setCollapseRepeated}
                        selectedSegmentId={selectedSegmentId}
                        onClearSelection={clearFilters}
                        onJumpSegment={jumpToSegment}
                        totalResultsCount={filteredResults.length}
                      />
                    </div>

                    <div className="bubble-list" onScroll={handleTranscriptScroll}>
                      {renderedSegments.length > 0 ? (
                        renderedSegments.map((row) => {
                          const theme = getSpeakerTheme(row.speaker, speakersList);
                          const isActive = row.textParts.some(p => p.segment_id === selectedSegmentId);
                          const isPlaying = row.textParts.some(p => p.segment_id === playingSegmentId);

                          return (
                            <TranscriptCard
                              key={row.segment_id}
                              row={row}
                              isActive={isActive}
                              isPlaying={isPlaying}
                              speakerTheme={theme}
                              onCardClick={setSelectedSegmentId}
                              onPlayClick={handleSegmentPlayClick}
                              activeSegmentId={selectedSegmentId}
                            />
                          );
                        })
                      ) : (
                        <div className="no-filter-results">
                          <p>No dialogue turns match the selected filter criteria.</p>
                          <button className="btn-sidebar-reset" onClick={clearFilters} style={{ width: 'auto', padding: '0.35rem 1rem' }}>Reset Filters</button>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Right Column: Inspector Panels */}
                  <div className="card-panel tabs-container">
                    <div className="details-tab-header">
                      <button className="tab-btn-detail active">
                        Turn Inspector
                      </button>
                    </div>

                    <div className="tab-details-content">
                        {/* Turn Inspector */}
                        {activeSegment ? (
                          <div className="inspector-content">
                            <div className="inspector-meta-row">
                              <span className="inspector-speaker" style={{ color: activeSegmentSpeakerTheme?.text }}>
                                {activeSegment.speaker} (Turn #{activeSegment.segment_id})
                              </span>
                              <span className="inspector-time">
                                {activeSegment.start_time.toFixed(1)}s – {activeSegment.end_time.toFixed(1)}s
                              </span>
                            </div>

                            <p className="inspector-quote">
                              "{activeSegment.text}"
                            </p>

                            <div className="inspector-metrics-section">
                              <h5>Classifications</h5>
                              <div className="metrics-grid" style={{ display: 'flex', flexDirection: 'column', gap: '0.65rem', marginTop: '0.35rem' }}>
                                <div>
                                  <span className="speaker-stat-label">Template</span>
                                  <div style={{ marginTop: '0.15rem' }}>
                                    <span
                                      className="badge-template"
                                      style={{
                                        borderColor: activeSegmentSpeakerTheme?.border,
                                        color: activeSegmentSpeakerTheme?.text,
                                        backgroundColor: activeSegmentSpeakerTheme?.bg
                                      }}
                                    >
                                      {activeSegment.template_label || "No Template Class"}
                                    </span>
                                  </div>
                                </div>

                                <div>
                                  <span className="speaker-stat-label">Emotion</span>
                                  <div style={{ marginTop: '0.15rem' }}>
                                    <EmotionBadge emotion={activeSegment.emotion} confidence={activeSegment.confidence} />
                                  </div>
                                </div>

                                <div>
                                  <span className="speaker-stat-label">Speaker Role & Provenance</span>
                                  <div style={{ marginTop: '0.15rem', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
                                    <RoleBadge role={activeSegment.role} confidence={activeSegment.role_confidence} />
                                    {report && report.lead_speaker === activeSegment.speaker && (
                                      <span 
                                        className="evaluation-leader-badge" 
                                        style={{
                                          backgroundColor: 'rgba(245, 158, 11, 0.15)',
                                          borderColor: 'rgba(245, 158, 11, 0.4)',
                                          color: '#f59e0b',
                                          border: '1px solid',
                                          fontSize: '0.62rem',
                                          padding: '0.05rem 0.3rem',
                                          borderRadius: '3px',
                                          fontWeight: '700',
                                          textTransform: 'uppercase',
                                          display: 'inline-flex',
                                          alignItems: 'center',
                                          gap: '0.2rem'
                                        }}
                                      >
                                        👑 Evaluation Leader
                                      </span>
                                    )}
                                  </div>
                                  {(() => {
                                    const spkAnalysis = speakerAnalysisData.find(s => s.name === activeSegment.speaker);
                                    if (devMode && spkAnalysis && spkAnalysis.xgboost) {
                                      return (
                                        <div style={{ 
                                          marginTop: '0.45rem', 
                                          padding: '0.35rem 0.5rem', 
                                          backgroundColor: 'rgba(255,255,255,0.01)', 
                                          borderRadius: '4px', 
                                          border: '1px solid rgba(255,255,255,0.02)',
                                          fontSize: '0.7rem', 
                                          color: 'var(--text-light)',
                                          display: 'flex',
                                          flexDirection: 'column',
                                          gap: '0.2rem'
                                        }}>
                                          <div><strong>Model (XGBoost):</strong> {spkAnalysis.xgboost.role} ({Math.round(spkAnalysis.xgboost.confidence * 100)}%)</div>
                                          <div><strong>Gemini Fallback:</strong> {spkAnalysis.gemini && spkAnalysis.gemini.used ? `Invoked (Result: ${spkAnalysis.gemini.role})` : "Not Invoked"}</div>
                                          <div><strong>Final Decision:</strong> {spkAnalysis.finalRole}</div>
                                          <div><strong>Source:</strong> {spkAnalysis.predictionSource === 'gemini' ? 'Gemini AI Fallback' : 'XGBoost Direct Model'}</div>
                                        </div>
                                      );
                                    }
                                    return null;
                                  })()}
                                </div>

                                {activeSegment.vader && (
                                  <div>
                                    <span className="speaker-stat-label">Sentiment Score</span>
                                    <div className="vader-compound-bar" style={{ marginTop: '0.2rem' }}>
                                      <div className="vader-track">
                                        <div
                                          className="vader-marker"
                                          style={{ left: `${((activeSegment.vader.compound + 1) / 2) * 100}%` }}
                                        />
                                      </div>
                                      <div className="vader-labels">
                                        <span>Negative</span>
                                        <span className="compound-val">Score: {activeSegment.vader.compound.toFixed(2)}</span>
                                        <span>Positive</span>
                                      </div>
                                    </div>
                                  </div>
                                )}
                              </div>
                            </div>
                          </div>
                        ) : (
                          <div className="inspector-placeholder">
                            <span className="placeholder-icon"><Search size={28} /></span>
                            <p>Select any utterance card in the Dialogue Browser to inspect specific acoustic and text parameters.</p>
                          </div>
                        )}
                    </div>
                  </div>
                </div>
                    </div>
                  )}
                </div>
              </>
            )}
          </div>
        )}

        {activeTab === 'report' && report && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem', maxWidth: '900px' }}>
            <div className="card-panel">
              <h4 style={{ marginBottom: '1rem', fontSize: '0.95rem', fontWeight: 700 }}>Guidelines Scorecard Summary</h4>
              <EvaluationSummary
                totalScore={report.total_score}
                vibeStats={emotionVibeStats}
                strengths={report.strengths}
                improvements={report.improvements}
              />

              <div className="report-categories-list" style={{ marginTop: '1rem' }}>
                {report.categories && report.categories.map((cat, idx) => (
                  <EvaluationAccordion
                    key={idx}
                    cat={cat}
                    results={results}
                    isExpanded={expandedCategory === cat.name}
                    onToggle={() => setExpandedCategory(expandedCategory === cat.name ? null : cat.name)}
                  />
                ))}
              </div>
            </div>

            {/* ── Audio Segment Feedback ── */}
            {report.segment_comments && report.segment_comments.length > 0 && (
              <div className="card-panel segment-feedback-section">
                <h4 className="segment-feedback-title" style={{ marginBottom: '1rem', fontSize: '0.95rem', fontWeight: 700 }}>Audio Segment Feedback</h4>
                <div className="segment-feedback-list" style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                  {report.segment_comments.map((seg, i) => {
                    const transcriptRow = results.find(r => r.segment_id === seg.segment_id);
                    return (
                      <div key={i} className="segment-comment-card">
                        <div className="segment-comment-header">
                          <h5 className="segment-comment-title">
                            {seg.segment_id != null
                              ? `Segment ${seg.segment_id}${transcriptRow ? ` — ${transcriptRow.speaker}` : ''}`
                              : (seg.comment.match(/^(SPEAKER_\d+)/) ? seg.comment.match(/^(SPEAKER_\d+)/)[1] : `Segment ${i + 1}`)}
                          </h5>
                          {transcriptRow && (
                            <button
                              className="segment-comment-link"
                              onClick={() => {
                                setActiveTab('pipeline');
                                jumpToSegment(seg.segment_id);
                              }}
                            >
                              View in Dialogue Browser
                            </button>
                          )}
                        </div>
                        {transcriptRow && (
                          <p className="segment-comment-quote">"{transcriptRow.text}"</p>
                        )}
                        <p className="segment-comment-text">{seg.comment}</p>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'guidelines' && (
          <div className="workspace-rag">
            <div className="rag-layout">
              <div className="card-panel rag-upload-card">
                <div className="upload-icon-wrapper">
                  <svg className="upload-doc-icon" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                    <polyline points="14 2 14 8 20 8"></polyline>
                    <line x1="16" x2="8" y1="13" y2="13"></line>
                    <line x1="16" x2="8" y1="17" y2="17"></line>
                    <polyline points="10 9 9 9 8 9"></polyline>
                  </svg>
                </div>
                <div className="upload-header-text">
                  <h4>Compliance Guideline Upload</h4>
                  <p>Upload a PDF or TXT compliance guidelines dictionary. SpeechInSight indexes segments using vectorized semantic search models into ChromaDB to benchmark call transcript quality.</p>
                </div>

                <div className="rag-controls">
                  <label className="file-select-label" style={{ alignSelf: 'center' }}>
                    Choose Guideline File
                    <input type="file" className="rag-file-input" accept=".txt,.pdf,.docx" onChange={handleRagFileChange} />
                  </label>
                  {ragFile && <span className="selected-doc-name" style={{ textAlign: 'center', fontSize: '0.8rem' }}>{ragFile.name}</span>}
                </div>

                <button
                  className="btn-primary"
                  disabled={!ragFile || ragLoading}
                  onClick={handleRagUpload}
                  style={{ alignSelf: 'center', maxWidth: '280px' }}
                >
                  {ragLoading ? "Indexing Guidelines..." : "Index Guidelines"}
                </button>

                {ragStatus && (
                  <div className="rag-status-box" style={{ marginTop: '1rem' }}>
                    <p>{ragStatus}</p>
                  </div>
                )}
              </div>

              <div className="guidelines-db-info">
                <h4>Guidelines Mapped in Database</h4>
                <p className="subtitle">Standard parameters verified during diarized turn auditing.</p>

                <div className="categories-grid">
                  <div className="cat-db-card">
                    <span className="cat-db-indicator orange" />
                    <div className="cat-db-content">
                      <h6>Warm Up Phase</h6>
                      <p>Greeting etiquette, tone consistency, and building conversational rapport.</p>
                    </div>
                  </div>
                  <div className="cat-db-card">
                    <span className="cat-db-indicator pink" />
                    <div className="cat-db-content">
                      <h6>Praise & Positivity</h6>
                      <p>Validation of team efforts, supportive feedback, and reinforcement checks.</p>
                    </div>
                  </div>
                  <div className="cat-db-card">
                    <span className="cat-db-indicator green" />
                    <div className="cat-db-content">
                      <h6>Suggestions Balance</h6>
                      <p>Ratio of positive vs. negative proposals, actionable critiques, and support.</p>
                    </div>
                  </div>
                  <div className="cat-db-card">
                    <span className="cat-db-indicator blue" />
                    <div className="cat-db-content">
                      <h6>Active Listening</h6>
                      <p>Feedback pauses, junior turn indicators, backchannel support checking.</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};
