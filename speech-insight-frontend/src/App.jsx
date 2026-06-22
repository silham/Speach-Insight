import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import './App.css'

// ── Neon color maps for minimal dark theme ──
const EMOTION_STYLES = {
  angry: { bg: 'rgba(248, 113, 113, 0.1)', text: '#F87171', border: 'rgba(248, 113, 113, 0.3)', label: 'Angry' },
  disgust: { bg: 'rgba(251, 146, 60, 0.1)', text: '#FB923C', border: 'rgba(251, 146, 60, 0.3)', label: 'Disgust' },
  fear: { bg: 'rgba(167, 139, 250, 0.1)', text: '#A78BFA', border: 'rgba(167, 139, 250, 0.3)', label: 'Fear' },
  happy: { bg: 'rgba(52, 211, 153, 0.1)', text: '#34D399', border: 'rgba(52, 211, 153, 0.3)', label: 'Happy' },
  sad: { bg: 'rgba(96, 165, 250, 0.1)', text: '#60A5FA', border: 'rgba(96, 165, 250, 0.3)', label: 'Sad' },
  surprise: { bg: 'rgba(244, 114, 182, 0.1)', text: '#F472B6', border: 'rgba(244, 114, 182, 0.3)', label: 'Surprise' },
  neutral: { bg: 'rgba(148, 163, 184, 0.1)', text: '#94A3B8', border: 'rgba(148, 163, 184, 0.3)', label: 'Neutral' },
}

const TEMPLATE_STYLES = {
  Direct: { border: 'rgba(248, 113, 113, 0.3)', text: '#F87171', bg: 'rgba(248, 113, 113, 0.1)' },
  Listen: { border: 'rgba(96, 165, 250, 0.3)', text: '#60A5FA', bg: 'rgba(96, 165, 250, 0.1)' },
  NSuggest: { border: 'rgba(251, 191, 36, 0.3)', text: '#FBBF24', bg: 'rgba(251, 191, 36, 0.1)' },
  PSuggest: { border: 'rgba(52, 211, 153, 0.3)', text: '#34D399', bg: 'rgba(52, 211, 153, 0.1)' },
  Praise: { border: 'rgba(244, 114, 182, 0.3)', text: '#F472B6', bg: 'rgba(244, 114, 182, 0.1)' },
  WarmUp: { border: 'rgba(251, 146, 60, 0.3)', text: '#FB923C', bg: 'rgba(251, 146, 60, 0.1)' },
}

const CATEGORY_NAMES = {
  template: 'Template Structure',
  warmup: 'Warm Up & Tone',
  praise: 'Praise & Positivity',
  suggest: 'Suggestions Balance',
  listen: 'Active Listening',
  direct: 'Clear Directives',
}

const CATEGORY_COLORS = {
  template: '#818CF8',
  warmup: '#FB923C',
  praise: '#F472B6',
  suggest: '#2DD4BF',
  listen: '#60A5FA',
  direct: '#F87171',
}

const SPEAKER_THEMES = [
  { bg: 'rgba(129, 140, 248, 0.1)', border: 'rgba(129, 140, 248, 0.3)', text: '#818CF8', primary: '#6366F1' },
  { bg: 'rgba(45, 212, 191, 0.1)', border: 'rgba(45, 212, 191, 0.3)', text: '#2DD4BF', primary: '#14B8A6' },
  { bg: 'rgba(251, 146, 60, 0.1)', border: 'rgba(251, 146, 60, 0.3)', text: '#FB923C', primary: '#F97316' },
  { bg: 'rgba(244, 114, 182, 0.1)', border: 'rgba(244, 114, 182, 0.3)', text: '#F472B6', primary: '#EC4899' },
  { bg: 'rgba(56, 189, 248, 0.1)', border: 'rgba(56, 189, 248, 0.3)', text: '#38BDF8', primary: '#0EA5E9' },
  { bg: 'rgba(52, 211, 153, 0.1)', border: 'rgba(52, 211, 153, 0.3)', text: '#34D399', primary: '#10B981' },
]

function SvgIcon({ name, className = "" }) {
  if (name === "upload") {
    return (
      <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
        <polyline points="17 8 12 3 7 8" />
        <line x1="12" y1="3" x2="12" y2="15" />
      </svg>
    )
  }
  if (name === "play") {
    return (
      <svg className={className} viewBox="0 0 24 24" fill="currentColor">
        <polygon points="5 3 19 12 5 21 5 3" />
      </svg>
    )
  }
  if (name === "pause") {
    return (
      <svg className={className} viewBox="0 0 24 24" fill="currentColor">
        <rect x="6" y="4" width="4" height="16" />
        <rect x="14" y="4" width="4" height="16" />
      </svg>
    )
  }
  if (name === "dashboard") {
    return (
      <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="3" width="7" height="9" />
        <rect x="14" y="3" width="7" height="5" />
        <rect x="14" y="12" width="7" height="9" />
        <rect x="3" y="16" width="7" height="5" />
      </svg>
    )
  }
  if (name === "document") {
    return (
      <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
        <polyline points="14 2 14 8 20 8" />
        <line x1="16" y1="13" x2="8" y2="13" />
        <line x1="16" y1="17" x2="8" y2="17" />
      </svg>
    )
  }
  if (name === "search") {
    return (
      <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="11" cy="11" r="8" />
        <line x1="21" y1="21" x2="16.65" y2="16.65" />
      </svg>
    )
  }
  if (name === "check") {
    return (
      <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="20 6 9 17 4 12" />
      </svg>
    )
  }
  if (name === "chevron") {
    return (
      <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <polyline points="6 9 12 15 18 9" />
      </svg>
    )
  }
  if (name === "filter") {
    return (
      <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" />
      </svg>
    )
  }
  if (name === "chart") {
    return (
      <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <line x1="18" y1="20" x2="18" y2="10" />
        <line x1="12" y1="20" x2="12" y2="4" />
        <line x1="6" y1="20" x2="6" y2="14" />
      </svg>
    )
  }
  return null
}

function Soundwave() {
  return (
    <div className="soundwave-anim">
      <div className="soundwave-bar" />
      <div className="soundwave-bar" />
      <div className="soundwave-bar" />
      <div className="soundwave-bar" />
    </div>
  )
}

function EmotionBadge({ emotion, confidence }) {
  const style = EMOTION_STYLES[emotion.toLowerCase()] || EMOTION_STYLES.neutral
  return (
    <span
      className="badge-emotion animate-badge"
      style={{ backgroundColor: style.bg, color: style.text, borderColor: style.border }}
    >
      <span className="dot animate-pulse" style={{ backgroundColor: style.text }} />
      {style.label} <span className="conf">{Math.round(confidence * 100)}%</span>
    </span>
  )
}

function TemplateBadge({ label }) {
  if (!label) return null
  const style = TEMPLATE_STYLES[label] || { border: '#cbd5e1', text: '#475569', bg: '#f1f5f9' }
  return (
    <span
      className="badge-template"
      style={{ borderColor: style.border, color: style.text, backgroundColor: style.bg }}
    >
      {label}
    </span>
  )
}

function App() {
  const [activeTab, setActiveTab] = useState('pipeline')

  // Pipeline execution state
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [status, setStatus] = useState("")
  const [results, setResults] = useState([])
  const [report, setReport] = useState(null)
  const [metadata, setMetadata] = useState(null)

  // Filtering states for transcript
  const [searchTerm, setSearchTerm] = useState("")
  const [speakerFilter, setSpeakerFilter] = useState("all")

  // Interactive selection states
  const [selectedSegmentId, setSelectedSegmentId] = useState(null)
  const [playingSegmentId, setPlayingSegmentId] = useState(null)
  const [expandedCategory, setExpandedCategory] = useState(null)
  const [rightPanelTab, setRightPanelTab] = useState('report') // 'report' or 'inspector'
  const [isTranscriptOpen, setIsTranscriptOpen] = useState(false)

  // RAG upload state
  const [ragFile, setRagFile] = useState(null)
  const [ragLoading, setRagLoading] = useState(false)
  const [ragStatus, setRagStatus] = useState("")

  const audioInstanceRef = useRef(null)

  useEffect(() => {
    return () => {
      if (audioInstanceRef.current) {
        audioInstanceRef.current.pause()
      }
    }
  }, [])

  const handleFileChange = (e) => {
    if (e.target.files[0]) {
      setFile(e.target.files[0])
      setStatus("")
    }
  }

  const handleUpload = async () => {
    if (!file) return

    const formData = new FormData()
    formData.append("file", file)

    setLoading(true)
    setStatus("Initiating speech transcription, emotion mapping, and compliance scoring...")
    setResults([])
    setReport(null)
    setMetadata(null)
    setSelectedSegmentId(null)
    setPlayingSegmentId(null)

    try {
      const response = await axios.post("http://127.0.0.1:8000/analyze", formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })

      const data = response.data
      setResults(data.data)
      setMetadata({
        job_id: data.job_id,
        lead_speaker: data.lead_speaker,
        total_speakers: data.total_speakers,
        total_segments: data.total_segments,
        total_duration: data.total_duration,
        filename: file.name
      })
      
      if (data.data.length > 0) {
        setSelectedSegmentId(0) // Default segment selection
      }

      setStatus("Analysis completed successfully.")

      // Fetch performance report
      try {
        const reportRes = await axios.get(`http://127.0.0.1:8000/report/${data.job_id}`)
        setReport(reportRes.data)
      } catch (reportErr) {
        console.warn("Performance report not available:", reportErr)
      }
    } catch (error) {
      console.error(error)
      setStatus("Error: " + (error.response?.data?.detail || "Could not connect to service"))
    } finally {
      setLoading(false)
    }
  }

  // RAG functions
  const handleRagFileChange = (e) => {
    if (e.target.files[0]) {
      setRagFile(e.target.files[0])
      setRagStatus("")
    }
  }

  const handleRagUpload = async () => {
    if (!ragFile) return

    const formData = new FormData()
    formData.append("file", ragFile)

    setRagLoading(true)
    setRagStatus("Indexing reference guidelines document...")

    try {
      const response = await axios.post("http://127.0.0.1:8000/rag/upload", formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      setRagStatus(`Success! Indexed ${response.data.chunks_added} segments into ChromaDB guidelines base.`)
    } catch (error) {
      console.error(error)
      setRagStatus("Error: " + (error.response?.data?.detail || "Upload failed"))
    } finally {
      setRagLoading(false)
      setRagFile(null)
    }
  }

  // Play a segment clip
  const togglePlaySegment = (url, segmentId) => {
    if (playingSegmentId === segmentId) {
      if (audioInstanceRef.current) {
        audioInstanceRef.current.pause()
      }
      setPlayingSegmentId(null)
      return
    }

    if (audioInstanceRef.current) {
      audioInstanceRef.current.pause()
    }

    const fullUrl = url.startsWith('http') ? url : `http://127.0.0.1:8000${url}`
    const audio = new Audio(fullUrl)
    audioInstanceRef.current = audio
    setPlayingSegmentId(segmentId)

    audio.play().catch(e => {
      console.error("Audio playback error:", e)
      setPlayingSegmentId(null)
    })

    audio.onended = () => {
      setPlayingSegmentId(null)
    }
  }

  const selectAndScrollToSegment = (segmentId, audioUrl) => {
    setSelectedSegmentId(segmentId)
    setRightPanelTab('inspector')
    const element = document.getElementById(`bubble-${segmentId}`)
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
    }
    if (audioUrl) {
      togglePlaySegment(audioUrl, segmentId)
    }
  }

  const toggleCategory = (catName) => {
    setExpandedCategory(expandedCategory === catName ? null : catName)
  }

  // Filter transcript segments
  const filteredResults = results.filter(row => {
    const matchesSearch = row.text.toLowerCase().includes(searchTerm.toLowerCase())
    const matchesSpeaker = speakerFilter === "all" || row.speaker === speakerFilter
    return matchesSearch && matchesSpeaker
  })

  // Dynamic speaker metrics calculations
  const totalDuration = metadata?.total_duration || 1
  const speakersList = Array.from(new Set(results.map(r => r.speaker)))

  const getSpeakerTheme = (speaker) => {
    const idx = speakersList.indexOf(speaker) % SPEAKER_THEMES.length
    return SPEAKER_THEMES[idx >= 0 ? idx : 0]
  }

  const speakerStats = speakersList.map(speaker => {
    const segments = results.filter(r => r.speaker === speaker)
    const talkTime = segments.reduce((acc, r) => acc + (r.end_time - r.start_time), 0)
    const percentage = (talkTime / totalDuration) * 100
    const theme = getSpeakerTheme(speaker)
    return {
      name: speaker,
      talkTime: talkTime.toFixed(1),
      percentage: percentage.toFixed(1),
      theme
    }
  })

  // Conversation overall emotion vibe metrics
  const emotionSummary = {}
  results.forEach(r => {
    emotionSummary[r.emotion] = (emotionSummary[r.emotion] || 0) + 1
  })
  const emotionVibeStats = Object.entries(emotionSummary)
    .map(([emo, count]) => ({
      name: emo,
      percentage: ((count / results.length) * 100).toFixed(0)
    }))
    .sort((a, b) => b.percentage - a.percentage)

  const activeSegment = results.find(r => r.segment_id === selectedSegmentId)

  return (
    <div className="saas-layout">
      {/* Sidebar Navigation */}
      <aside className="sidebar-nav">
        <div className="sidebar-brand">
          <div className="brand-icon-wrapper">
            <SvgIcon name="waveform" className="brand-logo-icon" />
          </div>
          <div className="brand-details">
            <h3>SpeechInSight</h3>
            <span>Coaching Appraisals</span>
          </div>
        </div>

        <nav className="nav-menu">
          <button
            className={`nav-item-btn ${activeTab === 'pipeline' ? 'active' : ''}`}
            onClick={() => setActiveTab('pipeline')}
          >
            <SvgIcon name="dashboard" className="nav-icon" />
            <span>Analysis Dashboard</span>
          </button>
          <button
            className={`nav-item-btn ${activeTab === 'rag' ? 'active' : ''}`}
            onClick={() => setActiveTab('rag')}
          >
            <SvgIcon name="document" className="nav-icon" />
            <span>Guidelines DB</span>
          </button>
        </nav>

        {metadata && (
          <div className="sidebar-job-badge">
            <span className="badge-title">Active Session</span>
            <p className="badge-filename">{metadata.filename}</p>
            <div className="badge-row">
              <span>{Math.round(metadata.total_duration)}s</span>
              <span className="dot-sep" />
              <span>{metadata.total_segments} turns</span>
            </div>
            <button
              onClick={() => {
                setMetadata(null)
                setResults([])
                setReport(null)
                setSelectedSegmentId(null)
                setPlayingSegmentId(null)
              }}
              className="btn-sidebar-reset"
            >
              Analyze New
            </button>
          </div>
        )}
      </aside>

      {/* Main Content Workspace */}
      <div className="saas-content">
        {activeTab === 'pipeline' && (
          <div className="workspace-pipeline">
            {/* Upload Dashboard card */}
            {!metadata && (
              <div className="upload-container fade-in">
                <div className="upload-card">
                  <div className="upload-icon-wrapper">
                    <SvgIcon name="upload" className="upload-hero-icon" />
                  </div>
                  <h3>Upload Conversation</h3>
                  <p>Upload a meeting audio or video file to check performance guidelines, diarize speakers, and map emotions.</p>
                  
                  <div className="upload-controls">
                    <label className="file-select-label">
                      Browse Files
                      <input
                        type="file"
                        onChange={handleFileChange}
                        accept="audio/*,video/*"
                        className="file-raw-input"
                      />
                    </label>
                    {file ? (
                      <span className="selected-filename">{file.name}</span>
                    ) : (
                      <span className="upload-guideline-text">Supports WAV, MP3, FLAC, MP4, MOV</span>
                    )}
                  </div>

                  {file && (
                    <button
                      onClick={handleUpload}
                      disabled={loading}
                      className="btn-primary"
                    >
                      {loading ? "Processing speech files..." : "Start Analysis Pipeline"}
                    </button>
                  )}

                  {status && <p className="status-message">{status}</p>}
                </div>
              </div>
            )}

            {loading && (
              <div className="loading-card fade-in">
                <div className="spinner" />
                <h4>Processing speech files...</h4>
                <p>Segmenting speaker turns, transcribing utterance phonetics, and evaluating guidelines compliance. Please hold on.</p>
              </div>
            )}

            {/* Results UI */}
            {metadata && !loading && (
              <div className="pipeline-dashboard-grid fade-in">
                
                {/* Visual Speech Timeline row */}
                <div className="dashboard-row double-split">
                  
                  {/* Speech Timeline Card */}
                  <section className="timeline-container card-panel flex-grow">
                    <div className="section-header">
                      <div className="header-text-group">
                        <h4>Conversation Turn Timeline</h4>
                        <span className="section-subtitle">Visual chronology of speaker turns. Tap blocks to inspect details and play.</span>
                      </div>
                      {playingSegmentId !== null && (
                        <div className="timeline-play-status">
                          <Soundwave />
                          <span>Playing turn #{playingSegmentId}</span>
                        </div>
                      )}
                    </div>
                    
                    <div className="speech-timeline">
                      {results.map((row) => {
                        const duration = row.end_time - row.start_time
                        const percentWidth = (duration / metadata.total_duration) * 100
                        const isSelected = row.segment_id === selectedSegmentId
                        const isPlaying = row.segment_id === playingSegmentId
                        const theme = getSpeakerTheme(row.speaker)

                        return (
                          <div
                            key={row.segment_id}
                            className={`timeline-block ${isSelected ? 'selected' : ''} ${isPlaying ? 'playing' : ''}`}
                            style={{
                              width: `${Math.max(percentWidth, 1.2)}%`,
                              backgroundColor: theme.primary,
                              borderColor: isSelected || isPlaying ? '#000000' : theme.border,
                            }}
                            onClick={() => selectAndScrollToSegment(row.segment_id, row.audio_url)}
                            title={`${row.speaker} (${row.start_time.toFixed(1)}s - ${row.end_time.toFixed(1)}s): "${row.text.substring(0, 40)}..."`}
                          >
                            <span className="timeline-block-label" style={{ color: '#ffffff' }}>
                              {row.speaker.split('_')[1] || row.speaker}
                            </span>
                          </div>
                        )
                      })}
                    </div>
                  </section>

                  {/* Speaker breakdown Card */}
                  <section className="speaker-breakdown card-panel flex-shrink">
                    <div className="section-header">
                      <h4>Talk-Time Allocation</h4>
                    </div>
                    <div className="speaker-breakdown-list">
                      {speakerStats.map(spk => (
                        <div key={spk.name} className="spk-breakdown-item">
                          <div className="spk-breakdown-labels">
                            <span className="spk-breakdown-name" style={{ color: spk.theme.text }}>{spk.name}</span>
                            <span className="spk-breakdown-time">{spk.talkTime}s ({spk.percentage}%)</span>
                          </div>
                          <div className="spk-breakdown-track">
                            <div
                              className="spk-breakdown-fill"
                              style={{
                                width: `${spk.percentage}%`,
                                backgroundColor: spk.theme.primary
                              }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </section>
                </div>

                {/* Second row: Analysis Split */}
                <div className="analysis-row">
                  {/* Left Column: Performance Report */}
                  <section className="analysis-column card-panel">
                    <div className="section-header">
                      <h4>Evaluation Report</h4>
                    </div>
                    <div className="tab-details-content fade-in">
                      {report ? (
                        <div className="report-tab-layout">
                          <div className="report-radial-header">
                            <div className="report-intro-text">
                              <h4>Guidelines Compliance</h4>
                              <p className="report-subtext">Meeting score evaluated against index rules</p>
                            </div>
                            
                            <div className="radial-score-circle">
                              <svg className="radial-svg" viewBox="0 0 100 100">
                                <defs>
                                  <linearGradient id="radialGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                                    <stop offset="0%" stopColor="#4f46e5" />
                                    <stop offset="100%" stopColor="#10b981" />
                                  </linearGradient>
                                </defs>
                                <circle cx="50" cy="50" r="42" className="radial-bg" />
                                <circle
                                  cx="50" cy="50" r="42"
                                  className="radial-fill"
                                  style={{
                                    stroke: 'url(#radialGradient)',
                                    strokeDasharray: `${(report.total_score / 100) * 264} 264`
                                  }}
                                />
                              </svg>
                              <div className="radial-inner-label">
                                <span className="score-num">{Math.round(report.total_score)}</span>
                                <span className="score-total">/100</span>
                              </div>
                            </div>
                          </div>

                          {/* Overall Vibe Check summary */}
                          {emotionVibeStats.length > 0 && (
                            <div className="meeting-vibe-summary">
                              <span className="vibe-label">Conversation Vibe Check</span>
                              <div className="vibe-pills-row">
                                {emotionVibeStats.slice(0, 3).map(vibe => {
                                  const style = EMOTION_STYLES[vibe.name.toLowerCase()] || EMOTION_STYLES.neutral
                                  return (
                                    <span
                                      key={vibe.name}
                                      className="vibe-pill font-bold"
                                      style={{
                                        color: style.text,
                                        backgroundColor: style.bg,
                                        border: `1px solid ${style.border}`
                                      }}
                                    >
                                      {vibe.name} ({vibe.percentage}%)
                                    </span>
                                  )
                                })}
                              </div>
                            </div>
                          )}

                          {/* Evaluation category breakdown list */}
                          <div className="report-categories-list">
                            {report.categories && report.categories.map((cat, idx) => {
                              const isExpanded = expandedCategory === cat.name
                              const progressPercent = (cat.score / cat.max_score) * 100
                              const catColor = CATEGORY_COLORS[cat.name] || '#6366f1'
                              return (
                                <div key={idx} className={`eval-row ${isExpanded ? 'open' : ''}`}>
                                  <div className="eval-summary" onClick={() => toggleCategory(cat.name)}>
                                    <div className="eval-info">
                                      <span className="eval-name">{CATEGORY_NAMES[cat.name] || cat.name}</span>
                                      <span className="eval-score" style={{ color: catColor }}>{cat.score}/{cat.max_score}</span>
                                    </div>
                                    <div className="eval-progress-container">
                                      <div className="eval-progress-bar" style={{ width: `${progressPercent}%`, backgroundColor: catColor }} />
                                    </div>
                                    <div className="eval-dropdown-toggle">
                                      <SvgIcon name="chevron" className="icon-chevron" />
                                    </div>
                                  </div>
                                  {isExpanded && (
                                    <div className="eval-details-pane">
                                      <p className="eval-description">{cat.description}</p>
                                    </div>
                                  )}
                                </div>
                              )
                            })}
                          </div>

                          {/* Strengths & Improvements */}
                          <div className="report-summaries-grid">
                            {report.strengths && report.strengths.length > 0 && (
                              <div className="summary-list-card strength">
                                <h5>Key Strengths</h5>
                                <ul>
                                  {report.strengths.slice(0, 3).map((str, i) => <li key={i}>{str}</li>)}
                                </ul>
                              </div>
                            )}
                            {report.improvements && report.improvements.length > 0 && (
                              <div className="summary-list-card improvement">
                                <h5>Recommendations</h5>
                                <ul>
                                  {report.improvements.slice(0, 3).map((imp, i) => <li key={i}>{imp}</li>)}
                                </ul>
                              </div>
                            )}
                          </div>
                        </div>
                      ) : (
                        <div className="report-empty-state">
                          <p>Compliance report data is not available for this session.</p>
                        </div>
                      )}
                    </div>
                  </section>

                  {/* Right Column: Segment Inspector */}
                  <section className="analysis-column card-panel">
                    <div className="section-header">
                      <h4>Utterance Inspector</h4>
                    </div>
                    <div className="tab-details-content fade-in" style={{ marginTop: '1rem' }}>
                      {activeSegment ? (
                        <div className="inspector-content">
                          <div className="inspector-meta-row">
                            <span className="inspector-speaker">{activeSegment.speaker}</span>
                            <span className="inspector-time">
                              {activeSegment.start_time.toFixed(1)}s - {activeSegment.end_time.toFixed(1)}s
                            </span>
                          </div>
                          
                          <blockquote className="inspector-quote">
                            "{activeSegment.text}"
                          </blockquote>

                          {/* Paralinguistics */}
                          <div className="inspector-metrics-section">
                            <h5>Vocal Acoustics</h5>
                            <div className="metrics-grid">
                              <div className="metric-metric">
                                <div className="metric-meta">
                                  <span>Pitch</span>
                                  <span className="metric-value">{Math.round(activeSegment.paralinguistic?.pitch || 0)} Hz</span>
                                </div>
                                <div className="metric-bar-track">
                                  <div className="metric-bar-fill" style={{ width: `${Math.min((activeSegment.paralinguistic?.pitch || 0) / 4, 100)}%` }} />
                                </div>
                              </div>

                              <div className="metric-metric">
                                <div className="metric-meta">
                                  <span>Speaking Rate</span>
                                  <span className="metric-value">{(activeSegment.paralinguistic?.speaking_rate || 0).toFixed(1)} syl/s</span>
                                </div>
                                <div className="metric-bar-track">
                                  <div className="metric-bar-fill" style={{ width: `${Math.min((activeSegment.paralinguistic?.speaking_rate || 0) * 15, 100)}%` }} />
                                </div>
                              </div>

                              <div className="metric-metric">
                                <div className="metric-meta">
                                  <span>Energy</span>
                                  <span className="metric-value">{(activeSegment.paralinguistic?.energy || 0).toExponential(2)}</span>
                                </div>
                                <div className="metric-bar-track">
                                  <div className="metric-bar-fill" style={{ width: `${Math.min((activeSegment.paralinguistic?.energy || 0) * 8000, 100)}%` }} />
                                </div>
                              </div>
                            </div>
                          </div>

                          {/* Sentiment / VADER */}
                          {activeSegment.vader && (
                            <div className="inspector-metrics-section">
                              <h5>Linguistic Sentiment</h5>
                              <div className="vader-compound-bar">
                                <div className="vader-track">
                                  <div
                                    className="vader-marker"
                                    style={{ left: `${((activeSegment.vader.compound + 1) / 2) * 100}%` }}
                                  />
                                </div>
                                <div className="vader-labels">
                                  <span>Negative</span>
                                  <span className="compound-val">Compound: {activeSegment.vader.compound?.toFixed(2)}</span>
                                  <span>Positive</span>
                                </div>
                              </div>
                            </div>
                          )}

                          {/* All emotions confidence */}
                          {activeSegment.all_emotions && Object.keys(activeSegment.all_emotions).length > 0 && (
                            <div className="inspector-metrics-section">
                              <h5>Emotion Distribution</h5>
                              <div className="emotions-distribution-list">
                                {Object.entries(activeSegment.all_emotions)
                                  .sort((a, b) => b[1] - a[1])
                                  .slice(0, 4)
                                  .map(([emo, conf]) => {
                                    const percent = Math.round(conf * 100)
                                    const style = EMOTION_STYLES[emo.toLowerCase()] || EMOTION_STYLES.neutral
                                    return (
                                      <div key={emo} className="emo-dist-row">
                                        <div className="emo-dist-labels">
                                          <span className="emo-dist-name">{emo}</span>
                                          <span className="emo-dist-pct">{percent}%</span>
                                        </div>
                                        <div className="emo-dist-track">
                                          <div
                                            className="emo-dist-fill"
                                            style={{
                                              width: `${percent}%`,
                                              backgroundColor: style.text
                                            }}
                                          />
                                        </div>
                                      </div>
                                    )
                                  })}
                              </div>
                            </div>
                          )}
                        </div>
                      ) : (
                        <div className="inspector-placeholder">
                          <SvgIcon name="waveform" className="placeholder-icon" />
                          <p>Click on any speaker turn in the transcript or timeline to map acoustic parameters and sentiment profiles here.</p>
                        </div>
                      )}
                    </div>
                  </section>
                </div>

                {/* Third row: Transcript Accordion */}
                <div className="transcript-accordion-container card-panel">
                  <button 
                    className="transcript-accordion-toggle" 
                    onClick={() => setIsTranscriptOpen(!isTranscriptOpen)}
                  >
                    <h4>Conversation Transcript</h4>
                    <SvgIcon name="chevron" className={`accordion-icon ${isTranscriptOpen ? 'open' : ''}`} />
                  </button>
                  
                  <div className={`transcript-accordion-content ${isTranscriptOpen ? 'open' : ''}`}>
                    <div className="chat-stream-card" style={{ height: '500px', marginTop: '1rem', borderTop: '1px solid var(--border-color)', paddingTop: '1rem' }}>
                      
                      {/* Transcript search and filter bar */}
                      <div className="transcript-control-header" style={{ borderBottom: 'none', paddingBottom: '0.5rem' }}>
                        <div className="transcript-filters">
                          {/* Search */}
                          <div className="search-input-wrapper">
                            <SvgIcon name="search" className="search-icon" />
                            <input
                              type="text"
                              placeholder="Search transcript..."
                              value={searchTerm}
                              onChange={(e) => setSearchTerm(e.target.value)}
                              className="filter-text-input"
                            />
                          </div>

                          {/* Speaker filter */}
                          <div className="select-filter-wrapper">
                            <SvgIcon name="filter" className="filter-icon" />
                            <select
                              value={speakerFilter}
                              onChange={(e) => setSpeakerFilter(e.target.value)}
                              className="filter-select-input"
                            >
                              <option value="all">All Speakers</option>
                              {speakersList.map(spk => (
                                <option key={spk} value={spk}>{spk}</option>
                              ))}
                            </select>
                          </div>
                        </div>
                      </div>

                      <div className="bubble-list">
                        {filteredResults.length > 0 ? (
                          filteredResults.map((row) => {
                            const isSelected = row.segment_id === selectedSegmentId
                            const isPlaying = playingSegmentId === row.segment_id
                            const theme = getSpeakerTheme(row.speaker)

                            return (
                              <div
                                key={row.segment_id}
                                id={`bubble-${row.segment_id}`}
                                className={`bubble-wrapper ${isSelected ? 'active-bubble' : ''}`}
                                style={{
                                  borderLeft: isSelected
                                    ? `4px solid ${theme.primary}`
                                    : `1px solid var(--border-color)`
                                }}
                                onClick={() => {
                                  setSelectedSegmentId(row.segment_id)
                                  if (!isTranscriptOpen) setIsTranscriptOpen(true)
                                }}
                              >
                                <div className="bubble-header">
                                  <span className="bubble-speaker font-semibold" style={{ backgroundColor: theme.bg, color: theme.text, borderColor: theme.border }}>
                                    {row.speaker}
                                  </span>
                                  <span className="bubble-time">
                                    {row.start_time.toFixed(1)}s – {row.end_time.toFixed(1)}s
                                  </span>
                                </div>
                                
                                <div className="bubble-body">
                                  <p className="bubble-text">{row.text}</p>
                                  
                                  <div className="bubble-meta-row">
                                    <div className="bubble-badges">
                                      <EmotionBadge emotion={row.emotion} confidence={row.confidence} />
                                      <TemplateBadge label={row.template_label} />
                                      {row.sarcasm && <span className="badge-sarcasm">Sarcasm ({Math.round(row.sarcasm_score * 100)}%)</span>}
                                      {row.ambiguity_score > 0.5 && <span className="badge-ambiguity">Ambiguous</span>}
                                    </div>
                                    
                                    <button
                                      className={`bubble-play-btn ${isPlaying ? 'playing' : ''}`}
                                      style={{
                                        backgroundColor: isPlaying ? theme.primary : 'var(--bg-card)',
                                        color: isPlaying ? 'white' : 'var(--text-secondary)',
                                        borderColor: isPlaying ? theme.primary : 'var(--border-color)'
                                      }}
                                      onClick={(e) => {
                                        e.stopPropagation()
                                        togglePlaySegment(row.audio_url, row.segment_id)
                                      }}
                                    >
                                      <SvgIcon name={isPlaying ? "pause" : "play"} className="play-icon" />
                                      {isPlaying ? "Pause" : "Play"}
                                    </button>
                                  </div>
                                </div>
                              </div>
                            )
                          })
                        ) : (
                          <div className="no-filter-results">
                            <p>No turns match your filters. Try clearing your search.</p>
                            <button
                              onClick={() => { setSearchTerm(""); setSpeakerFilter("all"); }}
                              className="btn-secondary"
                            >
                              Reset filters
                            </button>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'rag' && (
          <div className="workspace-rag fade-in">
            <div className="rag-layout">
              <div className="rag-upload-card card-panel">
                <div className="upload-header">
                  <SvgIcon name="document" className="upload-doc-icon" />
                  <div className="upload-header-text">
                    <h4>Reference Guidelines Database</h4>
                    <p>Index coaching files (PDF, DOCX, or TXT) to compile evaluation compliance matrices.</p>
                  </div>
                </div>

                <div className="rag-controls">
                  <div className="rag-file-dropzone">
                    <input
                      type="file"
                      onChange={handleRagFileChange}
                      accept=".pdf,.docx,.txt"
                      className="rag-file-input"
                      id="rag-file"
                    />
                    <label htmlFor="rag-file" className="rag-file-label">
                      {ragFile ? (
                        <span className="selected-doc-name font-semibold">File Selected: {ragFile.name}</span>
                      ) : (
                        <span>Drag reference guidelines here or click to browse files</span>
                      )}
                    </label>
                  </div>

                  <button
                    onClick={handleRagUpload}
                    disabled={ragLoading || !ragFile}
                    className="btn-primary"
                  >
                    {ragLoading ? "Processing and Indexing Chunks..." : "Index Guidelines Document"}
                  </button>
                </div>

                {ragStatus && (
                  <div className="rag-status-box animate-fade-in">
                    <SvgIcon name="check" className="icon-success" />
                    <p>{ragStatus}</p>
                  </div>
                )}
              </div>

              {/* RAG DB Guidelines visual grid */}
              <div className="guidelines-db-info card-panel">
                <h4>Operational Evaluation Matrix</h4>
                <p className="subtitle">Guidelines are classified by NLP models into five categories to evaluate recording sessions:</p>
                
                <div className="categories-grid">
                  <div className="cat-db-card card-hover-effect">
                    <span className="cat-db-indicator orange animate-pulse" />
                    <div className="cat-db-content">
                      <h6>Warm Up (WarmUp)</h6>
                      <p>Guidelines for greetings, ice-breakers, and conversation starters.</p>
                    </div>
                  </div>

                  <div className="cat-db-card card-hover-effect">
                    <span className="cat-db-indicator pink animate-pulse" />
                    <div className="cat-db-content">
                      <h6>Praise & Reinforcements (Praise)</h6>
                      <p>Rules regarding compliments and appreciation tokens.</p>
                    </div>
                  </div>

                  <div className="cat-db-card card-hover-effect">
                    <span className="cat-db-indicator green animate-pulse" />
                    <div className="cat-db-content">
                      <h6>Constructive Feedback (Suggest)</h6>
                      <p>Rules for recommendations, negative and positive balance suggestions.</p>
                    </div>
                  </div>

                  <div className="cat-db-card card-hover-effect">
                    <span className="cat-db-indicator blue" />
                    <div className="cat-db-content">
                      <h6>Listening Space (Listen)</h6>
                      <p>Acoustic parameters for conversational gaps and active feedback.</p>
                    </div>
                  </div>

                  <div className="cat-db-card card-hover-effect">
                    <span className="cat-db-indicator red" />
                    <div className="cat-db-content">
                      <h6>Direct Instructions (Direct)</h6>
                      <p>Command structures and actionable compliance parameters.</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App