import { useState } from 'react'
import axios from 'axios'
import './App.css'

/* ── Emotion colour map (Ekman 7) ── */
const EMOTION_COLORS = {
  angry: { bg: '#fee2e2', text: '#991b1b', border: '#fca5a5' },
  disgust: { bg: '#fef3c7', text: '#92400e', border: '#fcd34d' },
  fear: { bg: '#ede9fe', text: '#5b21b6', border: '#c4b5fd' },
  happy: { bg: '#dcfce7', text: '#166534', border: '#86efac' },
  sad: { bg: '#dbeafe', text: '#1e3a8a', border: '#93c5fd' },
  surprise: { bg: '#fce7f3', text: '#9d174d', border: '#f9a8d4' },
  neutral: { bg: '#f3f4f6', text: '#374151', border: '#d1d5db' },
}

/* ── Template category colour map ── */
const TEMPLATE_COLORS = {
  Direct: { bg: '#fef2f2', text: '#991b1b', border: '#fecaca' },
  Listen: { bg: '#eff6ff', text: '#1e40af', border: '#bfdbfe' },
  NSuggest: { bg: '#fefce8', text: '#854d0e', border: '#fde68a' },
  PSuggest: { bg: '#f0fdf4', text: '#166534', border: '#bbf7d0' },
  Praise: { bg: '#fdf4ff', text: '#86198f', border: '#f0abfc' },
  WarmUp: { bg: '#fff7ed', text: '#9a3412', border: '#fed7aa' },
}

/* ── Category gradient map for report bars ── */
const CATEGORY_GRADIENTS = {
  template: 'linear-gradient(135deg, #6366f1, #818cf8)',
  warmup: 'linear-gradient(135deg, #f97316, #fb923c)',
  praise: 'linear-gradient(135deg, #a855f7, #c084fc)',
  suggest: 'linear-gradient(135deg, #10b981, #34d399)',
  listen: 'linear-gradient(135deg, #3b82f6, #60a5fa)',
  direct: 'linear-gradient(135deg, #ef4444, #f87171)',
}

function EmotionBadge({ emotion, confidence }) {
  const colors = EMOTION_COLORS[emotion] || EMOTION_COLORS.neutral
  return (
    <span
      className="emotion-badge"
      style={{ background: colors.bg, color: colors.text, border: `1px solid ${colors.border}` }}
    >
      {emotion} <span className="confidence">{Math.round(confidence * 100)}%</span>
    </span>
  )
}

function TemplateBadge({ label, confidence }) {
  if (!label) return <span className="template-badge-empty">—</span>
  const colors = TEMPLATE_COLORS[label] || { bg: '#f3f4f6', text: '#374151', border: '#d1d5db' }
  return (
    <span
      className="template-badge"
      style={{ background: colors.bg, color: colors.text, border: `1px solid ${colors.border}` }}
    >
      {label} <span className="confidence">{Math.round(confidence * 100)}%</span>
    </span>
  )
}

function SarcasmBadge() {
  return <span className="sarcasm-badge">sarcasm</span>
}

function AmbiguityIndicator({ score }) {
  if (score < 0.5) return null
  return (
    <span className="ambiguity-indicator" title={`Ambiguity: ${Math.round(score * 100)}%`}>
      ambiguous
    </span>
  )
}

/* ── Score Bar Component ── */
function ScoreBar({ name, score, maxScore, description, gradient }) {
  const pct = maxScore > 0 ? (score / maxScore) * 100 : 0
  const displayName = name.charAt(0).toUpperCase() + name.slice(1)
  return (
    <div className="score-bar-card">
      <div className="score-bar-header">
        <span className="score-bar-name">{displayName}</span>
        <span className="score-bar-value">{score} / {maxScore}</span>
      </div>
      <div className="score-bar-track">
        <div
          className="score-bar-fill"
          style={{ width: `${Math.min(pct, 100)}%`, background: gradient }}
        />
      </div>
      <p className="score-bar-desc">{description}</p>
    </div>
  )
}

/* ── Report Panel Component ── */
function ReportPanel({ report }) {
  if (!report) return null

  return (
    <div className="report-container fade-in">
      <div className="report-header glass-panel">
        <h2>Performance Report</h2>
        <div className="total-score-circle">
          <svg viewBox="0 0 120 120" className="score-ring">
            <circle cx="60" cy="60" r="52" className="score-ring-bg" />
            <circle
              cx="60" cy="60" r="52"
              className="score-ring-fill"
              style={{
                strokeDasharray: `${(report.total_score / 100) * 327} 327`
              }}
            />
          </svg>
          <div className="score-ring-text">
            <span className="score-ring-number">{Math.round(report.total_score)}</span>
            <span className="score-ring-label">/100</span>
          </div>
        </div>
      </div>

      <div className="report-categories">
        {report.categories && report.categories.map((cat, idx) => (
          <ScoreBar
            key={idx}
            name={cat.name}
            score={cat.score}
            maxScore={cat.max_score}
            description={cat.description}
            gradient={CATEGORY_GRADIENTS[cat.name] || CATEGORY_GRADIENTS.template}
          />
        ))}
      </div>

      <div className="report-insights">
        {report.strengths && report.strengths.length > 0 && (
          <div className="insight-card strengths-card glass-panel">
            <h3>Strengths</h3>
            <ul>
              {report.strengths.map((s, i) => <li key={i}>{s}</li>)}
            </ul>
          </div>
        )}
        {report.improvements && report.improvements.length > 0 && (
          <div className="insight-card improvements-card glass-panel">
            <h3>Areas for Improvement</h3>
            <ul>
              {report.improvements.map((s, i) => <li key={i}>{s}</li>)}
            </ul>
          </div>
        )}
      </div>
    </div>
  )
}

function App() {
  // Tabs: 'pipeline' or 'rag'
  const [activeTab, setActiveTab] = useState('pipeline')

  // Pipeline State
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState([])
  const [status, setStatus] = useState("")
  const [report, setReport] = useState(null)

  // RAG State
  const [ragFile, setRagFile] = useState(null)
  const [ragLoading, setRagLoading] = useState(false)
  const [ragStatus, setRagStatus] = useState("")

  const handleFileChange = (e) => {
    setFile(e.target.files[0])
  }

  const handleUpload = async () => {
    if (!file) return alert("Please select a file!")

    const formData = new FormData()
    formData.append("file", file)

    setLoading(true)
    setStatus("Uploading and Analyzing... (This may take a minute)")
    setResults([])
    setReport(null)

    try {
      const response = await axios.post("http://127.0.0.1:8000/analyze", formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })

      setResults(response.data.data)
      setStatus("Analysis Complete!")

      // Fetch the report
      try {
        const reportRes = await axios.get(`http://127.0.0.1:8000/report/${response.data.job_id}`)
        setReport(reportRes.data)
      } catch (reportErr) {
        console.warn("Report not available:", reportErr)
      }
    } catch (error) {
      console.error(error)
      setStatus("Error: " + (error.response?.data?.detail || "Connection failed"))
    } finally {
      setLoading(false)
    }
  }

  // --- RAG Handlers ---
  const handleRagFileChange = (e) => {
    setRagFile(e.target.files[0])
  }

  const handleRagUpload = async () => {
    if (!ragFile) return alert("Please select a document!")

    const formData = new FormData()
    formData.append("file", ragFile)

    setRagLoading(true)
    setRagStatus("Uploading to Knowledge Base...")

    try {
      const response = await axios.post("http://127.0.0.1:8000/rag/upload", formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })
      setRagStatus(`Document indexed! Added ${response.data.chunks_added} chunks.`)
    } catch (error) {
      console.error(error)
      setRagStatus("Error: " + (error.response?.data?.detail || "Upload failed"))
    } finally {
      setRagLoading(false)
      setRagFile(null)
    }
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>SpeechInSight</h1>
        <div className="tabs">
          <button
            className={`tab-btn ${activeTab === 'pipeline' ? 'active' : ''}`}
            onClick={() => setActiveTab('pipeline')}
          >
            Audio Pipeline
          </button>
          <button
            className={`tab-btn ${activeTab === 'rag' ? 'active' : ''}`}
            onClick={() => setActiveTab('rag')}
          >
            Knowledge Base (RAG)
          </button>
        </div>
      </header>

      {activeTab === 'pipeline' && (
        <div className="tab-content fade-in">
          {/* Upload Section */}
          <div className="card glass-panel">
            <h2>Audio Analysis</h2>
            <div className="upload-group">
              <input type="file" onChange={handleFileChange} accept="audio/*,video/*" className="file-input" />
              <button onClick={handleUpload} disabled={loading || !file} className="primary-btn">
                {loading ? "Processing..." : "Start Analysis Pipeline"}
              </button>
            </div>
            <p className="status-text">{status}</p>
          </div>

          {/* Report */}
          <ReportPanel report={report} />

          {/* Results Table */}
          {results.length > 0 && (
            <div className="table-container fade-in">
              <table className="results-table">
                <thead>
                  <tr>
                    <th>Speaker</th>
                    <th>Transcript</th>
                    <th>Emotion</th>
                    <th>Template</th>
                    <th>Audio</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((row, index) => (
                    <tr key={index}>
                      <td><strong>{row.speaker}</strong></td>
                      <td className="transcript-cell">{row.text}</td>
                      <td className="emotion-cell">
                        <EmotionBadge emotion={row.emotion} confidence={row.confidence} />
                        {row.sarcasm && <SarcasmBadge />}
                        <AmbiguityIndicator score={row.ambiguity_score} />
                      </td>
                      <td className="template-cell">
                        <TemplateBadge label={row.template_label} confidence={row.template_confidence} />
                      </td>
                      <td>
                        <audio controls src={`http://127.0.0.1:8000${row.audio_url}`} className="audio-player" />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {activeTab === 'rag' && (
        <div className="tab-content fade-in">
          <div className="card glass-panel rag-upload-panel">
            <h2>Knowledge Base Document Upload</h2>
            <p className="rag-description">
              Upload PDF, DOCX, or TXT documents to build the knowledge base.
              These documents are used to evaluate meeting segments against your organization's guidelines.
            </p>
            <div className="upload-group">
              <input type="file" onChange={handleRagFileChange} accept=".pdf,.docx,.txt" className="file-input" />
              <button onClick={handleRagUpload} disabled={ragLoading || !ragFile} className="primary-btn">
                {ragLoading ? "Uploading..." : "Index Document"}
              </button>
            </div>
            <p className="status-text">{ragStatus}</p>
          </div>
        </div>
      )}
    </div>
  )
}

export default App