import { useState } from 'react'
import axios from 'axios'
import './App.css'

/* ── Emotion colour map (Ekman 7) ── */
const EMOTION_COLORS = {
  angry:    { bg: '#fee2e2', text: '#991b1b', border: '#fca5a5' },
  disgust:  { bg: '#fef3c7', text: '#92400e', border: '#fcd34d' },
  fear:     { bg: '#ede9fe', text: '#5b21b6', border: '#c4b5fd' },
  happy:    { bg: '#dcfce7', text: '#166534', border: '#86efac' },
  sad:      { bg: '#dbeafe', text: '#1e3a8a', border: '#93c5fd' },
  surprise: { bg: '#fce7f3', text: '#9d174d', border: '#f9a8d4' },
  neutral:  { bg: '#f3f4f6', text: '#374151', border: '#d1d5db' },
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

function SarcasmBadge() {
  return <span className="sarcasm-badge">🎭 sarcasm</span>
}

function AmbiguityIndicator({ score }) {
  if (score < 0.5) return null
  return (
    <span className="ambiguity-indicator" title={`Ambiguity: ${Math.round(score * 100)}%`}>
      ⚠️ ambiguous
    </span>
  )
}

function App() {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState([])
  const [status, setStatus] = useState("")

  const handleFileChange = (e) => {
    setFile(e.target.files[0])
  }

  const handleUpload = async () => {
    if (!file) return alert("Please select a file!")

    const formData = new FormData()
    formData.append("file", file)

    setLoading(true)
    setStatus("🚀 Uploading and Analyzing... (This may take a minute)")
    setResults([])

    try {
      const response = await axios.post("http://127.0.0.1:8000/analyze", formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })

      setResults(response.data.data)
      setStatus("✅ Analysis Complete!")
    } catch (error) {
      console.error(error)
      setStatus("❌ Error: " + (error.response?.data?.detail || "Connection failed"))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app-container">
      <h1>🎙️ SpeechInSight Dashboard</h1>

      {/* Upload Section */}
      <div className="card">
        <input type="file" onChange={handleFileChange} accept="audio/*,video/*" />
        <button onClick={handleUpload} disabled={loading || !file}>
          {loading ? "Processing..." : "Start Analysis Pipeline"}
        </button>
      </div>

      <p>{status}</p>

      {/* Results Table */}
      {results.length > 0 && (
        <table className="results-table">
          <thead>
            <tr>
              <th>Speaker</th>
              <th>Transcript</th>
              <th>Emotion</th>
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
                <td>
                  <audio controls src={`http://127.0.0.1:8000${row.audio_url}`} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

export default App