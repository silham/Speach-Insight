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
  // Tabs: 'pipeline' or 'rag'
  const [activeTab, setActiveTab] = useState('pipeline')

  // Pipeline State
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState([])
  const [status, setStatus] = useState("")

  // RAG State
  const [ragFile, setRagFile] = useState(null)
  const [ragLoading, setRagLoading] = useState(false)
  const [ragStatus, setRagStatus] = useState("")
  const [chatMessages, setChatMessages] = useState([])
  const [chatInput, setChatInput] = useState("")
  const [chatLoading, setChatLoading] = useState(false)

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
      setRagStatus(`✅ Document indexed! Added ${response.data.chunks_added} chunks.`)
    } catch (error) {
      console.error(error)
      setRagStatus("❌ Error: " + (error.response?.data?.detail || "Upload failed"))
    } finally {
      setRagLoading(false)
      setRagFile(null)
    }
  }

  const handleChatSubmit = async (e) => {
    e.preventDefault()
    if (!chatInput.trim()) return

    const userMessage = { role: "user", text: chatInput }
    setChatMessages(prev => [...prev, userMessage])
    setChatInput("")
    setChatLoading(true)

    try {
      const response = await axios.post("http://127.0.0.1:8000/rag/ask", { query: userMessage.text })
      const aiMessage = { role: "ai", text: response.data.answer, context: response.data.context }
      setChatMessages(prev => [...prev, aiMessage])
    } catch (error) {
      console.error(error)
      const errMessage = { role: "ai", text: "❌ Error: Could not get a response." }
      setChatMessages(prev => [...prev, errMessage])
    } finally {
      setChatLoading(false)
    }
  }

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>🎙️ SpeechInSight</h1>
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
          <div className="rag-layout">
            {/* RAG Upload Sidebar */}
            <div className="card glass-panel rag-sidebar">
              <h2>Add Document</h2>
              <p>Upload PDF or TXT to add to the knowledge base.</p>
              <input type="file" onChange={handleRagFileChange} accept=".pdf,.txt" className="file-input" />
              <button onClick={handleRagUpload} disabled={ragLoading || !ragFile} className="primary-btn full-width">
                {ragLoading ? "Uploading..." : "Index Document"}
              </button>
              <p className="status-text">{ragStatus}</p>
            </div>

            {/* Chat Interface */}
            <div className="card glass-panel chat-container">
              <div className="chat-history">
                {chatMessages.length === 0 && (
                  <div className="empty-chat">
                    <p>Ask a question based on your uploaded documents.</p>
                  </div>
                )}
                {chatMessages.map((msg, idx) => (
                  <div key={idx} className={`chat-message ${msg.role}`}>
                    <div className="bubble">
                      <p>{msg.text}</p>
                      {msg.context && (
                        <div className="context-tooltip">Used {msg.context.length} context chunks</div>
                      )}
                    </div>
                  </div>
                ))}
                {chatLoading && (
                  <div className="chat-message ai">
                    <div className="bubble loading-bubble">Thinking...</div>
                  </div>
                )}
              </div>
              <form onSubmit={handleChatSubmit} className="chat-input-form">
                <input
                  type="text"
                  value={chatInput}
                  onChange={(e) => setChatInput(e.target.value)}
                  placeholder="Ask a question..."
                  className="chat-input"
                  disabled={chatLoading}
                />
                <button type="submit" disabled={chatLoading || !chatInput.trim()} className="send-btn">
                  Send
                </button>
              </form>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App