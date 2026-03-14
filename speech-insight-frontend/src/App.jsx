import { useState } from 'react'
import axios from 'axios'
import './App.css'

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
      // Send file to Python API
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
    <div style={{ maxWidth: "800px", margin: "0 auto", padding: "20px" }}>
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
        <table border="1" cellPadding="10" style={{ width: "100%", borderCollapse: "collapse", marginTop: "20px" }}>
          <thead>
            <tr>
              <th>Speaker</th>
              <th>Transcript</th>
              <th>Audio</th>
            </tr>
          </thead>
          <tbody>
            {results.map((row, index) => (
              <tr key={index}>
                <td><strong>{row.speaker}</strong></td>
                <td>{row.text}</td>
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