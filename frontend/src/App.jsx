import { useState } from 'react'
import axios from 'axios'
import './App.css'

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [previewURL, setPreviewURL] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleFileChange = (e) => {
    const file = e.target.files[0]
    if (file) {
      setSelectedFile(file)
      setPreviewURL(URL.createObjectURL(file))
      setResult(null)
      setError(null)
    }
  }

  const handleUpload = async () => {
    if (!selectedFile) return
    setLoading(true)
    setError(null)

    const formData = new FormData()
    formData.append("file", selectedFile)

    try {
      const response = await axios.post("http://localhost:8000/analyze_receipt/", formData, {
        headers: { "Content-Type": "multipart/form-data" }
      })
      setResult(response.data)
    } catch (err) {
      setError(err.response?.data?.detail || "An error occurred while connecting to the server.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="container">
      <header>
        <h1>
  <span>🧾</span>
  <span>AI Receipt Analyzer</span>
</h1>
        <p>Upload a receipt image and let our Layout-Aware GNN extract the key information automatically.</p>
      </header>

      <main className="main-content">
        {/* Upload Section */}
        <div className="upload-section card">
          <input type="file" accept="image/*" onChange={handleFileChange} id="fileInput" className="file-input" />
          <label htmlFor="fileInput" className="upload-btn">
            <span className="icon">📁</span>
            {selectedFile ? 'Change Image' : 'Select Receipt Image'}
          </label>

          {previewURL && (
            <div className="preview-box">
              <img src={previewURL} alt="Preview" className="image-preview" />
              <button className="analyze-btn" onClick={handleUpload} disabled={loading}>
                {loading ? '🧠 AI is analyzing...' : '✨ Analyze Receipt'}
              </button>
            </div>
          )}
          {error && <div className="error-message">❌ {error}</div>}
        </div>

        {/* Result Section */}
        {result && result.status === "success" && (
          <div className="result-section card">
            <h2>Extraction Summary</h2>
            <p className="subtitle">Total lines detected: <strong>{result.total_lines_detected}</strong></p>
            
            <ul className="summary-list">
              {result.summary.map((item, index) => (
                <li key={index} className="summary-item">
                  <span className="confidence-badge">
                    {Math.round(item.confidence * 100)}%
                  </span>
                  <span className="extracted-text">{item.text}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </main>
    </div>
  )
}

export default App