import React, { useState, useRef, useEffect, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import axios from 'axios';
import './App.css';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const AGENTS = [
  { id: 'EEBC Expert',                    icon: '🏛️', label: 'EEBC Expert',           color: '#00d4ff', section: 'General' },
  { id: 'Compliance Checker',             icon: '✅', label: 'Compliance Checker',     color: '#00e676', section: 'Compliance' },
  { id: 'ETTV/RTTV Calculator',           icon: '🔢', label: 'ETTV/RTTV Calculator',   color: '#f5a623', section: 'Thermal' },
  { id: 'Solution Advisor',               icon: '💡', label: 'Solution Advisor',        color: '#9c6cfb', section: 'Solutions' },
  { id: 'Envelope Specialist',            icon: '🧱', label: 'Envelope Specialist',     color: '#ff7043', section: 'Sec 4' },
  { id: 'Lighting Specialist',            icon: '💡', label: 'Lighting Specialist',     color: '#ffd600', section: 'Sec 5' },
  { id: 'HVAC Specialist',                icon: '❄️', label: 'HVAC Specialist',         color: '#40c4ff', section: 'Sec 6' },
  { id: 'Service Water Heating Specialist',icon: '🚿',label: 'SWH Specialist',          color: '#ff5252', section: 'Sec 7' },
  { id: 'Electrical Power Specialist',    icon: '⚡', label: 'Electrical Specialist',   color: '#e040fb', section: 'Sec 8' },
];

const TABS = ['💬 Chat', '📐 ETTV/RTTV', '📋 Compliance', '📖 About'];

// ── Typing indicator ──────────────────────────────────────────────────────────
function TypingDots() {
  return (
    <div className="typing-dots">
      {[0,1,2].map(i => (
        <span key={i} style={{ animationDelay: `${i * 0.18}s` }} />
      ))}
    </div>
  );
}

// ── Message bubble ────────────────────────────────────────────────────────────
function MessageBubble({ msg }) {
  const isUser = msg.role === 'user';
  const agent = msg.agent ? AGENTS.find(a => a.id === msg.agent) : null;
  const color = agent?.color || '#00d4ff';

  return (
    <div className={`msg-row ${isUser ? 'msg-user' : 'msg-ai'}`}>
      {!isUser && (
        <div className="msg-avatar" style={{ borderColor: color }}>
          {agent ? agent.icon : '🤖'}
        </div>
      )}
      <div className={`msg-bubble ${isUser ? 'bubble-user' : 'bubble-ai'}`}
           style={!isUser ? { borderLeftColor: color } : {}}>
        {!isUser && msg.agentsUsed && (
          <div className="msg-meta">
            {msg.agentsUsed.map(a => {
              const ag = AGENTS.find(x => x.id === a);
              return (
                <span key={a} className="agent-badge" style={{ borderColor: ag?.color || '#00d4ff', color: ag?.color || '#00d4ff' }}>
                  {ag?.icon} {ag?.label || a}
                </span>
              );
            })}
            {msg.routingMethod && (
              <span className="routing-badge">via {msg.routingMethod}</span>
            )}
          </div>
        )}
        <div className="msg-content">
          {isUser ? (
            <p>{msg.content}</p>
          ) : (
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
          )}
        </div>
        {msg.timestamp && (
          <div className="msg-time">{msg.timestamp}</div>
        )}
      </div>
    </div>
  );
}

// ── Chat Tab ──────────────────────────────────────────────────────────────────
function ChatTab({ mode }) {
  const [messages, setMessages] = useState([{
    id: 0, role: 'ai', content:
      `**Welcome to the EEBC 2021 AI Compliance System!**\n\nI am your orchestrated multi-agent assistant. Ask me anything about:\n- **Compliance checking** for your building parameters\n- **ETTV/RTTV calculations** with step-by-step working\n- **Solution recommendations** for non-compliant items\n- **Specialist questions** on Envelope, Lighting, HVAC, SWH, or Electrical\n\nWhat would you like to know?`,
    agentsUsed: ['EEBC Expert'],
  }]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [agentMode, setAgentMode] = useState('auto'); // 'auto' | specific agent id
  const bottomRef = useRef(null);
  const textareaRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const sendMessage = useCallback(async () => {
    const q = input.trim();
    if (!q || loading) return;

    const userMsg = { id: Date.now(), role: 'user', content: q };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      let answer, agentsUsed, routingMethod, timestamp;

      if (agentMode === 'auto') {
        const res = await axios.post(`${API_BASE}/agent/ask`, { question: q, session_id: 'default' });
        answer = res.data.answer;
        agentsUsed = res.data.agents_used;
        routingMethod = res.data.routing_method;
        timestamp = res.data.timestamp;
      } else {
        const res = await axios.post(`${API_BASE}/tools/rag`, { question: q, agent_type: agentMode });
        answer = res.data.answer;
        agentsUsed = [agentMode];
        routingMethod = 'direct';
        timestamp = res.data.timestamp;
      }

      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: 'ai',
        content: answer,
        agentsUsed,
        routingMethod,
        timestamp,
      }]);
    } catch (err) {
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: 'ai',
        content: `**Error:** ${err.response?.data?.detail || err.message}\n\nPlease ensure the backend is running at \`${API_BASE}\`.`,
        agentsUsed: [],
      }]);
    } finally {
      setLoading(false);
    }
  }, [input, loading, agentMode]);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  };

  const clearMemory = async () => {
    try {
      await axios.delete(`${API_BASE}/agent/memory`, { data: { session_id: 'default' } });
      setMessages([{ id: Date.now(), role: 'ai', content: 'Memory cleared. Starting fresh conversation.', agentsUsed: [] }]);
    } catch {}
  };

  return (
    <div className="chat-layout">
      {/* Agent selector */}
      <div className="agent-selector-bar">
        <span className="selector-label">Route to:</span>
        <div className="agent-pills">
          <button
            className={`agent-pill ${agentMode === 'auto' ? 'active' : ''}`}
            onClick={() => setAgentMode('auto')}
          >
            🧠 Auto-route
          </button>
          {AGENTS.map(a => (
            <button
              key={a.id}
              className={`agent-pill ${agentMode === a.id ? 'active' : ''}`}
              style={agentMode === a.id ? { borderColor: a.color, color: a.color, background: a.color + '20' } : {}}
              onClick={() => setAgentMode(a.id)}
            >
              {a.icon} {a.section}
            </button>
          ))}
        </div>
        <button className="clear-btn" onClick={clearMemory} title="Clear memory">🗑️</button>
      </div>

      {/* Messages */}
      <div className="messages-area">
        {messages.map(msg => <MessageBubble key={msg.id} msg={msg} />)}
        {loading && (
          <div className="msg-row msg-ai">
            <div className="msg-avatar">🤖</div>
            <div className="msg-bubble bubble-ai thinking-bubble">
              <TypingDots />
              <span className="thinking-text">Analysing with EEBC 2021 knowledge base...</span>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="input-area">
        <div className="input-wrapper">
          <textarea
            ref={textareaRef}
            className="chat-input"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about EEBC 2021 compliance, ETTV/RTTV calculations, solutions..."
            rows={2}
            disabled={loading}
          />
          <button
            className="send-btn"
            onClick={sendMessage}
            disabled={loading || !input.trim()}
          >
            {loading ? <span className="spinner" /> : '↑'}
          </button>
        </div>
        <p className="input-hint">Enter to send · Shift+Enter for new line</p>
      </div>
    </div>
  );
}

// ── ETTV / RTTV Tab ───────────────────────────────────────────────────────────
function ETTVTab() {
  const [mode, setMode] = useState('ETTV');
  const [result, setResult] = useState('');
  const [loading, setLoading] = useState(false);

  // ETTV fields
  const [ettv, setEttv] = useState({ Uw: '', Aw: '', TDeq: '', Uf: '', Af: '', DT: '5', SC: '', SF: '', At: '' });
  // RTTV fields
  const [rttv, setRttv] = useState({ Ur: '', Ar: '', TDeqr: '25', Us: '', As: '', DTs: '5', SCs: '', SFs: '', AtRoof: '' });

  const labelMap = {
    ETTV: [
      { key: 'Uw',   label: 'Uw — Opaque wall U-value',      unit: 'W/m²·K', hint: 'e.g. 0.5' },
      { key: 'Aw',   label: 'Aw — Opaque wall area',          unit: 'm²',     hint: 'e.g. 120' },
      { key: 'TDeq', label: 'TDeq — Equiv temp diff (wall)',   unit: '°C',     hint: 'e.g. 15' },
      { key: 'Uf',   label: 'Uf — Fenestration U-value',       unit: 'W/m²·K', hint: 'e.g. 3.0' },
      { key: 'Af',   label: 'Af — Fenestration area',          unit: 'm²',     hint: 'e.g. 30' },
      { key: 'DT',   label: 'ΔT — Indoor-outdoor temp diff',   unit: '°C',     hint: 'Default: 5' },
      { key: 'SC',   label: 'SC — Shading Coefficient',        unit: '',       hint: '0–1, e.g. 0.6' },
      { key: 'SF',   label: 'SF — Solar Factor (orientation)',  unit: 'W/m²',   hint: 'From EEBC Table 4.1' },
      { key: 'At',   label: 'At — Gross external wall area',   unit: 'm²',     hint: 'Aw + Af' },
    ],
    RTTV: [
      { key: 'Ur',     label: 'Ur — Opaque roof U-value',      unit: 'W/m²·K', hint: 'e.g. 0.5' },
      { key: 'Ar',     label: 'Ar — Opaque roof area',          unit: 'm²',     hint: 'e.g. 200' },
      { key: 'TDeqr',  label: 'TDeqr — Roof temp diff equiv',  unit: '°C',     hint: 'Default: 25' },
      { key: 'Us',     label: 'Us — Skylight U-value',          unit: 'W/m²·K', hint: 'e.g. 3.0' },
      { key: 'As',     label: 'As — Skylight area',             unit: 'm²',     hint: '0 if no skylight' },
      { key: 'DTs',    label: 'ΔTs — Skylight temp diff',       unit: '°C',     hint: 'Default: 5' },
      { key: 'SCs',    label: 'SCs — Skylight SC',              unit: '',       hint: '0–1' },
      { key: 'SFs',    label: 'SFs — Skylight Solar Factor',    unit: 'W/m²',   hint: 'From EEBC Table' },
      { key: 'AtRoof', label: 'Total roof area (Ar + As)',       unit: 'm²',     hint: 'Ar + As' },
    ],
  };

  const currentFields = mode === 'ETTV' ? ettv : rttv;
  const setFields = mode === 'ETTV' ? setEttv : setRttv;

  const calculate = async () => {
    setLoading(true);
    setResult('');
    const fields = mode === 'ETTV' ? ettv : rttv;
    const lines = labelMap[mode]
      .filter(f => fields[f.key])
      .map(f => `${f.label}: ${fields[f.key]} ${f.unit}`);

    const prompt = `Calculate ${mode} using these project parameters:\n${lines.join('\n')}\n\nPlease show full step-by-step working, state the result in W/m², compare to EEBC 2021 limit, and declare PASS or FAIL.`;

    try {
      const res = await axios.post(`${API_BASE}/tools/rag`, {
        question: prompt,
        agent_type: 'ETTV/RTTV Calculator',
      });
      setResult(res.data.answer);
    } catch (err) {
      setResult(`**Error:** ${err.response?.data?.detail || err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="ettv-layout">
      <div className="ettv-header">
        <h2 className="section-title">ETTV / RTTV Calculator</h2>
        <p className="section-subtitle">Step-by-step thermal transfer value calculations per EEBC 2021 Section 4</p>
        <div className="mode-toggle">
          {['ETTV', 'RTTV'].map(m => (
            <button
              key={m}
              className={`mode-btn ${mode === m ? 'active' : ''}`}
              onClick={() => setMode(m)}
            >
              {m === 'ETTV' ? '🧱 ETTV (Wall)' : '🏠 RTTV (Roof)'}
            </button>
          ))}
        </div>
      </div>

      <div className="ettv-grid">
        <div className="ettv-inputs">
          <div className="card">
            <h3 className="card-title">Input Parameters</h3>
            <div className="fields-grid">
              {labelMap[mode].map(f => (
                <div className="field-group" key={f.key}>
                  <label className="field-label">{f.label}</label>
                  <div className="field-input-wrap">
                    <input
                      className="field-input"
                      type="number"
                      placeholder={f.hint}
                      value={currentFields[f.key]}
                      onChange={e => setFields(prev => ({ ...prev, [f.key]: e.target.value }))}
                    />
                    {f.unit && <span className="field-unit">{f.unit}</span>}
                  </div>
                </div>
              ))}
            </div>
            <div className="limits-box">
              <span className="limit-item">ETTV limit: <strong>≤ 50 W/m²</strong></span>
              <span className="limit-item">RTTV limit: <strong>≤ 25 W/m²</strong></span>
            </div>
            <button className="calc-btn" onClick={calculate} disabled={loading}>
              {loading ? <><span className="spinner" /> Calculating...</> : `▶ Calculate ${mode}`}
            </button>
          </div>
        </div>

        <div className="ettv-result">
          <div className="card result-card">
            <h3 className="card-title">Calculation Result</h3>
            {loading && (
              <div className="loading-state">
                <TypingDots />
                <p>Running EEBC 2021 formula...</p>
              </div>
            )}
            {result ? (
              <div className="result-content">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{result}</ReactMarkdown>
              </div>
            ) : !loading && (
              <div className="empty-state">
                <div className="empty-icon">🔢</div>
                <p>Fill in the parameters and click Calculate</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Compliance Tab ────────────────────────────────────────────────────────────
function ComplianceTab() {
  const [question, setQuestion] = useState('');
  const [result, setResult] = useState('');
  const [solutions, setSolutions] = useState('');
  const [loading, setLoading] = useState(false);
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  const QUICK_CHECKS = [
    { label: 'Wall U-value check', q: 'My external wall has a U-value of 0.8 W/m²·K. Is this compliant with EEBC 2021?' },
    { label: 'Window WWR check', q: 'My building has a Window-to-Wall Ratio of 65%. Is this compliant?' },
    { label: 'LPD office check', q: 'My office building has a Lighting Power Density of 14 W/m². Is this compliant?' },
    { label: 'Chiller COP check', q: 'My chiller has a COP of 4.2. Does it comply with EEBC 2021 Section 6?' },
    { label: 'Roof U-value', q: 'Flat roof U-value is 0.6 W/m²·K. Is it compliant with EEBC 2021?' },
  ];

  const uploadAndCheck = async () => {
    if (!file) return;
    setUploading(true);
    setUploadStatus(null);
    const form = new FormData();
    form.append('file', file);
    try {
      const res = await axios.post(`${API_BASE}/upload`, form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setUploadStatus({ type: 'success', data: res.data });
      setFile(null);
      const autoQuery = `The document "${res.data.filename}" has been uploaded. Please analyze its content and check if it complies with EEBC 2021 standards. Identify any non-compliant parameters and flag them clearly.`;
      setQuestion(autoQuery);
      check(autoQuery);
    } catch (err) {
      setUploadStatus({ type: 'error', msg: err.response?.data?.detail || err.message });
    } finally {
      setUploading(false);
    }
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f?.name.endsWith('.pdf')) setFile(f);
  };

  const check = async (q) => {
    const query = q || question.trim();
    if (!query) return;
    setLoading(true);
    setResult('');
    setSolutions('');
    try {
      const [compRes, solRes] = await Promise.all([
        axios.post(`${API_BASE}/tools/rag`, { question: query, agent_type: 'Compliance Checker' }),
        axios.post(`${API_BASE}/tools/rag`, { question: `For this scenario: "${query}" — what solutions are recommended if it is non-compliant?`, agent_type: 'Solution Advisor' }),
      ]);
      setResult(compRes.data.answer);
      setSolutions(solRes.data.answer);
    } catch (err) {
      setResult(`**Error:** ${err.response?.data?.detail || err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="compliance-layout">
      <div className="section-header">
        <h2 className="section-title">Compliance Checker + Solution Advisor</h2>
        <p className="section-subtitle">Check your building parameters against EEBC 2021 — get instant compliance verdict and remediation advice</p>
      </div>

      <div className="card upload-compliance-card">
        <h3 className="card-title">📄 Upload PDF for Compliance Check</h3>
        <p style={{ color: '#8899aa', fontSize: '0.85rem', marginBottom: '1rem' }}>
          Upload a building document PDF to automatically check its compliance with EEBC 2021
        </p>
        <div
          className={`drop-zone ${dragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`}
          style={{ marginBottom: '1rem' }}
          onDragOver={e => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          onClick={() => inputRef.current?.click()}
        >
          <input ref={inputRef} type="file" accept=".pdf" style={{ display: 'none' }}
                 onChange={e => setFile(e.target.files[0])} />
          {file ? (
            <>
              <div className="drop-icon">📄</div>
              <p className="drop-filename">{file.name}</p>
              <p className="drop-size">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
            </>
          ) : (
            <>
              <div className="drop-icon">☁️</div>
              <p className="drop-text">Drop PDF here or click to browse</p>
              <p className="drop-hint">PDF files only</p>
            </>
          )}
        </div>
        {file && (
          <button className="upload-btn" onClick={uploadAndCheck} disabled={uploading}>
            {uploading ? <><span className="spinner" /> Uploading & checking...</> : '📤 Upload & Check Compliance'}
          </button>
        )}
        {uploadStatus?.type === 'success' && (
          <div className="status-card success" style={{ marginTop: '0.75rem' }}>
            <p>✅ <strong>{uploadStatus.data.filename}</strong> ingested — running compliance check below...</p>
          </div>
        )}
        {uploadStatus?.type === 'error' && (
          <div className="status-card error" style={{ marginTop: '0.75rem' }}>
            <p>❌ Upload failed: {uploadStatus.msg}</p>
          </div>
        )}
      </div>

      <div className="quick-checks">
        {QUICK_CHECKS.map(qc => (
          <button key={qc.label} className="quick-btn" onClick={() => { setQuestion(qc.q); check(qc.q); }}>
            {qc.label}
          </button>
        ))}
      </div>

      <div className="card">
        <textarea
          className="compliance-input"
          value={question}
          onChange={e => setQuestion(e.target.value)}
          placeholder="Describe your building parameter: e.g. 'My office building wall U-value is 0.9 W/m²·K. Is this compliant with EEBC 2021?'"
          rows={4}
        />
        <button className="check-btn" onClick={() => check()} disabled={loading || !question.trim()}>
          {loading ? <><span className="spinner" /> Checking...</> : '🔍 Check Compliance'}
        </button>
      </div>

      {loading && (
        <div className="checking-anim">
          <TypingDots />
          <span>Running compliance analysis + generating solutions...</span>
        </div>
      )}

      {result && (
        <div className="results-grid">
          <div className="card compliance-result-card">
            <h3 className="card-title">✅ Compliance Analysis</h3>
            <div className="result-content">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{result}</ReactMarkdown>
            </div>
          </div>
          {solutions && (
            <div className="card solution-result-card">
              <h3 className="card-title">💡 Solution Advisor</h3>
              <div className="result-content">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{solutions}</ReactMarkdown>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ── Upload Tab ────────────────────────────────────────────────────────────────
function UploadTab() {
  const [file, setFile] = useState(null);
  const [status, setStatus] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [vectorStatus, setVectorStatus] = useState(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  useEffect(() => {
    fetchVectorStatus();
  }, []);

  const fetchVectorStatus = async () => {
    try {
      const res = await axios.get(`${API_BASE}/vector/status`);
      setVectorStatus(res.data);
    } catch {}
  };

  const upload = async () => {
    if (!file) return;
    setUploading(true);
    setStatus(null);
    const form = new FormData();
    form.append('file', file);
    try {
      const res = await axios.post(`${API_BASE}/upload`, form, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setStatus({ type: 'success', data: res.data });
      fetchVectorStatus();
    } catch (err) {
      setStatus({ type: 'error', msg: err.response?.data?.detail || err.message });
    } finally {
      setUploading(false);
      setFile(null);
    }
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f?.name.endsWith('.pdf')) setFile(f);
  };

  return (
    <div className="upload-layout">
      <div className="section-header">
        <h2 className="section-title">EEBC Knowledge Base</h2>
        <p className="section-subtitle">Upload the EEBC 2021 PDF to populate the vector database</p>
      </div>

      {vectorStatus && (
        <div className="vector-status-card">
          <div className="vs-icon">🗄️</div>
          <div>
            <p className="vs-title">Vector Store Status</p>
            <p className="vs-count"><strong>{vectorStatus.document_chunks?.toLocaleString()}</strong> document chunks indexed</p>
            <p className="vs-collection">Collection: {vectorStatus.collection}</p>
          </div>
          <span className={`vs-badge ${vectorStatus.document_chunks > 0 ? 'ok' : 'empty'}`}>
            {vectorStatus.document_chunks > 0 ? '✅ Ready' : '⚠️ Empty'}
          </span>
        </div>
      )}

      <div
        className={`drop-zone ${dragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`}
        onDragOver={e => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => inputRef.current?.click()}
      >
        <input ref={inputRef} type="file" accept=".pdf" style={{ display: 'none' }}
               onChange={e => setFile(e.target.files[0])} />
        {file ? (
          <>
            <div className="drop-icon">📄</div>
            <p className="drop-filename">{file.name}</p>
            <p className="drop-size">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
          </>
        ) : (
          <>
            <div className="drop-icon">☁️</div>
            <p className="drop-text">Drop EEBC 2021 PDF here or click to browse</p>
            <p className="drop-hint">PDF files only</p>
          </>
        )}
      </div>

      {file && (
        <button className="upload-btn" onClick={upload} disabled={uploading}>
          {uploading ? <><span className="spinner" /> Ingesting into vector DB...</> : '📤 Upload & Ingest'}
        </button>
      )}

      {status?.type === 'success' && (
        <div className="status-card success">
          <p>✅ <strong>{status.data.filename}</strong> ingested successfully!</p>
          <p>{status.data.chunks_created} chunks created in vector store</p>
        </div>
      )}
      {status?.type === 'error' && (
        <div className="status-card error">
          <p>❌ Upload failed: {status.msg}</p>
        </div>
      )}

      <div className="ingest-instructions">
        <h3>📋 Alternative: Bulk Ingestion via CLI</h3>
        <div className="code-block">
          <span># Place EEBC 2021 PDF in ./Data folder, then run:</span>
          <span>python ingest.py</span>
          <span></span>
          <span># Or watch mode (auto-processes new files):</span>
          <span>python ingest.py --watch</span>
        </div>
      </div>
    </div>
  );
}

// ── About Tab ─────────────────────────────────────────────────────────────────
function AboutTab() {
  const agents = AGENTS;
  return (
    <div className="about-layout">
      <div className="about-hero">
        <h1>EEBC 2021 Agentic AI System</h1>
        <p>Multi-agent RAG architecture for intelligent building energy code compliance</p>
      </div>

      <div className="arch-diagram">
        <div className="arch-node user-node">👤 User Query</div>
        <div className="arch-arrow">↓</div>
        <div className="arch-node orch-node">🧠 Orchestrator<br /><small>LLM Routing + Memory</small></div>
        <div className="arch-arrow">↓</div>
        <div className="arch-agents">
          {agents.map(a => (
            <div key={a.id} className="arch-agent-chip" style={{ borderColor: a.color }}>
              {a.icon} {a.label}
            </div>
          ))}
        </div>
        <div className="arch-arrow">↓</div>
        <div className="arch-node rag-node">📚 RAG (ChromaDB + HuggingFace)<br /><small>EEBC 2021 Vector Store · MMR k=20</small></div>
        <div className="arch-arrow">↓</div>
        <div className="arch-node llm-node">⚡ Groq LLaMA-3.3-70B<br /><small>Final Answer Generation</small></div>
      </div>

      <div className="agents-grid">
        {agents.map(a => (
          <div key={a.id} className="agent-card" style={{ borderTopColor: a.color }}>
            <div className="agent-card-icon" style={{ color: a.color }}>{a.icon}</div>
            <h4 className="agent-card-name">{a.label}</h4>
            <span className="agent-card-section">{a.section}</span>
          </div>
        ))}
      </div>

      <div className="tech-stack">
        <h3>Technology Stack</h3>
        <div className="tech-grid">
          {[
            { label: 'LLM', val: 'Groq LLaMA-3.3-70B' },
            { label: 'Embeddings', val: 'all-MiniLM-L6-v2' },
            { label: 'Vector DB', val: 'ChromaDB (MMR k=20)' },
            { label: 'Backend', val: 'FastAPI + LangChain' },
            { label: 'Frontend', val: 'React 18' },
            { label: 'PDF Loader', val: 'PyMuPDF' },
          ].map(t => (
            <div key={t.label} className="tech-item">
              <span className="tech-label">{t.label}</span>
              <span className="tech-val">{t.val}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ── App Shell ─────────────────────────────────────────────────────────────────
export default function App() {
  const [tab, setTab] = useState(0);

  return (
    <div className="app-shell">
      {/* Header */}
      <header className="app-header">
        <div className="header-brand">
          <div className="brand-icon">⚡</div>
          <div>
            <h1 className="brand-title">EEBC 2021 AI</h1>
            <p className="brand-sub">Agentic Compliance System</p>
          </div>
        </div>
        <div className="header-status">
          <span className="status-dot" />
          <span>API Connected</span>
        </div>
      </header>

      {/* Tabs */}
      <nav className="tab-nav">
        {TABS.map((t, i) => (
          <button
            key={t}
            className={`tab-btn ${tab === i ? 'active' : ''}`}
            onClick={() => setTab(i)}
          >
            {t}
          </button>
        ))}
      </nav>

      {/* Content */}
      <main className="app-main">
        {tab === 0 && <ChatTab />}
        {tab === 1 && <ETTVTab />}
        {tab === 2 && <ComplianceTab />}
        {tab === 3 && <AboutTab />}
      </main>
    </div>
  );
}
