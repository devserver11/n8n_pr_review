import React, { useState, useEffect, useRef } from "react";
import ReactMarkdown from 'react-markdown';
import axios from "axios";
import "./App.css";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

function App() {
  // Collection status
  const [collectionStatus, setCollectionStatus] = useState(null);
  const [loadingStatus, setLoadingStatus] = useState(true);

  // Upload state
  const [file, setFile] = useState(null);
  const [lenderName, setLenderName] = useState("");
  const [lenderType, setLenderType] = useState("");
  const [uploading, setUploading] = useState(false);

  // Chat state
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  
  // Conversation state
  const [conversations, setConversations] = useState([]);
  const [activeConversationId, setActiveConversationId] = useState(null);
  const [loadingConversations, setLoadingConversations] = useState(false);

  // Active page
  const [activePage, setActivePage] = useState("chat");

  // Refs
  const messagesEndRef = useRef(null);
  const fileInputRef = useRef(null);

  // Auto-scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Fetch Collection Status
  const fetchCollectionStatus = async () => {
    setLoadingStatus(true);
    try {
      const res = await axios.get(`${API_URL}/collection/status`);
      setCollectionStatus(res.data);
    } catch (err) {
      console.error("Failed to fetch collection status:", err);
      setCollectionStatus({ status: "error", points_count: 0, documents: [] });
    } finally {
      setLoadingStatus(false);
    }
  };

  useEffect(() => {
    fetchCollectionStatus();
    fetchConversations();
  }, []);

  // Fetch Conversations
  const fetchConversations = async () => {
    setLoadingConversations(true);
    try {
      const res = await axios.get(`${API_URL}/conversations`);
      setConversations(res.data.conversations || []);
    } catch (err) {
      console.error("Failed to fetch conversations:", err);
    } finally {
      setLoadingConversations(false);
    }
  };

  // Load a specific conversation
  const loadConversation = async (conversationId) => {
    try {
      const res = await axios.get(`${API_URL}/conversations/${conversationId}`);
      setMessages(res.data.messages.map(msg => ({
        role: msg.role,
        content: msg.content,
        sources: msg.sources || []
      })));
      setActiveConversationId(conversationId);
      setActivePage("chat");
    } catch (err) {
      console.error("Failed to load conversation:", err);
      alert("Failed to load conversation");
    }
  };

  // Start new conversation
  const startNewConversation = () => {
    setMessages([]);
    setActiveConversationId(null);
    setActivePage("chat");
  };

  // Delete conversation
  const deleteConversation = async (conversationId, e) => {
    e.stopPropagation(); // Prevent triggering loadConversation
    if (!window.confirm("Delete this conversation?")) return;
    
    try {
      await axios.delete(`${API_URL}/conversations/${conversationId}`);
      await fetchConversations();
      if (activeConversationId === conversationId) {
        startNewConversation();
      }
    } catch (err) {
      console.error("Failed to delete conversation:", err);
      alert("Failed to delete conversation");
    }
  };

  // Upload PDF
  const uploadPdf = async () => {
    if (!file) return alert("Select a PDF first");
    if (!lenderName || !lenderType) return alert("Please enter Lender Name and Lender Type");

    setUploading(true);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("lender_name", lenderName);
    formData.append("lender_type", lenderType);

    try {
      await axios.post(`${API_URL}/upload`, formData);
      await fetchCollectionStatus();
      setLenderName("");
      setLenderType("");
      setFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
      alert("✅ PDF uploaded and indexed successfully!");
    } catch (err) {
      console.error(err);
      alert("❌ PDF parsing failed");
    } finally {
      setUploading(false);
    }
  };

  // Send Chat Message
  const sendMessage = async () => {
    if (!input.trim()) return;

    const userMsg = { role: "user", content: input };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setChatLoading(true);

    try {
      const res = await axios.post(`${API_URL}/chat`, {
        question: userMsg.content,
        conversation_id: activeConversationId,
        top_k: 5
      });

      // Update conversation ID if new conversation was created
      if (res.data.conversation_id && !activeConversationId) {
        setActiveConversationId(res.data.conversation_id);
      }

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: res.data.answer,
          sources: res.data.sources || []
        }
      ]);

      // Refresh conversations list
      fetchConversations();
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "Server is busy. Please try again later." }
      ]);
    } finally {
      setChatLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
    setActiveConversationId(null);
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey && !chatLoading) {
      e.preventDefault();
      sendMessage();
    }
  };

  // Delete Document by file_id
  const deleteDocument = async (fileId, fileName) => {
    if (!window.confirm(`Are you sure you want to delete "${fileName}"?\n\nThis will remove all chunks associated with this document.`)) {
      return;
    }

    try {
      const res = await axios.delete(`${API_URL}/document/${encodeURIComponent(fileId)}`);
      if (res.data.status === "success") {
        alert(`✅ ${res.data.message}`);
        await fetchCollectionStatus();
      } else {
        alert(`❌ Failed to delete: ${res.data.message}`);
      }
    } catch (err) {
      console.error("Delete failed:", err);
      alert("❌ Failed to delete document. Please try again.");
    }
  };

  return (
    <div className="app-layout">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="sidebar-logo">
            <div className="logo-icon">📋</div>
            <div className="logo-text">
              <h1>MortgageLens</h1>
              <span>Document Intelligence</span>
            </div>
          </div>
        </div>

        <nav className="sidebar-nav">
          <div className="nav-section">
            <div className="nav-section-title">Main Menu</div>
            
            <button
              className={`nav-item ${activePage === "chat" && !activeConversationId ? "active" : ""}`}
              onClick={startNewConversation}
            >
              <span className="icon">✨</span>
              <span className="label">New Chat</span>
            </button>

            <button
              className={`nav-item ${activePage === "upload" ? "active" : ""}`}
              onClick={() => setActivePage("upload")}
            >
              <span className="icon">📤</span>
              <span className="label">Upload PDF</span>
            </button>

            <button
              className={`nav-item ${activePage === "documents" ? "active" : ""}`}
              onClick={() => setActivePage("documents")}
            >
              <span className="icon">📂</span>
              <span className="label">Documents</span>
            </button>
          </div>

          {/* Conversation History */}
          <div className="nav-section conversations-section">
            <div className="nav-section-title">
              Chat History
              {loadingConversations && <span className="loading-dot">...</span>}
            </div>
            
            <div className="conversations-list">
              {conversations.length === 0 ? (
                <div className="no-conversations">No conversations yet</div>
              ) : (
                conversations.map((conv) => (
                  <div
                    key={conv.conversation_id}
                    className={`conversation-item ${activeConversationId === conv.conversation_id ? "active" : ""}`}
                    onClick={() => loadConversation(conv.conversation_id)}
                  >
                    <span className="conv-icon">💬</span>
                    <div className="conv-info">
                      <div className="conv-title">{conv.title}</div>
                      <div className="conv-meta">{conv.message_count/2} messages</div>
                    </div>
                    <button
                      className="conv-delete"
                      onClick={(e) => deleteConversation(conv.conversation_id, e)}
                      title="Delete conversation"
                    >
                      ×
                    </button>
                  </div>
                ))
              )}
            </div>
          </div>
        </nav>

        <div className="sidebar-footer">
          <div className="stats-grid">
            <div className="stat-item">
              <div className="stat-value">{collectionStatus?.documents_count || 0}</div>
              <div className="stat-label">Docs</div>
            </div>
            <div className="stat-item">
              <div className="stat-value">{collectionStatus?.points_count || 0}</div>
              <div className="stat-label">Chunks</div>
            </div>
          </div>
          <div className={`status-badge ${collectionStatus?.status === "ready" ? "ready" : "empty"}`}>
            {collectionStatus?.status === "ready" ? "Ready" : "Empty"}
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="main-content">
        {/* Chat Page */}
        {activePage === "chat" && (
          <>
            <div className="content-header">
              <div className="page-title">
                <span className="icon">💬</span>
                <div>
                  <h2>Chat with Documents</h2>
                  <p>Ask questions about your mortgage guidelines</p>
                </div>
              </div>
              {messages.length > 0 && (
                <button className="clear-btn" onClick={clearChat}>
                   New Chat
                </button>
              )}
            </div>

            <div className="content-card">
              {collectionStatus?.status !== "ready" ? (
                <div className="empty-state">
                  <div className="icon">📭</div>
                  <h3>No Documents Available</h3>
                  <p>Upload a PDF to start chatting with your mortgage guidelines</p>
                  <button className="empty-state-btn" onClick={() => setActivePage("upload")}>
                    Upload PDF
                  </button>
                </div>
              ) : (
                <div className="chat-container">
                  <div className="chat-messages">
                    {messages.length === 0 ? (
                      <div className="chat-empty">
                        <div className="icon">🤖</div>
                        <h3>Start a Conversation</h3>
                        <p>Ask me anything about your mortgage documents!</p>
                        <p style={{ fontSize: '13px', marginTop: '8px', opacity: 0.7 }}>
                          Try: "What is the max LTV for a VA loan?"
                        </p>
                      </div>
                    ) : (
                      messages.map((msg, idx) => (
                        <div key={idx} className={`message ${msg.role}`}>
                          <div className="message-bubble">
                            <div className="message-sender">
                              {msg.role === "user" ? "You" : "AI Advisor"}
                            </div>
                            <div className="message-content markdown-body">
                              <ReactMarkdown
                                components={{
                                  a: ({ children, ...props }) => (
                                    <span
                                      className="citation-tag"
                                      onClick={(e) => e.preventDefault()}
                                      {...props}
                                    >
                                      📎 {children}
                                    </span>
                                  )
                                }}
                              >
                                {msg.content}
                              </ReactMarkdown>
                            </div>
                          </div>
                        </div>
                      ))
                    )}
                    {chatLoading && (
                      <div className="message assistant">
                        <div className="typing-indicator">
                          <div className="typing-dots">
                            <span></span>
                            <span></span>
                            <span></span>
                          </div>
                          <span style={{ marginLeft: '8px', fontSize: '13px', color: 'var(--text-muted)' }}>
                            Analyzing documents...
                          </span>
                        </div>
                      </div>
                    )}
                    <div ref={messagesEndRef} />
                  </div>

                  <div className="chat-input-container">
                    <input
                      className="chat-input"
                      placeholder="Ask about mortgage guidelines, LTV limits, eligibility..."
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      onKeyDown={handleKeyPress}
                      disabled={chatLoading}
                    />
                    <button
                      className="send-btn"
                      onClick={sendMessage}
                      disabled={chatLoading || !input.trim()}
                    >
                      {chatLoading ? "⏳" : "🚀"} Send
                    </button>
                  </div>
                </div>
              )}
            </div>
          </>
        )}

        {/* Upload Page */}
        {activePage === "upload" && (
          <>
            <div className="content-header">
              <div className="page-title">
                <span className="icon">📤</span>
                <div>
                  <h2>Upload PDF</h2>
                  <p>Add mortgage guidelines to the knowledge base</p>
                </div>
              </div>
            </div>

            <div className="content-card">
              <div className="upload-container">
                <div className="upload-zone">
                  <div className="upload-icon">📁</div>
                  <div className="upload-text">
                    <h3>Upload Mortgage Guidelines</h3>
                    <p>Select a PDF file to parse and index</p>
                  </div>

                  <div className="form-row">
                    <div className="form-field">
                      <label>Lender Name</label>
                      <input
                        className="form-input"
                        type="text"
                        placeholder="e.g., Wells Fargo"
                        value={lenderName}
                        onChange={(e) => setLenderName(e.target.value)}
                      />
                    </div>
                    <div className="form-field">
                      <label>Product Type</label>
                      <input
                        className="form-input"
                        type="text"
                        placeholder="e.g., VA, FHA, Conventional"
                        value={lenderType}
                        onChange={(e) => setLenderType(e.target.value)}
                      />
                    </div>
                  </div>

                  <div className="file-input-wrapper">
                    <input
                      ref={fileInputRef}
                      className="file-input"
                      type="file"
                      accept=".pdf"
                      onChange={(e) => setFile(e.target.files[0])}
                    />
                    <div className="file-input-label">
                      📎 Choose PDF File
                    </div>
                  </div>

                  {file && (
                    <div className="selected-file">
                      <span className="icon">✅</span>
                      <span className="name">{file.name}</span>
                    </div>
                  )}

                  <button
                    className="upload-btn"
                    onClick={uploadPdf}
                    disabled={uploading || !file || !lenderName || !lenderType}
                  >
                    {uploading ? (
                      <>⏳ Processing...</>
                    ) : (
                      <>📥 Upload & Index</>
                    )}
                  </button>
                </div>
              </div>
            </div>
          </>
        )}

        {/* Documents Page */}
        {activePage === "documents" && (
          <>
            <div className="content-header">
              <div className="page-title">
                <span className="icon">📂</span>
                <div>
                  <h2>Indexed Documents</h2>
                  <p>View all uploaded and processed documents</p>
                </div>
              </div>
              <button className="clear-btn" onClick={fetchCollectionStatus}>
                🔄 Refresh
              </button>
            </div>

            <div className="content-card">
              <div className="documents-container">
                {collectionStatus?.documents?.length > 0 ? (
                  <div className="document-list">
                    {collectionStatus.documents.map((doc, idx) => (
                      <div key={doc.file_id || idx} className="document-card">
                        <div className="document-icon">📄</div>
                        <div className="document-info">
                          <div className="document-name">{doc.file_name}</div>
                          <div className="document-meta">
                            {doc.lender_name} • {doc.lender_type}
                          </div>
                          <div className="document-status">Indexed & Ready</div>
                        </div>
                        <button
                          className="delete-btn"
                          onClick={() => deleteDocument(doc.file_id, doc.file_name)}
                          title="Delete document"
                        >
                          🗑️
                        </button>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="empty-state">
                    <div className="icon">📭</div>
                    <h3>No Documents Yet</h3>
                    <p>Upload your first PDF to get started</p>
                    <button className="empty-state-btn" onClick={() => setActivePage("upload")}>
                      Upload PDF
                    </button>
                  </div>
                )}
              </div>
            </div>
          </>
        )}
      </main>
    </div>
  );
}

export default App;
