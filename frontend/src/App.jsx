import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import {
  Plus, Search, Paperclip, ArrowUp, FileText, Microscope,
  AlertTriangle, CheckCircle2, Info, Sparkles, FlaskConical, Activity,
  FileCheck2, RefreshCw, Trash2, Edit2, Check, X, UploadCloud,
  ChevronDown, Copy, Menu, ChevronRight, Download, SlidersHorizontal,
  Bot, Stethoscope, ShieldCheck
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import RangeGauge from "./components/RangeGauge";
import { SAMPLE_REPORTS } from "./data/sampleReports";

const API_URL =
  import.meta.env.VITE_API_BASE_URL ||
  (typeof window !== "undefined" && window.location.port !== "5173"
    ? window.location.origin
    : "http://localhost:8000");

/* ============================================================
   DEFAULT INITIAL THREAD TEMPLATE
   ============================================================ */

const DEFAULT_THREAD = {
  id: "t_initial",
  title: "New Report Analysis",
  group: "Today",
  attached: false,
  flagged: false,
  result: null,
  messages: [],
};

const loadSavedThreads = () => {
  try {
    const saved = localStorage.getItem("patho_gemini_threads");
    if (saved) {
      const parsed = JSON.parse(saved);
      if (Array.isArray(parsed) && parsed.length > 0) return parsed;
    }
  } catch (e) {
    console.error("Local storage error:", e);
  }
  return [DEFAULT_THREAD];
};

const loadSavedActiveId = (validThreads) => {
  try {
    const saved = localStorage.getItem("patho_gemini_active_id");
    if (saved && validThreads.some((t) => t.id === saved)) return saved;
  } catch (e) {}
  return validThreads[0]?.id || DEFAULT_THREAD.id;
};

/* ============================================================
   STATUS BADGE COMPONENT
   ============================================================ */

function StatusBadge({ status }) {
  const normStatus = (status || "UNKNOWN").toUpperCase();
  const map = {
    NORMAL: { label: "NORMAL", class: "normal", icon: CheckCircle2 },
    HIGH: { label: "HIGH", class: "high", icon: AlertTriangle },
    LOW: { label: "LOW", class: "low", icon: Info },
    BORDERLINE_HIGH: { label: "BORDERLINE HIGH", class: "high", icon: AlertTriangle },
    BORDERLINE_LOW: { label: "BORDERLINE LOW", class: "low", icon: Info },
    CRITICAL: { label: "CRITICAL", class: "critical", icon: AlertTriangle },
    CRITICAL_HIGH: { label: "CRITICAL HIGH", class: "critical", icon: AlertTriangle },
    CRITICAL_LOW: { label: "CRITICAL LOW", class: "critical", icon: AlertTriangle },
  };
  const b = map[normStatus] || { label: normStatus.replace("_", " "), class: "low", icon: Info };
  const IconComponent = b.icon;

  return (
    <span className={`status-pill ${b.class}`}>
      <IconComponent size={11} />
      <span>{b.label}</span>
    </span>
  );
}

/* ============================================================
   MAIN GEMINI-INSPIRED CLINICAL INTERFACE
   ============================================================ */

export default function App() {
  const [threads, setThreads] = useState(loadSavedThreads);
  const [activeId, setActiveId] = useState(() => loadSavedActiveId(threads));

  // Navigation & Drawer
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [editingId, setEditingId] = useState(null);
  const [editTitle, setEditTitle] = useState("");

  // Report & Matrix View Controls
  const [matrixOpen, setMatrixOpen] = useState(true);
  const [matrixFilter, setMatrixFilter] = useState("ALL"); // ALL, ABNORMAL, CRITICAL, NORMAL
  const [matrixSearch, setMatrixSearch] = useState("");
  const [activeTab, setActiveTab] = useState("matrix"); // "matrix" | "insights"

  // Chat & Engine
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [chatLoading, setChatLoading] = useState(false);
  const [engineInfo, setEngineInfo] = useState(null);
  const [copiedIdx, setCopiedIdx] = useState(null);
  const [attachMenuOpen, setAttachMenuOpen] = useState(false);

  const fileInputRef = useRef(null);
  const chatBottomRef = useRef(null);
  const textareaRef = useRef(null);

  const activeThread = threads.find((t) => t.id === activeId) || threads[0] || DEFAULT_THREAD;
  const result = activeThread?.result;
  const messages = activeThread?.messages || [];
  const hasStarted = messages.length > 0 || result !== null;

  // Fetch active LLM engine info
  useEffect(() => {
    axios
      .get(`${API_URL}/chat/info`)
      .then((res) => setEngineInfo(res.data))
      .catch((err) => console.log("AI info endpoint:", err.message));
  }, []);

  // Sync to localStorage
  useEffect(() => {
    try {
      localStorage.setItem("patho_gemini_threads", JSON.stringify(threads));
      localStorage.setItem("patho_gemini_active_id", activeId);
    } catch (e) {
      console.error("Local storage sync error:", e);
    }
  }, [threads, activeId]);

  // Auto-scroll chat to bottom
  useEffect(() => {
    if (hasStarted) {
      chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages.length, chatLoading, hasStarted]);

  // Handle PDF upload
  const handleFileUpload = async (uploadedFile) => {
    if (!uploadedFile) return;
    setLoading(true);
    setAttachMenuOpen(false);

    const formData = new FormData();
    formData.append("file", uploadedFile);

    try {
      const response = await axios.post(`${API_URL}/upload`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      const resData = response.data;
      const patientName =
        resData.extraction?.patient?.name &&
        !["name", "unknown", "patient name"].includes(resData.extraction.patient.name.toLowerCase().trim())
          ? resData.extraction.patient.name
          : "Pathology Specimen";

      const category = resData.extraction?.document_category || "Report";
      const newTitle = `${category} — ${patientName}`;

      const totalTests = resData.extraction?.sections?.flatMap((s) => s.tests)?.length || 0;
      const abnormalCount = (resData.extraction?.sections?.flatMap((s) => s.tests) || []).filter(
        (t) => t.status && !t.status.toUpperCase().includes("NORMAL")
      ).length;

      const welcomeMsg = {
        sender: "ai",
        text: `### 🧪 Pathology Report Analyzed: ${newTitle}\n\n**${patientName}** | Total Biomarkers: **${totalTests}** (${abnormalCount} flagged out of reference bounds).\n\n${
          resData.interpretation?.summary || "Report processed successfully."
        }\n\n*Review the interactive Biomarker Matrix above or ask me any question like "Is the patient okay?", "Explain out-of-range tests", or "What dietary changes are recommended?"*`,
      };

      setThreads((prev) =>
        prev.map((t) =>
          t.id === activeId
            ? {
                ...t,
                title: newTitle,
                attached: true,
                flagged: resData.interpretation?.requires_review,
                result: resData,
                messages: [welcomeMsg],
              }
            : t
        )
      );
    } catch (err) {
      console.error("Upload error:", err);
      alert(err.response?.data?.detail || "Failed to process PDF report. Please verify backend server is online.");
    } finally {
      setLoading(false);
    }
  };

  // Handle loading sample report
  const handleLoadSample = (sample) => {
    const newTitle = sample.name;
    const totalTests = sample.data.extraction?.sections?.flatMap((s) => s.tests)?.length || 0;
    const abnormalCount = (sample.data.extraction?.sections?.flatMap((s) => s.tests) || []).filter(
      (t) => t.status && !t.status.toUpperCase().includes("NORMAL")
    ).length;

    const sampleMsg = {
      sender: "ai",
      text: `### 📋 Loaded Demo Sample: ${sample.name}\n\n**${sample.subtitle}**\n\nTotal Biomarkers Evaluated: **${totalTests}** (${abnormalCount} out of standard range).\n\n${sample.data.interpretation.summary}\n\n*You can now explore the biological range gauges above or ask any clinical questions!*`,
    };

    setThreads((prev) =>
      prev.map((t) =>
        t.id === activeId
          ? {
              ...t,
              title: newTitle,
              attached: true,
              flagged: sample.data.interpretation?.requires_review,
              result: sample.data,
              messages: [sampleMsg],
            }
          : t
      )
    );
  };

  // Handle chat submission
  const handleSendQuestion = async (overrideQuestion = null) => {
    const question = (overrideQuestion || input).trim();
    if (!question || chatLoading) return;

    const userMsg = { sender: "user", text: question };

    setThreads((prev) =>
      prev.map((t) => (t.id === activeId ? { ...t, messages: [...t.messages, userMsg] } : t))
    );

    if (!overrideQuestion) setInput("");
    setChatLoading(true);

    try {
      const payload = {
        question: question,
        history: messages.slice(-6).map((m) => ({ sender: m.sender, text: m.text })),
      };
      if (result) {
        payload.report_data = result;
        if (result.file_id) {
          payload.file_id = result.file_id;
        }
      }

      const response = await axios.post(`${API_URL}/chat`, payload);
      const aiMsg = { sender: "ai", text: response.data.answer || "No response received." };

      setThreads((prev) =>
        prev.map((t) => (t.id === activeId ? { ...t, messages: [...t.messages, aiMsg] } : t))
      );
    } catch (err) {
      console.error("Chat error:", err);
      const detail = err.response?.data?.detail || err.message || "";
      const errorMsg = {
        sender: "ai",
        text: `⚠️ **Clinical Assistant Notice**\n\nUnable to reach the service (${detail || "Network issue"}). Please verify your connection or try again.`,
      };
      setThreads((prev) =>
        prev.map((t) => (t.id === activeId ? { ...t, messages: [...t.messages, errorMsg] } : t))
      );
    } finally {
      setChatLoading(false);
    }
  };

  // Start fresh chat session
  const handleNewChat = () => {
    const newId = `t_${Date.now()}`;
    const newThread = {
      id: newId,
      title: "New Report Analysis",
      group: "Today",
      attached: false,
      flagged: false,
      result: null,
      messages: [],
    };
    setThreads((prev) => [newThread, ...prev]);
    setActiveId(newId);
    setSidebarOpen(false);
  };

  // Delete thread
  const handleDeleteThread = (id, e) => {
    e.stopPropagation();
    let filtered = threads.filter((t) => t.id !== id);
    if (filtered.length === 0) {
      filtered = [{ ...DEFAULT_THREAD, id: `t_${Date.now()}` }];
    }
    setThreads(filtered);
    if (activeId === id) {
      setActiveId(filtered[0].id);
    }
  };

  // Copy response
  const handleCopy = (text, idx) => {
    navigator.clipboard.writeText(text);
    setCopiedIdx(idx);
    setTimeout(() => setCopiedIdx(null), 2000);
  };

  // Biomarker metrics calculations
  const allTests = result?.extraction?.sections?.flatMap((s) => s.tests) || [];
  const normalTests = allTests.filter((t) => (t.status || "").toUpperCase() === "NORMAL");
  const abnormalTests = allTests.filter((t) => t.status && !t.status.toUpperCase().includes("NORMAL"));
  const criticalTests = allTests.filter((t) => (t.status || "").toUpperCase().includes("CRITICAL"));

  const filteredTests = allTests.filter((t) => {
    const name = (t.test_name_normalized || t.test_name_raw || "").toLowerCase();
    const matchesSearch = name.includes(matrixSearch.toLowerCase());
    const statusNorm = (t.status || "").toUpperCase();

    if (matrixFilter === "ABNORMAL") return matchesSearch && !statusNorm.includes("NORMAL");
    if (matrixFilter === "CRITICAL") return matchesSearch && statusNorm.includes("CRITICAL");
    if (matrixFilter === "NORMAL") return matchesSearch && statusNorm === "NORMAL";
    return matchesSearch;
  });

  const patient = result?.extraction?.patient || {};
  const patientNameDisplay =
    patient.name && !["name", "unknown", "patient name"].includes(patient.name.toLowerCase().trim())
      ? patient.name
      : "Pathology Specimen";

  return (
    <div className="gemini-canvas">
      {/* Mobile Backdrop */}
      {sidebarOpen && <div className="sidebar-backdrop" onClick={() => setSidebarOpen(false)} />}

      {/* ============================================================
          SIDEBAR: CLINICAL REPORT HISTORY & SESSIONS
          ============================================================ */}
      <aside
        className={`app-sidebar ${sidebarOpen ? "open" : ""}`}
        style={{
          width: 270,
          flexShrink: 0,
          background: "var(--bg-sidebar)",
          borderRight: "1px solid var(--border-subtle)",
          display: "flex",
          flexDirection: "column",
          height: "100vh",
        }}
      >
        {/* Brand Header */}
        <div style={{ padding: "20px 18px 16px" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 18 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <div
                style={{
                  width: 32,
                  height: 32,
                  borderRadius: 8,
                  background: "linear-gradient(135deg, #0284C7 0%, #38BDF8 100%)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  boxShadow: "0 0 14px rgba(56, 189, 248, 0.3)",
                }}
              >
                <Microscope size={18} color="white" />
              </div>
              <span style={{ fontWeight: 700, fontSize: 17, color: "#FFFFFF", letterSpacing: "-0.02em" }}>
                PathO <span style={{ color: "#38BDF8", fontSize: 13, fontWeight: 500 }}>AI</span>
              </span>
            </div>

            <button
              onClick={() => setSidebarOpen(false)}
              style={{ background: "none", border: "none", color: "#64748B", cursor: "pointer" }}
            >
              <X size={18} />
            </button>
          </div>

          {/* New Chat Button */}
          <button
            onClick={handleNewChat}
            style={{
              width: "100%",
              display: "flex",
              alignItems: "center",
              gap: 8,
              justifyContent: "center",
              padding: "10px 14px",
              background: "rgba(255, 255, 255, 0.06)",
              border: "1px solid var(--border-medium)",
              borderRadius: "var(--radius-capsule)",
              color: "#F1F5F9",
              fontSize: 13,
              fontWeight: 600,
              cursor: "pointer",
              transition: "all 0.15s ease",
            }}
            onMouseEnter={(e) => (e.currentTarget.style.background = "rgba(255, 255, 255, 0.1)")}
            onMouseLeave={(e) => (e.currentTarget.style.background = "rgba(255, 255, 255, 0.06)")}
          >
            <Plus size={16} /> New Chat
          </button>

          {/* Search History */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              marginTop: 12,
              padding: "7px 12px",
              background: "rgba(255, 255, 255, 0.04)",
              borderRadius: "var(--radius-sm)",
              border: "1px solid var(--border-subtle)",
            }}
          >
            <Search size={13} style={{ color: "#64748B" }} />
            <input
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search reports..."
              style={{
                border: "none",
                outline: "none",
                background: "transparent",
                color: "#E2E8F0",
                fontSize: 12,
                width: "100%",
              }}
            />
          </div>
        </div>

        {/* History Threads */}
        <div className="gemini-scroll" style={{ flex: 1, overflowY: "auto", padding: "6px 12px 16px" }}>
          {["Today", "Yesterday", "Earlier"].map((group) => {
            const groupThreads = threads
              .filter((t) => t.title.toLowerCase().includes(searchQuery.toLowerCase()))
              .filter((t) => t.group === group);

            if (!groupThreads.length) return null;

            return (
              <div key={group} style={{ marginBottom: 14 }}>
                <div
                  className="font-mono"
                  style={{
                    fontSize: 10,
                    fontWeight: 600,
                    textTransform: "uppercase",
                    letterSpacing: "0.08em",
                    color: "#64748B",
                    padding: "6px 8px",
                  }}
                >
                  {group}
                </div>
                {groupThreads.map((t) => {
                  const isActive = t.id === activeId;
                  const isEditing = editingId === t.id;

                  return (
                    <div
                      key={t.id}
                      onClick={() => {
                        if (!isEditing) {
                          setActiveId(t.id);
                          setSidebarOpen(false);
                        }
                      }}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 8,
                        padding: "8px 10px",
                        borderRadius: "var(--radius-sm)",
                        background: isActive ? "rgba(56, 189, 248, 0.12)" : "transparent",
                        border: isActive ? "1px solid rgba(56, 189, 248, 0.3)" : "1px solid transparent",
                        cursor: "pointer",
                        marginBottom: 3,
                        transition: "all 0.15s ease",
                      }}
                    >
                      <FileText size={14} style={{ color: isActive ? "#38BDF8" : "#64748B", flexShrink: 0 }} />

                      {isEditing ? (
                        <div style={{ display: "flex", alignItems: "center", gap: 4, flex: 1 }}>
                          <input
                            value={editTitle}
                            onChange={(e) => setEditTitle(e.target.value)}
                            onKeyDown={(e) => {
                              if (e.key === "Enter") {
                                setThreads((prev) =>
                                  prev.map((item) => (item.id === t.id ? { ...item, title: editTitle } : item))
                                );
                                setEditingId(null);
                              }
                            }}
                            autoFocus
                            onClick={(e) => e.stopPropagation()}
                            style={{
                              flex: 1,
                              fontSize: 12,
                              padding: "2px 4px",
                              borderRadius: 4,
                              border: "1px solid #38BDF8",
                              background: "#0F172A",
                              color: "white",
                              outline: "none",
                            }}
                          />
                          <button
                            onClick={(e) => {
                              e.stopPropagation();
                              setThreads((prev) =>
                                prev.map((item) => (item.id === t.id ? { ...item, title: editTitle } : item))
                              );
                              setEditingId(null);
                            }}
                            style={{ background: "none", border: "none", color: "#34D399", cursor: "pointer" }}
                          >
                            <Check size={14} />
                          </button>
                        </div>
                      ) : (
                        <>
                          <span
                            style={{
                              flex: 1,
                              fontSize: 12.5,
                              fontWeight: isActive ? 600 : 400,
                              color: isActive ? "#F8FAFC" : "#94A3B8",
                              whiteSpace: "nowrap",
                              overflow: "hidden",
                              textOverflow: "ellipsis",
                            }}
                          >
                            {t.title}
                          </span>
                          <div style={{ display: "flex", alignItems: "center", gap: 2 }}>
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                setEditingId(t.id);
                                setEditTitle(t.title);
                              }}
                              title="Rename"
                              style={{ background: "none", border: "none", color: "#64748B", cursor: "pointer", padding: 2 }}
                            >
                              <Edit2 size={11} />
                            </button>
                            <button
                              onClick={(e) => handleDeleteThread(t.id, e)}
                              title="Delete"
                              style={{ background: "none", border: "none", color: "#EF4444", cursor: "pointer", padding: 2 }}
                            >
                              <Trash2 size={11} />
                            </button>
                          </div>
                        </>
                      )}
                    </div>
                  );
                })}
              </div>
            );
          })}
        </div>

        {/* Engine Status */}
        <div
          style={{
            padding: "12px 16px",
            borderTop: "1px solid var(--border-subtle)",
            background: "rgba(0,0,0,0.2)",
            display: "flex",
            alignItems: "center",
            gap: 8,
          }}
        >
          <span
            style={{
              width: 7,
              height: 7,
              borderRadius: "50%",
              background: engineInfo?.is_cloud_llm ? "#34D399" : "#38BDF8",
              boxShadow: `0 0 8px ${engineInfo?.is_cloud_llm ? "#34D399" : "#38BDF8"}`,
            }}
          />
          <div style={{ fontSize: 11, color: "#94A3B8" }}>
            Clinical AI Engine Active
          </div>
        </div>
      </aside>

      {/* ============================================================
          MAIN VIEW CONTAINER
          ============================================================ */}
      <main style={{ flex: 1, display: "flex", flexDirection: "column", height: "100vh", overflow: "hidden" }}>
        {/* Sleek Top Navbar */}
        <header className="gemini-header">
          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <button
              onClick={() => setSidebarOpen(true)}
              style={{
                background: "rgba(255, 255, 255, 0.05)",
                border: "1px solid var(--border-subtle)",
                borderRadius: "var(--radius-sm)",
                padding: "7px 9px",
                color: "#94A3B8",
                cursor: "pointer",
                display: "flex",
                alignItems: "center",
              }}
            >
              <Menu size={18} />
            </button>

            <div>
              <span style={{ fontSize: 14, fontWeight: 600, color: "#F8FAFC" }}>
                {activeThread?.title}
              </span>
              {result && (
                <span className="font-mono" style={{ fontSize: 11, color: "#64748B", marginLeft: 8 }}>
                  • {allTests.length} Biomarkers
                </span>
              )}
            </div>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <input
              type="file"
              ref={fileInputRef}
              style={{ display: "none" }}
              accept=".pdf"
              onChange={(e) => handleFileUpload(e.target.files[0])}
            />

            {result && (
              <button
                onClick={() => window.print()}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                  padding: "6px 12px",
                  borderRadius: "var(--radius-capsule)",
                  border: "1px solid var(--border-subtle)",
                  background: "rgba(255, 255, 255, 0.05)",
                  color: "#94A3B8",
                  fontSize: 12,
                  cursor: "pointer",
                }}
              >
                <Download size={13} /> Print Report
              </button>
            )}

            <button
              onClick={handleNewChat}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 6,
                padding: "6px 14px",
                borderRadius: "var(--radius-capsule)",
                border: "1px solid var(--border-medium)",
                background: "rgba(255, 255, 255, 0.08)",
                color: "#F8FAFC",
                fontSize: 12,
                fontWeight: 600,
                cursor: "pointer",
              }}
            >
              <Plus size={13} /> New Chat
            </button>
          </div>
        </header>

        {/* Processing Notification */}
        {loading && (
          <div
            style={{
              padding: "10px 20px",
              background: "linear-gradient(90deg, rgba(2, 132, 199, 0.2) 0%, rgba(56, 189, 248, 0.2) 100%)",
              borderBottom: "1px solid rgba(56, 189, 248, 0.3)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: 10,
              color: "#7DD3FC",
              fontSize: 13,
              fontWeight: 500,
            }}
          >
            <RefreshCw size={15} className="animate-spin" />
            <span>Extracting biomarkers & analyzing pathology PDF with OCR & Clinical AI...</span>
          </div>
        )}

        {/* ============================================================
            WORKSPACE BODY: GEMINI HERO vs CONVERSATIONAL FEED
            ============================================================ */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden", position: "relative" }}>
          {!hasStarted ? (
            /* ============================================================
               1. GEMINI HERO STATE: "Where should we start?"
               ============================================================ */
            <div
              className="gemini-scroll"
              style={{
                flex: 1,
                overflowY: "auto",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                padding: "40px 24px",
              }}
            >
              <div style={{ width: "100%", maxWidth: 800, textAlign: "center" }}>
                {/* Iconic Gemini Heading */}
                <h1
                  style={{
                    fontSize: "2.75rem",
                    fontWeight: 400,
                    color: "#F8FAFC",
                    letterSpacing: "-0.02em",
                    marginBottom: 32,
                  }}
                >
                  Where should we start?
                </h1>

                {/* Gemini Floating Input Capsule */}
                <div className="gemini-capsule-wrapper" style={{ marginBottom: 24 }}>
                  <div className="gemini-capsule">
                    {/* Upload / Paperclip Button */}
                    <div style={{ position: "relative" }}>
                      <button
                        onClick={() => fileInputRef.current?.click()}
                        className="gemini-attach-btn"
                        title="Upload pathology report PDF"
                      >
                        <Paperclip size={18} />
                      </button>
                    </div>

                    {/* Central Textarea */}
                    <textarea
                      ref={textareaRef}
                      className="gemini-input"
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" && !e.shiftKey) {
                          e.preventDefault();
                          handleSendQuestion();
                        }
                      }}
                      placeholder="Ask PathO about your pathology report, biomarkers, or health..."
                      rows={1}
                    />

                    {/* Right Controls: Send Arrow */}
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <button
                        onClick={() => handleSendQuestion()}
                        disabled={!input.trim() || chatLoading}
                        className={`gemini-send-btn ${input.trim() ? "ready" : ""}`}
                      >
                        <ArrowUp size={18} />
                      </button>
                    </div>
                  </div>
                </div>

                {/* Quick Starter Suggestion Pills (Gemini style) */}
                <div
                  style={{
                    display: "flex",
                    flexWrap: "wrap",
                    alignItems: "center",
                    justifyContent: "center",
                    gap: 10,
                    marginBottom: 36,
                  }}
                >
                  <button
                    onClick={() => handleLoadSample(SAMPLE_REPORTS[0])}
                    className="gemini-pill-btn sample"
                    title="Load preloaded abnormal report"
                  >
                    <FlaskConical size={14} />
                    <span>Abnormal CBC Sample (Anemia & Glucose)</span>
                  </button>

                  <button
                    onClick={() => handleLoadSample(SAMPLE_REPORTS[1])}
                    className="gemini-pill-btn sample"
                    title="Load normal wellness screen"
                  >
                    <CheckCircle2 size={14} />
                    <span>Normal Health Screen Sample</span>
                  </button>

                  <button
                    onClick={() =>
                      handleSendQuestion(
                        "What foods and evidence-based dietary strategies help safely optimize vital nutrient levels and energy?"
                      )
                    }
                    className="gemini-pill-btn"
                  >
                    <span>🥗 Evidence-Based Nutrition</span>
                  </button>

                  <button
                    onClick={() =>
                      handleSendQuestion(
                        "Which blood tests require 10-12 hours of overnight fasting, and is drinking water permitted?"
                      )
                    }
                    className="gemini-pill-btn"
                  >
                    <span>🕒 Fasting & Water Rules</span>
                  </button>

                  <button
                    onClick={() =>
                      handleSendQuestion(
                        "What is the clinical difference between a 3-month HbA1c test and a daily Fasting Blood Sugar test?"
                      )
                    }
                    className="gemini-pill-btn"
                  >
                    <span>🩺 HbA1c vs Fasting Glucose</span>
                  </button>

                  <button
                    onClick={() =>
                      handleSendQuestion(
                        "What are the most common laboratory blood test markers doctors check for unexplained fatigue?"
                      )
                    }
                    className="gemini-pill-btn"
                  >
                    <span>⚡ Causes of Fatigue</span>
                  </button>
                </div>

                {/* Security Guarantee Note */}
                <div
                  style={{
                    display: "inline-flex",
                    alignItems: "center",
                    gap: 6,
                    fontSize: 12,
                    color: "#64748B",
                  }}
                >
                  <ShieldCheck size={14} style={{ color: "#34D399" }} />
                  <span>Clinical decision support. Uploaded reports remain private and in-memory.</span>
                </div>
              </div>
            </div>
          ) : (
            /* ============================================================
               2. ACTIVE CONVERSATIONAL FEED + EMBEDDED REPORT DASHBOARD
               ============================================================ */
            <div className="gemini-scroll" style={{ flex: 1, overflowY: "auto", padding: "24px 20px 140px" }}>
              <div style={{ maxWidth: 840, margin: "0 auto", display: "flex", flexDirection: "column", gap: 24 }}>
                {/* EMBEDDED CLINICAL REPORT CARD (IF REPORT LOADED) */}
                {result && (
                  <div className="clinical-report-widget">
                    {/* Patient Demographic Strip */}
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                        flexWrap: "wrap",
                        gap: 14,
                        borderBottom: "1px solid var(--border-subtle)",
                        paddingBottom: 16,
                        marginBottom: 16,
                      }}
                    >
                      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                        <div
                          style={{
                            width: 42,
                            height: 42,
                            borderRadius: "50%",
                            background: "linear-gradient(135deg, #0284C7 0%, #38BDF8 100%)",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            fontSize: 17,
                            fontWeight: 700,
                            color: "white",
                          }}
                        >
                          {patientNameDisplay[0]}
                        </div>
                        <div>
                          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                            <h3 style={{ fontSize: 16, fontWeight: 700, color: "#F8FAFC" }}>{patientNameDisplay}</h3>
                            {result.interpretation?.requires_review && (
                              <span className="status-pill high">Review Required</span>
                            )}
                          </div>
                          <div style={{ fontSize: 12, color: "#94A3B8", marginTop: 2 }}>
                            {patient.age ? `${patient.age} Yrs` : "Age Unspecified"} • {patient.sex || "Sex Unspecified"}{" "}
                            • Lab: {result.extraction?.laboratory_name || "Diagnostic Laboratory"}
                          </div>
                        </div>
                      </div>

                      {patient.patient_id && (
                        <div className="font-mono" style={{ fontSize: 11.5, color: "#64748B" }}>
                          UHID: <strong style={{ color: "#F1F5F9" }}>{patient.patient_id}</strong>
                        </div>
                      )}
                    </div>

                    {/* KPI Summary Grid */}
                    <div
                      style={{
                        display: "grid",
                        gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))",
                        gap: 10,
                        marginBottom: 18,
                      }}
                    >
                      <div
                        style={{
                          background: "rgba(255, 255, 255, 0.03)",
                          padding: "10px 14px",
                          borderRadius: "var(--radius-sm)",
                          border: "1px solid var(--border-subtle)",
                        }}
                      >
                        <div style={{ fontSize: 11, color: "#94A3B8" }}>Total Biomarkers</div>
                        <div style={{ fontSize: 19, fontWeight: 700, color: "#F8FAFC", marginTop: 2 }}>
                          {allTests.length}
                        </div>
                      </div>

                      <div
                        style={{
                          background: "var(--status-normal-bg)",
                          padding: "10px 14px",
                          borderRadius: "var(--radius-sm)",
                          border: "1px solid var(--status-normal-border)",
                        }}
                      >
                        <div style={{ fontSize: 11, color: "var(--status-normal)" }}>In Normal Range</div>
                        <div style={{ fontSize: 19, fontWeight: 700, color: "var(--status-normal)", marginTop: 2 }}>
                          {normalTests.length}
                        </div>
                      </div>

                      <div
                        style={{
                          background: abnormalTests.length > 0 ? "var(--status-high-bg)" : "rgba(255, 255, 255, 0.03)",
                          padding: "10px 14px",
                          borderRadius: "var(--radius-sm)",
                          border:
                            abnormalTests.length > 0
                              ? "1px solid var(--status-high-border)"
                              : "1px solid var(--border-subtle)",
                        }}
                      >
                        <div
                          style={{
                            fontSize: 11,
                            color: abnormalTests.length > 0 ? "var(--status-high)" : "#94A3B8",
                          }}
                        >
                          Out of Bounds
                        </div>
                        <div
                          style={{
                            fontSize: 19,
                            fontWeight: 700,
                            color: abnormalTests.length > 0 ? "var(--status-high)" : "#F8FAFC",
                            marginTop: 2,
                          }}
                        >
                          {abnormalTests.length}
                        </div>
                      </div>

                      <div
                        style={{
                          background:
                            criticalTests.length > 0 ? "var(--status-critical-bg)" : "rgba(255, 255, 255, 0.03)",
                          padding: "10px 14px",
                          borderRadius: "var(--radius-sm)",
                          border:
                            criticalTests.length > 0
                              ? "1px solid var(--status-critical-border)"
                              : "1px solid var(--border-subtle)",
                        }}
                      >
                        <div
                          style={{
                            fontSize: 11,
                            color: criticalTests.length > 0 ? "var(--status-critical)" : "#94A3B8",
                          }}
                        >
                          Critical Flags
                        </div>
                        <div
                          style={{
                            fontSize: 19,
                            fontWeight: 700,
                            color: criticalTests.length > 0 ? "var(--status-critical)" : "#F8FAFC",
                            marginTop: 2,
                          }}
                        >
                          {criticalTests.length}
                        </div>
                      </div>
                    </div>

                    {/* View Switcher: Biomarker Matrix vs Clinical Insights */}
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 8,
                        borderBottom: "1px solid var(--border-subtle)",
                        paddingBottom: 10,
                        marginBottom: 16,
                      }}
                    >
                      <button
                        onClick={() => setActiveTab("matrix")}
                        style={{
                          padding: "6px 14px",
                          borderRadius: "var(--radius-capsule)",
                          border: "none",
                          background: activeTab === "matrix" ? "#38BDF8" : "transparent",
                          color: activeTab === "matrix" ? "#0F172A" : "#94A3B8",
                          fontSize: 12.5,
                          fontWeight: 600,
                          cursor: "pointer",
                        }}
                      >
                        Biomarker Matrix ({allTests.length})
                      </button>

                      <button
                        onClick={() => setActiveTab("insights")}
                        style={{
                          padding: "6px 14px",
                          borderRadius: "var(--radius-capsule)",
                          border: "none",
                          background: activeTab === "insights" ? "#38BDF8" : "transparent",
                          color: activeTab === "insights" ? "#0F172A" : "#94A3B8",
                          fontSize: 12.5,
                          fontWeight: 600,
                          cursor: "pointer",
                        }}
                      >
                        Clinical Monograph & Action Plan
                      </button>
                    </div>

                    {activeTab === "matrix" ? (
                      /* BIOMARKER MATRIX WITH VISUAL RANGE GAUGE */
                      <div>
                        {/* Table Filters */}
                        <div
                          style={{
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "space-between",
                            flexWrap: "wrap",
                            gap: 10,
                            marginBottom: 12,
                          }}
                        >
                          <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                            {[
                              { label: `All (${allTests.length})`, val: "ALL" },
                              { label: `Abnormal (${abnormalTests.length})`, val: "ABNORMAL" },
                              { label: `Critical (${criticalTests.length})`, val: "CRITICAL" },
                              { label: `Normal (${normalTests.length})`, val: "NORMAL" },
                            ].map((f) => (
                              <button
                                key={f.val}
                                onClick={() => setMatrixFilter(f.val)}
                                style={{
                                  padding: "4px 10px",
                                  borderRadius: "var(--radius-capsule)",
                                  border:
                                    matrixFilter === f.val
                                      ? "1px solid #38BDF8"
                                      : "1px solid var(--border-subtle)",
                                  background:
                                    matrixFilter === f.val ? "rgba(56, 189, 248, 0.15)" : "transparent",
                                  color: matrixFilter === f.val ? "#38BDF8" : "#94A3B8",
                                  fontSize: 11.5,
                                  cursor: "pointer",
                                }}
                              >
                                {f.label}
                              </button>
                            ))}
                          </div>

                          <div
                            style={{
                              display: "flex",
                              alignItems: "center",
                              gap: 6,
                              padding: "4px 10px",
                              background: "rgba(255, 255, 255, 0.04)",
                              borderRadius: "var(--radius-sm)",
                              border: "1px solid var(--border-subtle)",
                            }}
                          >
                            <Search size={12} style={{ color: "#64748B" }} />
                            <input
                              value={matrixSearch}
                              onChange={(e) => setMatrixSearch(e.target.value)}
                              placeholder="Filter biomarkers..."
                              style={{
                                border: "none",
                                outline: "none",
                                background: "transparent",
                                fontSize: 11.5,
                                color: "#F8FAFC",
                              }}
                            />
                          </div>
                        </div>

                        {/* Interactive Table with Biological Range Gauges */}
                        <div style={{ overflowX: "auto" }}>
                          <table className="clinical-table-dark">
                            <thead>
                              <tr>
                                <th>Analyte / Biomarker</th>
                                <th>Result</th>
                                <th>Biological Gauge</th>
                                <th>Ref Interval</th>
                                <th>Status</th>
                              </tr>
                            </thead>
                            <tbody>
                              {filteredTests.map((test, idx) => (
                                <tr key={idx}>
                                  <td>
                                    <div style={{ fontWeight: 600, color: "#F8FAFC" }}>
                                      {test.test_name_normalized || test.test_name_raw}
                                    </div>
                                    {test.test_name_normalized &&
                                      test.test_name_raw !== test.test_name_normalized && (
                                        <div style={{ fontSize: 10.5, color: "#64748B", marginTop: 1 }}>
                                          Raw: {test.test_name_raw}
                                        </div>
                                      )}
                                  </td>
                                  <td>
                                    <span style={{ fontWeight: 700, fontSize: "1rem", color: "#F8FAFC" }}>
                                      {test.value_raw}
                                    </span>
                                    <span
                                      className="font-mono"
                                      style={{ fontSize: "0.75rem", color: "#94A3B8", marginLeft: 4 }}
                                    >
                                      {test.unit_normalized || test.unit_raw}
                                    </span>
                                  </td>
                                  <td>
                                    <RangeGauge
                                      value={test.value_numeric || test.value_raw}
                                      min={test.ref_min}
                                      max={test.ref_max}
                                      status={test.status}
                                    />
                                  </td>
                                  <td>
                                    <span className="font-mono" style={{ fontSize: 11.5, color: "#94A3B8" }}>
                                      {test.reference_range_display || test.reference_range_raw || "—"}
                                    </span>
                                  </td>
                                  <td>
                                    <StatusBadge status={test.status} />
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      </div>
                    ) : (
                      /* CLINICAL MONOGRAPH & ACTION PLAN */
                      <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                        <div
                          style={{
                            background: "rgba(56, 189, 248, 0.08)",
                            borderLeft: "3px solid #38BDF8",
                            padding: "12px 16px",
                            borderRadius: "var(--radius-sm)",
                          }}
                        >
                          <div
                            style={{
                              display: "flex",
                              alignItems: "center",
                              gap: 6,
                              fontSize: 12.5,
                              fontWeight: 700,
                              color: "#38BDF8",
                              marginBottom: 4,
                            }}
                          >
                            <FlaskConical size={14} /> Clinical Interpretation
                          </div>
                          <div style={{ fontSize: 13, color: "#E2E8F0", lineHeight: 1.6 }}>
                            {result.interpretation?.clinical_interpretation || result.interpretation?.summary}
                          </div>
                        </div>

                        <div
                          style={{
                            background: "rgba(251, 191, 36, 0.08)",
                            borderLeft: "3px solid #FBBF24",
                            padding: "12px 16px",
                            borderRadius: "var(--radius-sm)",
                          }}
                        >
                          <div
                            style={{
                              display: "flex",
                              alignItems: "center",
                              gap: 6,
                              fontSize: 12.5,
                              fontWeight: 700,
                              color: "#FBBF24",
                              marginBottom: 4,
                            }}
                          >
                            <AlertTriangle size={14} /> Diagnostic & Pattern Impressions
                          </div>
                          <div style={{ fontSize: 13, color: "#E2E8F0", lineHeight: 1.6, whiteSpace: "pre-line" }}>
                            {result.interpretation?.pathologist_remarks || "No abnormal clinical pattern detected."}
                          </div>
                        </div>

                        <div
                          style={{
                            background: "rgba(52, 211, 153, 0.08)",
                            borderLeft: "3px solid #34D399",
                            padding: "12px 16px",
                            borderRadius: "var(--radius-sm)",
                          }}
                        >
                          <div
                            style={{
                              display: "flex",
                              alignItems: "center",
                              gap: 6,
                              fontSize: 12.5,
                              fontWeight: 700,
                              color: "#34D399",
                              marginBottom: 4,
                            }}
                          >
                            <FileCheck2 size={14} /> Suggested Action Plan
                          </div>
                          <div style={{ fontSize: 13, color: "#E2E8F0", lineHeight: 1.6, whiteSpace: "pre-line" }}>
                            {result.interpretation?.suggested_clinical_correlation ||
                              "Maintain routine healthcare follow-up with your primary care physician."}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* CONVERSATION MESSAGE BUBBLES */}
                {messages.map((m, idx) => (
                  <div
                    key={idx}
                    style={{
                      display: "flex",
                      justifyContent: m.sender === "user" ? "flex-end" : "flex-start",
                      gap: 12,
                    }}
                  >
                    {m.sender === "ai" && (
                      <div
                        style={{
                          width: 32,
                          height: 32,
                          borderRadius: "50%",
                          background: "linear-gradient(135deg, #0284C7 0%, #38BDF8 100%)",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          flexShrink: 0,
                          marginTop: 2,
                        }}
                      >
                        <Bot size={16} color="white" />
                      </div>
                    )}

                    <div
                      style={{
                        maxWidth: "88%",
                        borderRadius:
                          m.sender === "user" ? "20px 4px 20px 20px" : "4px 20px 20px 20px",
                        padding: "14px 18px",
                        background: m.sender === "user" ? "#1E293B" : "rgba(19, 29, 49, 0.7)",
                        color: m.sender === "user" ? "#FFFFFF" : "#F8FAFC",
                        border: "1px solid var(--border-medium)",
                        boxShadow: "0 4px 16px rgba(0, 0, 0, 0.2)",
                      }}
                    >
                      {m.sender === "ai" ? (
                        <div className="clinical-markdown">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.text}</ReactMarkdown>
                          <div
                            style={{
                              display: "flex",
                              alignItems: "center",
                              justifyContent: "flex-end",
                              marginTop: 10,
                              paddingTop: 8,
                              borderTop: "1px solid var(--border-subtle)",
                            }}
                          >
                            <button
                              onClick={() => handleCopy(m.text, idx)}
                              style={{
                                display: "flex",
                                alignItems: "center",
                                gap: 4,
                                background: "none",
                                border: "none",
                                fontSize: 11,
                                color: "#94A3B8",
                                cursor: "pointer",
                              }}
                            >
                              {copiedIdx === idx ? (
                                <>
                                  <Check size={12} style={{ color: "#34D399" }} />
                                  <span style={{ color: "#34D399" }}>Copied</span>
                                </>
                              ) : (
                                <>
                                  <Copy size={12} />
                                  <span>Copy</span>
                                </>
                              )}
                            </button>
                          </div>
                        </div>
                      ) : (
                        <div style={{ fontSize: 14, lineHeight: 1.5 }}>{m.text}</div>
                      )}
                    </div>
                  </div>
                ))}

                {/* AI Typing Indicator */}
                {chatLoading && (
                  <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
                    <div
                      style={{
                        width: 32,
                        height: 32,
                        borderRadius: "50%",
                        background: "linear-gradient(135deg, #0284C7 0%, #38BDF8 100%)",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                      }}
                    >
                      <Bot size={16} color="white" />
                    </div>
                    <div
                      style={{
                        background: "rgba(19, 29, 49, 0.7)",
                        border: "1px solid var(--border-medium)",
                        borderRadius: "4px 20px 20px 20px",
                        padding: "12px 18px",
                        display: "flex",
                        alignItems: "center",
                        gap: 8,
                        fontSize: 13,
                        color: "#94A3B8",
                      }}
                    >
                      <RefreshCw size={14} className="animate-spin" />
                      <span>Synthesizing clinical response...</span>
                    </div>
                  </div>
                )}
                <div ref={chatBottomRef} />
              </div>
            </div>
          )}

          {/* ============================================================
              PINNED GEMINI INPUT CAPSULE (ALWAYS ACCESSIBLE AT BOTTOM)
              ============================================================ */}
          {hasStarted && (
            <div
              style={{
                position: "absolute",
                bottom: 0,
                left: 0,
                right: 0,
                padding: "16px 20px 20px",
                background: "linear-gradient(180deg, transparent 0%, rgba(8, 12, 23, 0.95) 40%, #080C17 100%)",
                zIndex: 30,
              }}
            >
              <div className="gemini-capsule-wrapper">
                <div className="gemini-capsule">
                  {/* Upload / Paperclip Button */}
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="gemini-attach-btn"
                    title="Upload pathology report PDF"
                  >
                    <Paperclip size={18} />
                  </button>

                  {/* Central Textarea */}
                  <textarea
                    className="gemini-input"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        handleSendQuestion();
                      }
                    }}
                    placeholder={
                      result
                        ? "Ask PathO about your report results or biomarkers..."
                        : "Ask a medical question or upload a PDF report..."
                    }
                    rows={1}
                  />

                  {/* Right Controls */}
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <button
                      onClick={() => handleSendQuestion()}
                      disabled={!input.trim() || chatLoading}
                      className={`gemini-send-btn ${input.trim() ? "ready" : ""}`}
                    >
                      <ArrowUp size={18} />
                    </button>
                  </div>
                </div>

                <div
                  style={{
                    textAlign: "center",
                    fontSize: 10.5,
                    color: "#64748B",
                    marginTop: 8,
                  }}
                >
                  PathO Clinical Copilot may make mistakes. Review critical medical findings with a qualified doctor.
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
