import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import {
  Plus, Search, Paperclip, ArrowUp, FileText, Microscope,
  AlertTriangle, CheckCircle2, Info, Sparkles, FlaskConical, Activity,
  FileCheck2, RefreshCw, Trash2, Edit2, Check, X, UploadCloud,
  ChevronRight, Copy, Menu, HeartPulse, Stethoscope, ShieldCheck,
  Download, ArrowRight, Filter, Bot, User, Share2
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
   DEFAULT THREAD DEFINITION & LOCALSTORAGE
   ============================================================ */

const DEFAULT_THREAD = {
  id: "t_initial",
  title: "New Report Analysis",
  group: "Today",
  attached: false,
  flagged: false,
  result: null,
  messages: [
    {
      sender: "ai",
      text: "### 👋 Welcome to **PathO Clinical AI**\n\nI can analyze your blood tests, pathology PDF reports, and answer any medical questions grounded in clinical guidelines.\n\n* **Upload a lab report PDF** using the dropzone, or\n* **Try one of our instant sample reports** to test the biomarker matrix, or\n* **Ask any medical question** directly below!",
    },
  ],
};

const loadSavedThreads = () => {
  try {
    const saved = localStorage.getItem("patho_threads_v3");
    if (saved) {
      const parsed = JSON.parse(saved);
      if (Array.isArray(parsed) && parsed.length > 0) return parsed;
    }
  } catch (e) {
    console.error("Failed to load threads from localStorage:", e);
  }
  return [DEFAULT_THREAD];
};

const loadSavedActiveId = (validThreads) => {
  try {
    const saved = localStorage.getItem("patho_active_id_v3");
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
   MAIN PATH-O APPLICATION
   ============================================================ */

export default function App() {
  const [threads, setThreads] = useState(loadSavedThreads);
  const [activeId, setActiveId] = useState(() => loadSavedActiveId(threads));

  // Sidebar & Nav state
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [editingId, setEditingId] = useState(null);
  const [editTitle, setEditTitle] = useState("");

  // Report & File state
  const [loading, setLoading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [activeTab, setActiveTab] = useState("overview"); // "overview" | "biomarkers" | "copilot"
  const [biomarkerFilter, setBiomarkerFilter] = useState("ALL"); // "ALL" | "ABNORMAL" | "CRITICAL" | "NORMAL"
  const [biomarkerSearch, setBiomarkerSearch] = useState("");

  // Chat state
  const [input, setInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [engineInfo, setEngineInfo] = useState(null);
  const [copiedIdx, setCopiedIdx] = useState(null);

  const fileInputRef = useRef(null);
  const chatBottomRef = useRef(null);

  const activeThread = threads.find((t) => t.id === activeId) || threads[0] || DEFAULT_THREAD;
  const result = activeThread?.result;
  const messages = activeThread?.messages || [];

  // Fetch engine info from backend
  useEffect(() => {
    axios
      .get(`${API_URL}/chat/info`)
      .then((res) => setEngineInfo(res.data))
      .catch((err) => console.log("AI info endpoint note:", err.message));
  }, []);

  // Save threads to localStorage
  useEffect(() => {
    try {
      localStorage.setItem("patho_threads_v3", JSON.stringify(threads));
      localStorage.setItem("patho_active_id_v3", activeId);
    } catch (e) {
      console.error("Local storage sync error:", e);
    }
  }, [threads, activeId]);

  // Auto-scroll chat
  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, chatLoading]);

  // Handle PDF upload
  const handleFileUpload = async (uploadedFile) => {
    if (!uploadedFile) return;
    setLoading(true);

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
        text: `### 🧪 Report Processed: ${newTitle}\n\n**${patientName}** | Total Tests Analyzed: **${totalTests}** (${abnormalCount} out of reference bounds).\n\n${
          resData.interpretation?.summary || "Report analyzed successfully."
        }\n\n*Feel free to ask questions like "Explain my abnormal tests", "What diet changes should I make?", or "What should I ask my physician?"*`,
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
      setActiveTab("overview");
    } catch (err) {
      console.error("Upload error:", err);
      alert(err.response?.data?.detail || "Failed to process PDF report. Please verify backend server is online.");
    } finally {
      setLoading(false);
    }
  };

  // Handle loading instant sample report
  const handleLoadSample = (sample) => {
    const newTitle = sample.name;
    const totalTests = sample.data.extraction?.sections?.flatMap((s) => s.tests)?.length || 0;
    const abnormalCount = (sample.data.extraction?.sections?.flatMap((s) => s.tests) || []).filter(
      (t) => t.status && !t.status.toUpperCase().includes("NORMAL")
    ).length;

    const sampleMsg = {
      sender: "ai",
      text: `### 📋 Loaded Sample: ${sample.name}\n\n**${sample.subtitle}**\n\nTotal Biomarkers: **${totalTests}** (${abnormalCount} out of range).\n\n${sample.data.interpretation.summary}\n\n*You can now ask any question regarding this report or review the extracted biomarker matrix.*`,
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
    setActiveTab("overview");
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
      if (result?.file_id) {
        payload.file_id = result.file_id;
      }

      const response = await axios.post(`${API_URL}/chat`, payload);
      const aiMsg = { sender: "ai", text: response.data.answer || "No response received." };

      setThreads((prev) =>
        prev.map((t) => (t.id === activeId ? { ...t, messages: [...t.messages, aiMsg] } : t))
      );
    } catch (err) {
      console.error("Chat error:", err);
      const errorMsg = { sender: "ai", text: "⚠️ Unable to connect to PathO clinical intelligence service." };
      setThreads((prev) =>
        prev.map((t) => (t.id === activeId ? { ...t, messages: [...t.messages, errorMsg] } : t))
      );
    } finally {
      setChatLoading(false);
    }
  };

  // New analysis session
  const handleNewChat = () => {
    const newId = `t_${Date.now()}`;
    const newThread = {
      id: newId,
      title: "New Report Analysis",
      group: "Today",
      attached: false,
      flagged: false,
      result: null,
      messages: [
        {
          sender: "ai",
          text: "### 👋 Welcome to **PathO Clinical AI**\n\nUpload a PDF pathology report or select a sample to begin detailed biomarker extraction, risk stratification, and clinical explanation.",
        },
      ],
    };
    setThreads((prev) => [newThread, ...prev]);
    setActiveId(newId);
    setActiveTab("overview");
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

  // Copy message text
  const handleCopy = (text, idx) => {
    navigator.clipboard.writeText(text);
    setCopiedIdx(idx);
    setTimeout(() => setCopiedIdx(null), 2000);
  };

  // Extracted Biomarker Calculations
  const allTests = result?.extraction?.sections?.flatMap((s) => s.tests) || [];
  const normalTests = allTests.filter((t) => (t.status || "").toUpperCase() === "NORMAL");
  const abnormalTests = allTests.filter((t) => t.status && !t.status.toUpperCase().includes("NORMAL"));
  const criticalTests = allTests.filter((t) => (t.status || "").toUpperCase().includes("CRITICAL"));

  const filteredTests = allTests.filter((t) => {
    const name = (t.test_name_normalized || t.test_name_raw || "").toLowerCase();
    const matchesSearch = name.includes(biomarkerSearch.toLowerCase());

    const statusNorm = (t.status || "").toUpperCase();
    if (biomarkerFilter === "ABNORMAL") return matchesSearch && !statusNorm.includes("NORMAL");
    if (biomarkerFilter === "CRITICAL") return matchesSearch && statusNorm.includes("CRITICAL");
    if (biomarkerFilter === "NORMAL") return matchesSearch && statusNorm === "NORMAL";
    return matchesSearch;
  });

  const patient = result?.extraction?.patient || {};
  const patientNameDisplay =
    patient.name && !["name", "unknown", "patient name"].includes(patient.name.toLowerCase().trim())
      ? patient.name
      : "Patient Specimen";

  return (
    <div className="app-container">
      {/* Mobile Backdrop */}
      {sidebarOpen && <div className="sidebar-backdrop" onClick={() => setSidebarOpen(false)} />}

      {/* ============================================================
          SIDEBAR: CLINICAL HISTORY & NAVIGATION
          ============================================================ */}
      <aside
        className={`app-sidebar ${sidebarOpen ? "open" : ""}`}
        style={{
          width: 280,
          flexShrink: 0,
          background: "var(--bg-sidebar)",
          borderRight: "1px solid var(--border-dark)",
          display: "flex",
          flexDirection: "column",
          height: "100vh",
        }}
      >
        {/* Brand Header */}
        <div style={{ padding: "20px 20px 16px" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 20 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <div
                style={{
                  width: 36,
                  height: 36,
                  borderRadius: "var(--radius-sm)",
                  background: "linear-gradient(135deg, #0284C7 0%, #0369A1 100%)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  boxShadow: "0 4px 12px rgba(2, 132, 199, 0.35)",
                }}
              >
                <Microscope size={20} color="white" />
              </div>
              <div>
                <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                  <span style={{ fontWeight: 800, fontSize: 18, color: "#FFFFFF", letterSpacing: "-0.02em" }}>
                    PathO
                  </span>
                  <span
                    className="font-mono"
                    style={{
                      fontSize: 10,
                      fontWeight: 600,
                      padding: "2px 6px",
                      borderRadius: 4,
                      background: "rgba(2, 132, 199, 0.2)",
                      color: "#38BDF8",
                      border: "1px solid rgba(56, 189, 248, 0.3)",
                    }}
                  >
                    AI v3.0
                  </span>
                </div>
                <div style={{ fontSize: 11, color: "var(--text-faint)" }}>Clinical Intelligence</div>
              </div>
            </div>

            <button
              onClick={() => setSidebarOpen(false)}
              style={{
                display: "none",
                background: "transparent",
                border: "none",
                color: "#94A3B8",
                cursor: "pointer",
              }}
              className="mobile-close-btn"
            >
              <X size={20} />
            </button>
          </div>

          {/* New Analysis Button */}
          <button
            onClick={handleNewChat}
            style={{
              width: "100%",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: 8,
              padding: "11px 14px",
              background: "linear-gradient(135deg, #0284C7 0%, #0369A1 100%)",
              color: "#FFFFFF",
              border: "none",
              borderRadius: "var(--radius-md)",
              fontWeight: 600,
              fontSize: 13,
              cursor: "pointer",
              boxShadow: "0 4px 14px rgba(2, 132, 199, 0.3)",
              transition: "var(--transition-fast)",
            }}
            onMouseEnter={(e) => (e.currentTarget.style.filter = "brightness(1.08)")}
            onMouseLeave={(e) => (e.currentTarget.style.filter = "brightness(1)")}
          >
            <Plus size={16} /> New Report Analysis
          </button>

          {/* Search Input */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              marginTop: 14,
              padding: "8px 12px",
              background: "rgba(255, 255, 255, 0.05)",
              borderRadius: "var(--radius-sm)",
              border: "1px solid var(--border-dark)",
            }}
          >
            <Search size={14} style={{ color: "#64748B" }} />
            <input
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search reports..."
              style={{
                border: "none",
                outline: "none",
                background: "transparent",
                color: "#F1F5F9",
                fontSize: 12.5,
                width: "100%",
              }}
            />
          </div>
        </div>

        {/* History List */}
        <div className="sidebar-scroll" style={{ flex: 1, overflowY: "auto", padding: "8px 14px 16px" }}>
          {["Today", "Yesterday", "Earlier"].map((group) => {
            const groupThreads = threads
              .filter((t) => t.title.toLowerCase().includes(searchQuery.toLowerCase()))
              .filter((t) => t.group === group);

            if (!groupThreads.length) return null;

            return (
              <div key={group} style={{ marginBottom: 16 }}>
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
                        gap: 10,
                        padding: "9px 12px",
                        borderRadius: "var(--radius-sm)",
                        background: isActive ? "rgba(2, 132, 199, 0.15)" : "transparent",
                        border: isActive ? "1px solid rgba(2, 132, 199, 0.4)" : "1px solid transparent",
                        cursor: "pointer",
                        marginBottom: 4,
                        transition: "var(--transition-fast)",
                      }}
                    >
                      <FileText
                        size={15}
                        style={{
                          color: isActive ? "#38BDF8" : "#64748B",
                          flexShrink: 0,
                        }}
                      />

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
                              padding: "2px 6px",
                              borderRadius: 4,
                              border: "1px solid #38BDF8",
                              background: "#1E293B",
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
                            style={{ background: "none", border: "none", color: "#10B981", cursor: "pointer" }}
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
                              color: isActive ? "#F1F5F9" : "#94A3B8",
                              whiteSpace: "nowrap",
                              overflow: "hidden",
                              textOverflow: "ellipsis",
                            }}
                          >
                            {t.title}
                          </span>
                          <div style={{ display: "flex", alignItems: "center", gap: 3 }}>
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                setEditingId(t.id);
                                setEditTitle(t.title);
                              }}
                              title="Rename chat"
                              style={{
                                background: "transparent",
                                border: "none",
                                color: "#64748B",
                                padding: 3,
                                cursor: "pointer",
                              }}
                            >
                              <Edit2 size={12} />
                            </button>
                            <button
                              onClick={(e) => handleDeleteThread(t.id, e)}
                              title="Delete chat"
                              style={{
                                background: "transparent",
                                border: "none",
                                color: "#EF4444",
                                padding: 3,
                                cursor: "pointer",
                              }}
                            >
                              <Trash2 size={12} />
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

        {/* Engine Status Footer */}
        <div
          style={{
            padding: "14px 18px",
            borderTop: "1px solid var(--border-dark)",
            background: "rgba(0, 0, 0, 0.2)",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span
              style={{
                width: 8,
                height: 8,
                borderRadius: "50%",
                background: engineInfo?.is_cloud_llm ? "#10B981" : "#38BDF8",
                boxShadow: `0 0 8px ${engineInfo?.is_cloud_llm ? "#10B981" : "#38BDF8"}`,
              }}
            />
            <div>
              <div style={{ fontSize: 11.5, fontWeight: 600, color: "#E2E8F0" }}>
                {engineInfo?.provider === "gemini" ? "Gemini 2.5 Active" : engineInfo?.provider || "Clinical Engine"}
              </div>
              <div className="font-mono" style={{ fontSize: 10, color: "#64748B" }}>
                {engineInfo?.model || "Offline Monograph Ready"}
              </div>
            </div>
          </div>
          <ShieldCheck size={16} style={{ color: "#38BDF8" }} />
        </div>
      </aside>

      {/* ============================================================
          MAIN WORKSPACE
          ============================================================ */}
      <main style={{ flex: 1, display: "flex", flexDirection: "column", height: "100vh", overflow: "hidden" }}>
        {/* Header Bar */}
        <header className="glass-header" style={{ padding: "12px 24px" }}>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 16 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
              <button
                onClick={() => setSidebarOpen(true)}
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  justifyContent: "center",
                  padding: 8,
                  background: "#F1F5F9",
                  border: "1px solid var(--border-hairline)",
                  borderRadius: "var(--radius-sm)",
                  cursor: "pointer",
                }}
              >
                <Menu size={18} color="#475569" />
              </button>

              <div>
                <h1 style={{ fontSize: 16, fontWeight: 700, color: "var(--text-main)", letterSpacing: "-0.01em" }}>
                  {activeThread?.title}
                </h1>
                {result && (
                  <div style={{ display: "flex", alignItems: "center", gap: 8, marginTop: 2 }}>
                    <span className="font-mono" style={{ fontSize: 11, color: "var(--text-subtle)" }}>
                      Patient: <strong style={{ color: "var(--text-main)" }}>{patientNameDisplay}</strong>
                    </span>
                    <span style={{ color: "var(--border-strong)" }}>•</span>
                    <span className="font-mono" style={{ fontSize: 11, color: "var(--text-subtle)" }}>
                      Total Tests: <strong style={{ color: "var(--text-main)" }}>{allTests.length}</strong>
                    </span>
                  </div>
                )}
              </div>
            </div>

            {/* Quick Action Buttons */}
            <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
              <input
                type="file"
                ref={fileInputRef}
                style={{ display: "none" }}
                accept=".pdf"
                onChange={(e) => handleFileUpload(e.target.files[0])}
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                  padding: "8px 14px",
                  borderRadius: "var(--radius-md)",
                  border: "1px solid var(--border-hairline)",
                  background: "#FFFFFF",
                  fontSize: 12.5,
                  fontWeight: 600,
                  color: "var(--text-main)",
                  cursor: "pointer",
                  boxShadow: "var(--shadow-subtle)",
                }}
              >
                <UploadCloud size={15} style={{ color: "var(--primary)" }} />
                <span>Upload PDF</span>
              </button>

              {result && (
                <button
                  onClick={() => window.print()}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 6,
                    padding: "8px 14px",
                    borderRadius: "var(--radius-md)",
                    border: "1px solid var(--border-hairline)",
                    background: "#FFFFFF",
                    fontSize: 12.5,
                    fontWeight: 600,
                    color: "var(--text-main)",
                    cursor: "pointer",
                    boxShadow: "var(--shadow-subtle)",
                  }}
                >
                  <Download size={14} style={{ color: "var(--text-muted)" }} />
                  <span>Print Report</span>
                </button>
              )}
            </div>
          </div>
        </header>

        {/* Upload Loading Progress Bar */}
        {loading && (
          <div
            style={{
              padding: "12px 24px",
              background: "linear-gradient(90deg, #F0F9FF 0%, #E0F2FE 100%)",
              borderBottom: "1px solid var(--primary-border)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: 12,
              color: "var(--primary-deep)",
              fontWeight: 600,
              fontSize: 13,
            }}
          >
            <RefreshCw size={17} className="animate-spin" />
            <span>Processing pathology PDF report with OCR & Multi-Tier AI Parser...</span>
          </div>
        )}

        {/* ============================================================
            WORKSPACE VIEW: HERO STATE vs POPULATED DASHBOARD
            ============================================================ */}
        <div style={{ flex: 1, display: "flex", overflow: "hidden" }}>
          {!result ? (
            /* ============================================================
               HERO EMPTY STATE WITH DRAG & DROP AND SAMPLE PICKERS
               ============================================================ */
            <div className="custom-scroll" style={{ flex: 1, overflowY: "auto", padding: "32px 24px" }}>
              <div style={{ maxWidth: 880, margin: "0 auto", display: "flex", flexDirection: "column", gap: 32 }}>
                {/* Hero Badge & Title */}
                <div style={{ textAlign: "center", paddingTop: 12 }}>
                  <div
                    style={{
                      display: "inline-flex",
                      alignItems: "center",
                      gap: 6,
                      padding: "5px 14px",
                      borderRadius: "var(--radius-pill)",
                      background: "var(--primary-subtle)",
                      border: "1px solid var(--primary-border)",
                      color: "var(--primary-deep)",
                      fontSize: 12,
                      fontWeight: 600,
                      marginBottom: 16,
                    }}
                  >
                    <Sparkles size={14} style={{ color: "var(--primary)" }} />
                    <span>Next-Generation Clinical Pathology AI</span>
                  </div>
                  <h2
                    style={{
                      fontSize: 32,
                      fontWeight: 800,
                      color: "var(--text-main)",
                      letterSpacing: "-0.03em",
                      lineHeight: 1.25,
                      marginBottom: 10,
                    }}
                  >
                    Precision Lab Report Intelligence
                  </h2>
                  <p style={{ fontSize: 15, color: "var(--text-muted)", maxWidth: 600, margin: "0 auto" }}>
                    Instantly extract 60+ biomarkers, evaluate biological intervals, and receive grounded clinical
                    explanations powered by Google Gemini.
                  </p>
                </div>

                {/* Instant Medical Question & Prompt Bar in Hero */}
                <div
                  className="card-base"
                  style={{
                    padding: "18px 20px",
                    background: "#FFFFFF",
                    display: "flex",
                    flexDirection: "column",
                    gap: 12,
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", flexWrap: "wrap", gap: 8 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <Bot size={18} style={{ color: "var(--ai)" }} />
                      <span style={{ fontSize: 13, fontWeight: 700, color: "var(--text-main)" }}>
                        Ask PathO Clinical AI (No PDF required)
                      </span>
                    </div>
                    <span className="font-mono" style={{ fontSize: 11, color: "var(--text-subtle)" }}>
                      {engineInfo?.provider === "gemini" ? "✨ Powered by Gemini 2.5 Flash" : "Clinical Engine Ready"}
                    </span>
                  </div>

                  {/* Suggestion Chips */}
                  <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                    {[
                      { label: "💡 Boost Low Hemoglobin", q: "What foods and supplements help safely increase low hemoglobin and iron levels?" },
                      { label: "🩸 Good vs Bad Cholesterol", q: "Explain the clinical difference between HDL and LDL cholesterol and their heart disease risks." },
                      { label: "🕒 Fasting & Water Rules", q: "Which blood tests require 10-12 hours of overnight fasting, and is drinking water permitted?" },
                      { label: "⚡ Causes of Fatigue", q: "What are the most common laboratory blood test markers doctors check for unexplained fatigue?" },
                      { label: "🩺 HbA1c vs Glucose", q: "What is the clinical difference between a 3-month HbA1c test and a daily Fasting Blood Sugar test?" }
                    ].map((chip, idx) => (
                      <button key={idx} onClick={() => handleSendQuestion(chip.q)} className="chip-btn">
                        {chip.label}
                      </button>
                    ))}
                  </div>

                  {/* Input Row */}
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 8,
                      background: "var(--bg-app)",
                      border: "1px solid var(--border-hairline)",
                      borderRadius: "var(--radius-md)",
                      padding: "6px 8px 6px 14px",
                    }}
                  >
                    <input
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" && !e.shiftKey) {
                          e.preventDefault();
                          handleSendQuestion();
                        }
                      }}
                      placeholder="Ask any pathology, biomarker, diet or lab test question..."
                      style={{
                        flex: 1,
                        border: "none",
                        outline: "none",
                        background: "transparent",
                        fontSize: 13,
                        color: "var(--text-main)",
                      }}
                    />
                    <button
                      onClick={() => handleSendQuestion()}
                      disabled={!input.trim() || chatLoading}
                      style={{
                        padding: "7px 14px",
                        borderRadius: "var(--radius-sm)",
                        border: "none",
                        background: input.trim() ? "var(--primary)" : "#E2E8F0",
                        color: input.trim() ? "white" : "#94A3B8",
                        fontWeight: 600,
                        fontSize: 12,
                        cursor: input.trim() ? "pointer" : "default",
                        display: "flex",
                        alignItems: "center",
                        gap: 6,
                        transition: "var(--transition-fast)",
                      }}
                    >
                      <span>Ask AI</span>
                      <ArrowUp size={14} />
                    </button>
                  </div>
                </div>

                {/* Live Conversation in Hero if user asked questions without PDF */}
                {messages.length > 1 && (
                  <div className="card-base" style={{ padding: "20px 24px" }}>
                    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <Bot size={18} style={{ color: "var(--ai)" }} />
                        <h4 style={{ fontSize: 15, fontWeight: 700, color: "var(--text-main)" }}>Clinical Q&A Thread</h4>
                      </div>
                      <button
                        onClick={() => {
                          setThreads((prev) =>
                            prev.map((t) => (t.id === activeId ? { ...t, messages: [DEFAULT_THREAD.messages[0]] } : t))
                          );
                        }}
                        style={{
                          background: "none",
                          border: "none",
                          color: "var(--text-subtle)",
                          fontSize: 11.5,
                          cursor: "pointer",
                        }}
                      >
                        Reset Thread
                      </button>
                    </div>

                    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
                      {messages.map((m, idx) => (
                        <div
                          key={idx}
                          style={{
                            display: "flex",
                            justifyContent: m.sender === "user" ? "flex-end" : "flex-start",
                            gap: 10,
                          }}
                        >
                          {m.sender === "ai" && (
                            <div
                              style={{
                                width: 28,
                                height: 28,
                                borderRadius: "var(--radius-xs)",
                                background: "linear-gradient(135deg, #6366F1 0%, #4F46E5 100%)",
                                display: "flex",
                                alignItems: "center",
                                justifyContent: "center",
                                flexShrink: 0,
                                marginTop: 2,
                              }}
                            >
                              <Bot size={15} color="white" />
                            </div>
                          )}

                          <div
                            style={{
                              maxWidth: "85%",
                              borderRadius:
                                m.sender === "user" ? "16px 4px 16px 16px" : "4px 16px 16px 16px",
                              padding: "12px 16px",
                              background: m.sender === "user" ? "var(--primary)" : "var(--bg-app)",
                              color: m.sender === "user" ? "#FFFFFF" : "var(--text-main)",
                              border: m.sender === "user" ? "none" : "1px solid var(--border-hairline)",
                              boxShadow: "var(--shadow-subtle)",
                            }}
                          >
                            {m.sender === "ai" ? (
                              <div className="clinical-markdown">
                                <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.text}</ReactMarkdown>
                              </div>
                            ) : (
                              <div style={{ fontSize: 13.5, lineHeight: 1.5 }}>{m.text}</div>
                            )}
                          </div>
                        </div>
                      ))}

                      {chatLoading && (
                        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
                          <div
                            style={{
                              width: 28,
                              height: 28,
                              borderRadius: "var(--radius-xs)",
                              background: "linear-gradient(135deg, #6366F1 0%, #4F46E5 100%)",
                              display: "flex",
                              alignItems: "center",
                              justifyContent: "center",
                            }}
                          >
                            <Bot size={15} color="white" />
                          </div>
                          <div
                            style={{
                              background: "var(--bg-app)",
                              border: "1px solid var(--border-hairline)",
                              borderRadius: "4px 16px 16px 16px",
                              padding: "10px 16px",
                              display: "flex",
                              alignItems: "center",
                              gap: 6,
                              fontSize: 12.5,
                              color: "var(--text-muted)",
                            }}
                          >
                            <RefreshCw size={13} className="animate-spin" />
                            <span>Synthesizing clinical response...</span>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Drag & Drop Zone */}
                <div
                  className={`dropzone-box ${dragOver ? "drag-over" : ""}`}
                  onDragOver={(e) => {
                    e.preventDefault();
                    setDragOver(true);
                  }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={(e) => {
                    e.preventDefault();
                    setDragOver(false);
                    if (e.dataTransfer.files?.[0]) {
                      handleFileUpload(e.dataTransfer.files[0]);
                    }
                  }}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <div
                    style={{
                      width: 56,
                      height: 56,
                      borderRadius: "50%",
                      background: "rgba(2, 132, 199, 0.12)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      margin: "0 auto 16px",
                    }}
                  >
                    <UploadCloud size={28} style={{ color: "var(--primary)" }} />
                  </div>
                  <h3 style={{ fontSize: 16, fontWeight: 700, color: "var(--text-main)", marginBottom: 6 }}>
                    Drag & Drop your Pathology PDF here
                  </h3>
                  <p style={{ fontSize: 13, color: "var(--text-muted)", marginBottom: 16 }}>
                    Supports CBC, CMP, Lipid Profile, Thyroid, Renal & Liver panels up to 25MB
                  </p>
                  <button
                    style={{
                      padding: "8px 18px",
                      background: "var(--primary)",
                      color: "white",
                      border: "none",
                      borderRadius: "var(--radius-md)",
                      fontWeight: 600,
                      fontSize: 13,
                      cursor: "pointer",
                      boxShadow: "0 4px 12px rgba(2, 132, 199, 0.25)",
                    }}
                  >
                    Browse Files
                  </button>
                  <div
                    style={{
                      marginTop: 20,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      gap: 6,
                      fontSize: 11.5,
                      color: "var(--text-faint)",
                    }}
                  >
                    <ShieldCheck size={14} style={{ color: "var(--normal)" }} />
                    <span>In-memory processing. Your clinical records remain private and secure.</span>
                  </div>
                </div>

                {/* Quick Interactive Sample Selectors */}
                <div>
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 8,
                      marginBottom: 14,
                      fontSize: 13,
                      fontWeight: 700,
                      color: "var(--text-subtle)",
                      textTransform: "uppercase",
                      letterSpacing: "0.05em",
                    }}
                  >
                    <FlaskConical size={16} /> Instant Demo Samples (Click to Load):
                  </div>

                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(360px, 1fr))", gap: 16 }}>
                    {SAMPLE_REPORTS.map((sample) => (
                      <div
                        key={sample.id}
                        onClick={() => handleLoadSample(sample)}
                        className="card-base card-interactive"
                        style={{
                          padding: "18px 20px",
                          cursor: "pointer",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "space-between",
                          gap: 16,
                        }}
                      >
                        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
                          <div
                            style={{
                              width: 42,
                              height: 42,
                              borderRadius: "var(--radius-md)",
                              background: sample.badgeType === "warning" ? "var(--high-bg)" : "var(--normal-bg)",
                              display: "flex",
                              alignItems: "center",
                              justifyContent: "center",
                              flexShrink: 0,
                            }}
                          >
                            {sample.badgeType === "warning" ? (
                              <AlertTriangle size={20} style={{ color: "var(--high)" }} />
                            ) : (
                              <CheckCircle2 size={20} style={{ color: "var(--normal)" }} />
                            )}
                          </div>
                          <div>
                            <div style={{ fontWeight: 700, fontSize: 14.5, color: "var(--text-main)" }}>
                              {sample.name}
                            </div>
                            <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 2 }}>
                              {sample.subtitle}
                            </div>
                          </div>
                        </div>
                        <ArrowRight size={18} style={{ color: "var(--text-faint)" }} />
                      </div>
                    ))}
                  </div>
                </div>

                {/* Feature Highlights Bento Grid */}
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: 16 }}>
                  <div className="card-base" style={{ padding: 20 }}>
                    <div style={{ color: "var(--primary)", marginBottom: 12 }}>
                      <Activity size={24} />
                    </div>
                    <h4 style={{ fontSize: 14, fontWeight: 700, color: "var(--text-main)", marginBottom: 6 }}>
                      Multi-Tier Biomarker Engine
                    </h4>
                    <p style={{ fontSize: 12.5, color: "var(--text-muted)", lineHeight: 1.5 }}>
                      Combines PyMuPDF digital text extraction, EasyOCR for scanned pages, and normalized schema mapping
                      across 60+ parameters.
                    </p>
                  </div>

                  <div className="card-base" style={{ padding: 20 }}>
                    <div style={{ color: "var(--ai)", marginBottom: 12 }}>
                      <Sparkles size={24} />
                    </div>
                    <h4 style={{ fontSize: 14, fontWeight: 700, color: "var(--text-main)", marginBottom: 6 }}>
                      Clinical Decision Synthesis
                    </h4>
                    <p style={{ fontSize: 12.5, color: "var(--text-muted)", lineHeight: 1.5 }}>
                      Identifies multi-parameter syndromic patterns including Anemia, Impaired Glycemic Regulation, and
                      Renal clearance deviations.
                    </p>
                  </div>

                  <div className="card-base" style={{ padding: 20 }}>
                    <div style={{ color: "var(--normal)", marginBottom: 12 }}>
                      <Stethoscope size={24} />
                    </div>
                    <h4 style={{ fontSize: 14, fontWeight: 700, color: "var(--text-main)", marginBottom: 6 }}>
                      Grounded AI Copilot
                    </h4>
                    <p style={{ fontSize: 12.5, color: "var(--text-muted)", lineHeight: 1.5 }}>
                      Ask questions, generate diet plans, prepare doctor consultation checklists, and understand test
                      fasting rules.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            /* ============================================================
               POPULATED DASHBOARD: SPLIT SCREEN (REPORT + AI COPILOT)
               ============================================================ */
            <div className="workspace-split">
              {/* Left Section: Clinical Report Data & Biomarker Matrix */}
              <div className="split-left-pane custom-scroll" style={{ padding: "24px 28px" }}>
                <div style={{ display: "flex", flexDirection: "column", gap: 24, maxWidth: 960 }}>
                  {/* Patient Banner & Key Metrics Card */}
                  <div className="card-base" style={{ padding: "20px 24px" }}>
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "space-between",
                        flexWrap: "wrap",
                        gap: 16,
                        borderBottom: "1px solid var(--border-hairline)",
                        paddingBottom: 16,
                        marginBottom: 16,
                      }}
                    >
                      <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
                        <div
                          style={{
                            width: 44,
                            height: 44,
                            borderRadius: "50%",
                            background: "linear-gradient(135deg, #0284C7 0%, #0369A1 100%)",
                            color: "white",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            fontSize: 18,
                            fontWeight: 700,
                          }}
                        >
                          {patientNameDisplay[0]}
                        </div>
                        <div>
                          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                            <h3 style={{ fontSize: 17, fontWeight: 800, color: "var(--text-main)" }}>
                              {patientNameDisplay}
                            </h3>
                            {result.interpretation?.requires_review && (
                              <span className="status-pill high">Review Required</span>
                            )}
                          </div>
                          <div style={{ fontSize: 12.5, color: "var(--text-muted)", marginTop: 2 }}>
                            {patient.age ? `${patient.age} Yrs` : "Age Unspecified"} • {patient.sex || "Sex Unspecified"}{" "}
                            • Lab: {result.extraction?.laboratory_name || "Diagnostic Laboratory"}
                          </div>
                        </div>
                      </div>

                      {patient.patient_id && (
                        <div className="font-mono" style={{ fontSize: 11.5, color: "var(--text-subtle)" }}>
                          UHID: <strong style={{ color: "var(--text-main)" }}>{patient.patient_id}</strong>
                        </div>
                      )}
                    </div>

                    {/* Quick Metric KPI Badges */}
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))", gap: 12 }}>
                      <div
                        style={{
                          background: "var(--bg-app)",
                          padding: "12px 14px",
                          borderRadius: "var(--radius-md)",
                          border: "1px solid var(--border-hairline)",
                        }}
                      >
                        <div style={{ fontSize: 11, color: "var(--text-subtle)", fontWeight: 600 }}>Total Tests</div>
                        <div style={{ fontSize: 20, fontWeight: 800, color: "var(--text-main)", marginTop: 2 }}>
                          {allTests.length}
                        </div>
                      </div>

                      <div
                        style={{
                          background: "var(--normal-bg)",
                          padding: "12px 14px",
                          borderRadius: "var(--radius-md)",
                          border: "1px solid var(--normal-border)",
                        }}
                      >
                        <div style={{ fontSize: 11, color: "var(--normal)", fontWeight: 600 }}>In Normal Range</div>
                        <div style={{ fontSize: 20, fontWeight: 800, color: "var(--normal)", marginTop: 2 }}>
                          {normalTests.length}
                        </div>
                      </div>

                      <div
                        style={{
                          background: abnormalTests.length > 0 ? "var(--high-bg)" : "var(--bg-app)",
                          padding: "12px 14px",
                          borderRadius: "var(--radius-md)",
                          border:
                            abnormalTests.length > 0
                              ? "1px solid var(--high-border)"
                              : "1px solid var(--border-hairline)",
                        }}
                      >
                        <div
                          style={{
                            fontSize: 11,
                            color: abnormalTests.length > 0 ? "var(--high)" : "var(--text-subtle)",
                            fontWeight: 600,
                          }}
                        >
                          Out of Bounds
                        </div>
                        <div
                          style={{
                            fontSize: 20,
                            fontWeight: 800,
                            color: abnormalTests.length > 0 ? "var(--high)" : "var(--text-main)",
                            marginTop: 2,
                          }}
                        >
                          {abnormalTests.length}
                        </div>
                      </div>

                      <div
                        style={{
                          background: criticalTests.length > 0 ? "var(--critical-bg)" : "var(--bg-app)",
                          padding: "12px 14px",
                          borderRadius: "var(--radius-md)",
                          border:
                            criticalTests.length > 0
                              ? "1px solid var(--critical-border)"
                              : "1px solid var(--border-hairline)",
                        }}
                      >
                        <div
                          style={{
                            fontSize: 11,
                            color: criticalTests.length > 0 ? "var(--critical)" : "var(--text-subtle)",
                            fontWeight: 600,
                          }}
                        >
                          Critical Flags
                        </div>
                        <div
                          style={{
                            fontSize: 20,
                            fontWeight: 800,
                            color: criticalTests.length > 0 ? "var(--critical)" : "var(--text-main)",
                            marginTop: 2,
                          }}
                        >
                          {criticalTests.length}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Navigation View Switcher (Overview vs Biomarkers) */}
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 8,
                      borderBottom: "1px solid var(--border-hairline)",
                      paddingBottom: 8,
                    }}
                  >
                    <button
                      onClick={() => setActiveTab("overview")}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 6,
                        padding: "8px 16px",
                        borderRadius: "var(--radius-md)",
                        border: "none",
                        background: activeTab === "overview" ? "var(--primary)" : "transparent",
                        color: activeTab === "overview" ? "#FFFFFF" : "var(--text-muted)",
                        fontWeight: 600,
                        fontSize: 13,
                        cursor: "pointer",
                      }}
                    >
                      <Sparkles size={15} /> Clinical Synthesis
                    </button>
                    <button
                      onClick={() => setActiveTab("biomarkers")}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 6,
                        padding: "8px 16px",
                        borderRadius: "var(--radius-md)",
                        border: "none",
                        background: activeTab === "biomarkers" ? "var(--primary)" : "transparent",
                        color: activeTab === "biomarkers" ? "#FFFFFF" : "var(--text-muted)",
                        fontWeight: 600,
                        fontSize: 13,
                        cursor: "pointer",
                      }}
                    >
                      <Activity size={15} /> Biomarker Matrix ({allTests.length})
                    </button>
                  </div>

                  {activeTab === "overview" ? (
                    /* OVERVIEW & MONOGRAPH TAB */
                    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
                      {/* Executive Summary Card */}
                      <div className="card-base" style={{ padding: "20px 24px" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 12 }}>
                          <div
                            style={{
                              width: 32,
                              height: 32,
                              borderRadius: "var(--radius-sm)",
                              background: "var(--primary-subtle)",
                              display: "flex",
                              alignItems: "center",
                              justifyContent: "center",
                            }}
                          >
                            <Stethoscope size={18} style={{ color: "var(--primary)" }} />
                          </div>
                          <h4 style={{ fontSize: 16, fontWeight: 700, color: "var(--text-main)" }}>
                            Executive Clinical Synthesis
                          </h4>
                        </div>
                        <p style={{ fontSize: 14, color: "var(--text-muted)", lineHeight: 1.65 }}>
                          {result.interpretation?.summary || "No automated summary available."}
                        </p>
                      </div>

                      {/* 3-Part Clinical Impression Blocks */}
                      <div className="card-base" style={{ padding: "22px 24px" }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
                          <Activity size={19} style={{ color: "var(--primary)" }} />
                          <h4 style={{ fontSize: 16, fontWeight: 700, color: "var(--text-main)" }}>
                            Physiological Findings & Action Monograph
                          </h4>
                        </div>

                        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                          {/* Clinical Interpretation */}
                          <div
                            style={{
                              background: "var(--primary-subtle)",
                              borderLeft: "4px solid var(--primary)",
                              padding: "14px 18px",
                              borderRadius: "var(--radius-sm)",
                            }}
                          >
                            <div
                              style={{
                                display: "flex",
                                alignItems: "center",
                                gap: 6,
                                fontWeight: 700,
                                fontSize: 13,
                                color: "var(--primary-deep)",
                                marginBottom: 4,
                              }}
                            >
                              <FlaskConical size={15} /> Clinical Interpretation
                            </div>
                            <div style={{ fontSize: 13.5, color: "var(--text-main)", lineHeight: 1.6 }}>
                              {result.interpretation?.clinical_interpretation || result.interpretation?.summary}
                            </div>
                          </div>

                          {/* Pattern Impressions */}
                          <div
                            style={{
                              background: "var(--high-bg)",
                              borderLeft: "4px solid var(--high)",
                              padding: "14px 18px",
                              borderRadius: "var(--radius-sm)",
                            }}
                          >
                            <div
                              style={{
                                display: "flex",
                                alignItems: "center",
                                gap: 6,
                                fontWeight: 700,
                                fontSize: 13,
                                color: "var(--high)",
                                marginBottom: 4,
                              }}
                            >
                              <AlertTriangle size={15} /> Diagnostic & Pattern Impressions
                            </div>
                            <div style={{ fontSize: 13.5, color: "var(--text-main)", lineHeight: 1.6, whiteSpace: "pre-line" }}>
                              {result.interpretation?.pathologist_remarks || "No abnormal clinical pattern detected."}
                            </div>
                          </div>

                          {/* Suggested Action Plan */}
                          <div
                            style={{
                              background: "var(--normal-bg)",
                              borderLeft: "4px solid var(--normal)",
                              padding: "14px 18px",
                              borderRadius: "var(--radius-sm)",
                            }}
                          >
                            <div
                              style={{
                                display: "flex",
                                alignItems: "center",
                                gap: 6,
                                fontWeight: 700,
                                fontSize: 13,
                                color: "var(--normal)",
                                marginBottom: 4,
                              }}
                            >
                              <FileCheck2 size={15} /> Suggested Clinical Correlation & Action Plan
                            </div>
                            <div style={{ fontSize: 13.5, color: "var(--text-main)", lineHeight: 1.6, whiteSpace: "pre-line" }}>
                              {result.interpretation?.suggested_clinical_correlation ||
                                "Maintain routine healthcare follow-up with your primary physician."}
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  ) : (
                    /* BIOMARKER MATRIX TAB */
                    <div className="card-base" style={{ padding: "20px 24px" }}>
                      {/* Search & Filter Header */}
                      <div
                        style={{
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "space-between",
                          flexWrap: "wrap",
                          gap: 12,
                          marginBottom: 16,
                        }}
                      >
                        {/* Filter Status Pills */}
                        <div style={{ display: "flex", alignItems: "center", gap: 6, flexWrap: "wrap" }}>
                          {[
                            { label: `All (${allTests.length})`, val: "ALL" },
                            { label: `Abnormal (${abnormalTests.length})`, val: "ABNORMAL" },
                            { label: `Critical (${criticalTests.length})`, val: "CRITICAL" },
                            { label: `Normal (${normalTests.length})`, val: "NORMAL" },
                          ].map((f) => (
                            <button
                              key={f.val}
                              onClick={() => setBiomarkerFilter(f.val)}
                              style={{
                                padding: "5px 12px",
                                borderRadius: "var(--radius-pill)",
                                border:
                                  biomarkerFilter === f.val
                                    ? "1px solid var(--primary)"
                                    : "1px solid var(--border-hairline)",
                                background: biomarkerFilter === f.val ? "var(--primary-subtle)" : "#FFFFFF",
                                color: biomarkerFilter === f.val ? "var(--primary-deep)" : "var(--text-muted)",
                                fontSize: 12,
                                fontWeight: 600,
                                cursor: "pointer",
                              }}
                            >
                              {f.label}
                            </button>
                          ))}
                        </div>

                        {/* Search Input */}
                        <div
                          style={{
                            display: "flex",
                            alignItems: "center",
                            gap: 8,
                            padding: "6px 12px",
                            background: "var(--bg-app)",
                            borderRadius: "var(--radius-md)",
                            border: "1px solid var(--border-hairline)",
                          }}
                        >
                          <Search size={14} style={{ color: "var(--text-subtle)" }} />
                          <input
                            value={biomarkerSearch}
                            onChange={(e) => setBiomarkerSearch(e.target.value)}
                            placeholder="Filter biomarkers..."
                            style={{
                              border: "none",
                              outline: "none",
                              background: "transparent",
                              fontSize: 12.5,
                              color: "var(--text-main)",
                            }}
                          />
                        </div>
                      </div>

                      {/* Clinical Table */}
                      <div className="clinical-table-wrapper">
                        <table className="clinical-table">
                          <thead>
                            <tr>
                              <th>Analyte / Biomarker</th>
                              <th>Result Value</th>
                              <th>Biological Range Gauge</th>
                              <th>Ref Interval</th>
                              <th>Clinical Status</th>
                            </tr>
                          </thead>
                          <tbody>
                            {filteredTests.map((test, idx) => (
                              <tr key={idx}>
                                <td>
                                  <div style={{ fontWeight: 600, color: "var(--text-main)" }}>
                                    {test.test_name_normalized || test.test_name_raw}
                                  </div>
                                  {test.test_name_normalized && test.test_name_raw !== test.test_name_normalized && (
                                    <div style={{ fontSize: 11, color: "var(--text-faint)", marginTop: 2 }}>
                                      Raw: {test.test_name_raw}
                                    </div>
                                  )}
                                </td>
                                <td>
                                  <span style={{ fontWeight: 800, fontSize: "1.05rem", color: "var(--text-main)" }}>
                                    {test.value_raw}
                                  </span>
                                  <span
                                    className="font-mono"
                                    style={{
                                      fontSize: "0.8rem",
                                      color: "var(--text-subtle)",
                                      marginLeft: 5,
                                    }}
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
                                  <span className="font-mono" style={{ fontSize: 12, color: "var(--text-muted)" }}>
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
                  )}
                </div>
              </div>

              {/* Right Section: AI Clinical Copilot (Persistent Chatbot) */}
              <div className="split-right-pane">
                {/* Copilot Header */}
                <div
                  style={{
                    padding: "16px 20px",
                    borderBottom: "1px solid var(--border-hairline)",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                  }}
                >
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <div
                      style={{
                        width: 32,
                        height: 32,
                        borderRadius: "var(--radius-sm)",
                        background: "linear-gradient(135deg, #6366F1 0%, #4F46E5 100%)",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                      }}
                    >
                      <Bot size={18} color="white" />
                    </div>
                    <div>
                      <div style={{ fontWeight: 700, fontSize: 14, color: "var(--text-main)" }}>
                        AI Clinical Copilot
                      </div>
                      <div style={{ fontSize: 11, color: "var(--text-subtle)" }}>
                        {engineInfo?.is_cloud_llm ? "Grounded Gemini 2.5 Flash" : "Grounded Medical Engine"}
                      </div>
                    </div>
                  </div>

                  <span
                    className="font-mono"
                    style={{
                      fontSize: 10.5,
                      padding: "3px 8px",
                      borderRadius: "var(--radius-pill)",
                      background: "var(--ai-subtle)",
                      color: "var(--ai)",
                      fontWeight: 600,
                    }}
                  >
                    Active
                  </span>
                </div>

                {/* Messages Container */}
                <div className="custom-scroll" style={{ flex: 1, overflowY: "auto", padding: "20px 20px" }}>
                  <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
                    {messages.map((m, idx) => (
                      <div
                        key={idx}
                        style={{
                          display: "flex",
                          justifyContent: m.sender === "user" ? "flex-end" : "flex-start",
                          gap: 10,
                        }}
                      >
                        {m.sender === "ai" && (
                          <div
                            style={{
                              width: 28,
                              height: 28,
                              borderRadius: "var(--radius-xs)",
                              background: "linear-gradient(135deg, #6366F1 0%, #4F46E5 100%)",
                              display: "flex",
                              alignItems: "center",
                              justifyContent: "center",
                              flexShrink: 0,
                              marginTop: 2,
                            }}
                          >
                            <Bot size={15} color="white" />
                          </div>
                        )}

                        <div
                          style={{
                            maxWidth: "85%",
                            borderRadius:
                              m.sender === "user" ? "16px 4px 16px 16px" : "4px 16px 16px 16px",
                            padding: "12px 16px",
                            background: m.sender === "user" ? "var(--primary)" : "var(--bg-app)",
                            color: m.sender === "user" ? "#FFFFFF" : "var(--text-main)",
                            border: m.sender === "user" ? "none" : "1px solid var(--border-hairline)",
                            boxShadow: "var(--shadow-subtle)",
                            position: "relative",
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
                                  marginTop: 8,
                                  paddingTop: 6,
                                  borderTop: "1px solid var(--border-hairline)",
                                }}
                              >
                                <button
                                  onClick={() => handleCopy(m.text, idx)}
                                  style={{
                                    display: "flex",
                                    alignItems: "center",
                                    gap: 4,
                                    background: "transparent",
                                    border: "none",
                                    fontSize: 11,
                                    color: "var(--text-subtle)",
                                    cursor: "pointer",
                                  }}
                                >
                                  {copiedIdx === idx ? (
                                    <>
                                      <Check size={12} style={{ color: "var(--normal)" }} />
                                      <span style={{ color: "var(--normal)" }}>Copied</span>
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
                            <div style={{ fontSize: 13.5, lineHeight: 1.5 }}>{m.text}</div>
                          )}
                        </div>
                      </div>
                    ))}

                    {chatLoading && (
                      <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
                        <div
                          style={{
                            width: 28,
                            height: 28,
                            borderRadius: "var(--radius-xs)",
                            background: "linear-gradient(135deg, #6366F1 0%, #4F46E5 100%)",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                          }}
                        >
                          <Bot size={15} color="white" />
                        </div>
                        <div
                          style={{
                            background: "var(--bg-app)",
                            border: "1px solid var(--border-hairline)",
                            borderRadius: "4px 16px 16px 16px",
                            padding: "10px 16px",
                            display: "flex",
                            alignItems: "center",
                            gap: 6,
                            fontSize: 12.5,
                            color: "var(--text-muted)",
                          }}
                        >
                          <RefreshCw size={13} className="animate-spin" />
                          <span>Synthesizing clinical response...</span>
                        </div>
                      </div>
                    )}
                    <div ref={chatBottomRef} />
                  </div>
                </div>

                {/* Prompt Suggestion Chips */}
                <div style={{ padding: "8px 18px", borderTop: "1px solid var(--border-hairline)", background: "#FAFCFE" }}>
                  <div
                    className="custom-scroll"
                    style={{ display: "flex", gap: 6, overflowX: "auto", paddingBottom: 4 }}
                  >
                    {[
                      { label: "📋 Doctor Questions", q: "What specific questions should I ask my doctor about this report?" },
                      { label: "⚠️ Explain Abnormal", q: "Which test values are out of range and what do they indicate?" },
                      { label: "🥗 Diet & Lifestyle", q: "What specific diet and lifestyle changes are recommended based on these results?" },
                      { label: "🩺 Diagnostic Impressions", q: "What potential clinical conditions are implied by these findings?" },
                    ].map((chip, idx) => (
                      <button key={idx} onClick={() => handleSendQuestion(chip.q)} className="chip-btn">
                        {chip.label}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Chat Input Bar */}
                <div style={{ padding: "12px 18px 16px", background: "#FFFFFF", borderTop: "1px solid var(--border-hairline)" }}>
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 8,
                      background: "var(--bg-app)",
                      border: "1px solid var(--border-hairline)",
                      borderRadius: "var(--radius-lg)",
                      padding: "8px 10px 8px 14px",
                      boxShadow: "var(--shadow-subtle)",
                    }}
                  >
                    <textarea
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" && !e.shiftKey) {
                          e.preventDefault();
                          handleSendQuestion();
                        }
                      }}
                      placeholder="Ask PathO about your biomarkers or report..."
                      rows={1}
                      style={{
                        flex: 1,
                        border: "none",
                        outline: "none",
                        background: "transparent",
                        fontSize: 13,
                        color: "var(--text-main)",
                        resize: "none",
                      }}
                    />
                    <button
                      onClick={() => handleSendQuestion()}
                      disabled={!input.trim() || chatLoading}
                      style={{
                        width: 32,
                        height: 32,
                        borderRadius: "var(--radius-sm)",
                        border: "none",
                        background: input.trim() ? "var(--primary)" : "#E2E8F0",
                        color: input.trim() ? "white" : "#94A3B8",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        cursor: input.trim() ? "pointer" : "default",
                        transition: "var(--transition-fast)",
                      }}
                    >
                      <ArrowUp size={16} />
                    </button>
                  </div>
                  <div
                    style={{
                      fontSize: 10,
                      color: "var(--text-faint)",
                      textAlign: "center",
                      marginTop: 6,
                    }}
                  >
                    Investigational clinical support. Always review with a licensed medical professional.
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
