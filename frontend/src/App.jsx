import React, { useState, useRef, useEffect } from "react";
import axios from "axios";
import {
  Plus, Search, Paperclip, ArrowUp, FileText, Microscope,
  AlertTriangle, CheckCircle2, Info, Sparkles, FlaskConical, Activity,
  FileCheck2, RefreshCw, Trash2, Edit2, Check, X, UploadCloud
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API_URL =
  import.meta.env.VITE_API_BASE_URL ||
  (typeof window !== "undefined" && window.location.port !== "5173"
    ? window.location.origin
    : "http://localhost:8000");

/* ============================================================
   DEFAULT INITIAL THREAD TEMPLATE (FOR FRESH LOCALSTORAGE)
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
      text: "Hello! 👋 Welcome to **PathO**. Upload a pathology report PDF to begin clinical extraction and analysis.",
    },
  ],
};

const loadSavedThreads = () => {
  try {
    const saved = localStorage.getItem("patho_threads");
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
    const saved = localStorage.getItem("patho_active_id");
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
    NORMAL: { label: "NORMAL", class: "patho-badge-normal" },
    HIGH: { label: "HIGH", class: "patho-badge-high" },
    LOW: { label: "LOW", class: "patho-badge-low" },
    BORDERLINE_HIGH: { label: "BORDERLINE HIGH", class: "patho-badge-high" },
    BORDERLINE_LOW: { label: "BORDERLINE LOW", class: "patho-badge-low" },
    CRITICAL: { label: "CRITICAL", class: "patho-badge-critical" },
    CRITICAL_HIGH: { label: "CRITICAL HIGH", class: "patho-badge-critical" },
    CRITICAL_LOW: { label: "CRITICAL LOW", class: "patho-badge-critical" },
  };
  const b = map[normStatus] || { label: normStatus.replace("_", " "), class: "patho-badge-unknown" };

  return <span className={`patho-badge ${b.class}`}>{b.label}</span>;
}

/* ============================================================
   MAIN PATH-O APPLICATION (UNIFIED UI & LOCALSTORAGE PERSISTENCE)
   ============================================================ */

export default function App() {
  const [threads, setThreads] = useState(loadSavedThreads);
  const [activeId, setActiveId] = useState(() => loadSavedActiveId(threads));

  // Editing / Renaming Thread State
  const [editingId, setEditingId] = useState(null);
  const [editTitle, setEditTitle] = useState("");

  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [input, setInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [tableFilter, setTableFilter] = useState("");
  const [engineInfo, setEngineInfo] = useState(null);

  const fileInputRef = useRef(null);
  const chatBottomRef = useRef(null);

  const activeThread = threads.find((t) => t.id === activeId) || threads[0] || DEFAULT_THREAD;
  const result = activeThread?.result;
  const messages = activeThread?.messages || [];

  // Fetch active AI chatbot engine info
  useEffect(() => {
    axios
      .get(`${API_URL}/chat/info`)
      .then((res) => setEngineInfo(res.data))
      .catch((err) => console.log("AI info endpoint:", err));
  }, []);

  // Save state to localStorage whenever threads or activeId changes
  useEffect(() => {
    try {
      localStorage.setItem("patho_threads", JSON.stringify(threads));
      localStorage.setItem("patho_active_id", activeId);
    } catch (e) {
      console.error("Failed to save state to localStorage:", e);
    }
  }, [threads, activeId]);

  // Auto-scroll chat to bottom
  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages.length, chatLoading]);

  // Handle PDF Upload to Backend
  const handleFileUpload = async (uploadedFile) => {
    if (!uploadedFile) return;
    setFile(uploadedFile);
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
          : "Patient Report";

      const category = resData.extraction?.document_category || "Report";
      const newTitle = `${category} — ${patientName}`;

      const welcomeMsg = {
        sender: "ai",
        text: `## 🧪 Report Processed: ${newTitle}\n\n${
          resData.interpretation?.summary || "Report processed successfully."
        }\n\n**Patient:** ${patientName} | **Total Tests:** ${
          resData.extraction?.sections?.flatMap((s) => s.tests)?.length || 0
        }\n\n*Ask me any clinical questions or check the detailed biomarkers & interpretation cards above!*`,
      };

      // Update current active thread with upload result
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
      alert(err.response?.data?.detail || "Failed to process PDF report. Ensure backend is running.");
    } finally {
      setLoading(false);
    }
  };

  // Handle Chat Question Submission
  const handleSendQuestion = async (overrideQuestion = null) => {
    const question = (overrideQuestion || input).trim();
    if (!question || chatLoading) return;

    const userMsg = { sender: "user", text: question };

    // Append user message to active thread
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
      const errorMsg = { sender: "ai", text: "⚠️ Error communicating with PathO assistant server." };
      setThreads((prev) =>
        prev.map((t) => (t.id === activeId ? { ...t, messages: [...t.messages, errorMsg] } : t))
      );
    } finally {
      setChatLoading(false);
    }
  };

  // Start a new chat analysis thread
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
          text: "Hello! 👋 Welcome to **PathO**. Upload a pathology report PDF to begin clinical extraction and analysis.",
        },
      ],
    };
    setThreads((prev) => [newThread, ...prev]);
    setActiveId(newId);
    setFile(null);
  };

  // Delete a chat thread
  const handleDeleteThread = (id, e) => {
    e.stopPropagation();
    let filtered = threads.filter((t) => t.id !== id);
    if (filtered.length === 0) {
      const fresh = {
        ...DEFAULT_THREAD,
        id: `t_${Date.now()}`,
      };
      filtered = [fresh];
    }
    setThreads(filtered);
    if (activeId === id) {
      setActiveId(filtered[0].id);
    }
  };

  // Start renaming thread
  const handleStartRename = (t, e) => {
    e.stopPropagation();
    setEditingId(t.id);
    setEditTitle(t.title);
  };

  // Save renamed thread
  const handleSaveRename = (id, e) => {
    if (e) e.stopPropagation();
    if (editTitle.trim()) {
      setThreads((prev) =>
        prev.map((t) => (t.id === id ? { ...t, title: editTitle.trim() } : t))
      );
    }
    setEditingId(null);
  };

  // Filtered threads for search
  const filteredThreads = threads.filter((t) =>
    t.title.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Extracted Biomarkers Filter
  const allTests = result?.extraction?.sections?.flatMap((s) => s.tests) || [];
  const filteredTests = allTests.filter((t) => {
    const name = (t.test_name_normalized || t.test_name_raw || "").toLowerCase();
    return name.includes(tableFilter.toLowerCase());
  });

  const patient = result?.extraction?.patient || {};
  const patientNameDisplay =
    patient.name && !["name", "unknown", "patient name"].includes(patient.name.toLowerCase().trim())
      ? patient.name
      : "Not Specified";

  return (
    <div style={{ display: "flex", height: "100vh", overflow: "hidden", background: "var(--paper)" }}>
      {/* ============================================================
          SIDEBAR: PATH-O NAVIGATION, SEARCH, RENAME & DELETE CHATS
          ============================================================ */}
      <div
        style={{
          width: 270,
          flexShrink: 0,
          borderRight: "1px solid var(--hairline)",
          background: "var(--surface)",
          display: "flex",
          flexDirection: "column",
          height: "100vh",
        }}
      >
        {/* Brand Header */}
        <div style={{ padding: "16px 16px 12px" }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16 }}>
            <div
              style={{
                width: 32,
                height: 32,
                borderRadius: 9,
                background: "var(--primary)",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <Microscope size={18} color="white" />
            </div>
            <div>
              <span style={{ fontWeight: 800, fontSize: 18, color: "var(--primary-deep)", letterSpacing: "-0.02em" }}>
                PathO
              </span>
              <span className="mono" style={{ fontSize: 10, color: "var(--ink-faint)", marginLeft: 6 }}>
                v2.5
              </span>
            </div>
          </div>

          {/* New Analysis Button */}
          <button
            onClick={handleNewChat}
            style={{
              width: "100%",
              display: "flex",
              alignItems: "center",
              gap: 8,
              justifyContent: "center",
              border: "1px solid var(--hairline-strong)",
              background: "var(--surface)",
              cursor: "pointer",
              padding: "9px 12px",
              borderRadius: "var(--radius-md)",
              fontSize: 13,
              fontWeight: 600,
              color: "var(--ink)",
              transition: "var(--transition-fast)",
            }}
          >
            <Plus size={16} /> New Report Analysis
          </button>

          {/* Thread Search Bar */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              marginTop: 10,
              background: "var(--surface-2)",
              borderRadius: "var(--radius-md)",
              padding: "8px 10px",
              border: "1px solid var(--hairline)",
            }}
          >
            <Search size={14} style={{ color: "var(--ink-faint)" }} />
            <input
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search reports..."
              style={{
                border: "none",
                outline: "none",
                background: "transparent",
                fontSize: 12.5,
                width: "100%",
                color: "var(--ink)",
              }}
            />
          </div>
        </div>

        {/* Thread History Groups */}
        <div className="patho-scroll" style={{ flex: 1, overflowY: "auto", padding: "4px 10px 14px" }}>
          {["Today", "Yesterday", "Earlier"].map((group) => {
            const groupThreads = filteredThreads.filter((t) => t.group === group);
            if (!groupThreads.length) return null;
            return (
              <div key={group} style={{ marginBottom: 12 }}>
                <div
                  className="mono"
                  style={{
                    fontSize: 10,
                    fontWeight: 600,
                    letterSpacing: "0.07em",
                    color: "var(--ink-faint)",
                    padding: "8px 8px 4px",
                    textTransform: "uppercase",
                  }}
                >
                  {group}
                </div>
                {groupThreads.map((t) => {
                  const isEditing = editingId === t.id;
                  const isActive = activeId === t.id;
                  return (
                    <div
                      key={t.id}
                      onClick={() => {
                        if (!isEditing) {
                          setActiveId(t.id);
                        }
                      }}
                      className={`patho-thread ${isActive ? "active" : ""}`}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 8,
                        padding: "8px 10px",
                        borderRadius: "var(--radius-md)",
                        cursor: "pointer",
                        marginBottom: 3,
                        position: "relative",
                      }}
                    >
                      <FileText
                        size={14}
                        style={{ color: isActive ? "var(--primary)" : "var(--ink-faint)", flexShrink: 0 }}
                      />

                      {isEditing ? (
                        <div style={{ display: "flex", alignItems: "center", gap: 4, flex: 1 }}>
                          <input
                            value={editTitle}
                            onChange={(e) => setEditTitle(e.target.value)}
                            onKeyDown={(e) => {
                              if (e.key === "Enter") handleSaveRename(t.id, e);
                              if (e.key === "Escape") setEditingId(null);
                            }}
                            autoFocus
                            onClick={(e) => e.stopPropagation()}
                            style={{
                              flex: 1,
                              fontSize: 12,
                              padding: "2px 4px",
                              borderRadius: 4,
                              border: "1px solid var(--primary)",
                              outline: "none",
                              background: "var(--surface)",
                              color: "var(--ink)",
                            }}
                          />
                          <button
                            onClick={(e) => handleSaveRename(t.id, e)}
                            style={{ border: "none", background: "none", cursor: "pointer", padding: 2 }}
                          >
                            <Check size={14} style={{ color: "var(--normal)" }} />
                          </button>
                        </div>
                      ) : (
                        <>
                          <span
                            style={{
                              fontSize: 12.5,
                              fontWeight: isActive ? 600 : 500,
                              color: isActive ? "var(--primary-deep)" : "var(--ink)",
                              whiteSpace: "nowrap",
                              overflow: "hidden",
                              textOverflow: "ellipsis",
                              flex: 1,
                            }}
                          >
                            {t.title}
                          </span>

                          {/* Rename & Delete Actions */}
                          <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                            <button
                              onClick={(e) => handleStartRename(t, e)}
                              title="Rename Chat"
                              style={{
                                border: "none",
                                background: "none",
                                cursor: "pointer",
                                padding: 3,
                                opacity: 0.6,
                                color: "var(--ink-soft)",
                              }}
                            >
                              <Edit2 size={12} />
                            </button>
                            <button
                              onClick={(e) => handleDeleteThread(t.id, e)}
                              title="Delete Chat"
                              style={{
                                border: "none",
                                background: "none",
                                cursor: "pointer",
                                padding: 3,
                                opacity: 0.6,
                                color: "var(--critical)",
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
      </div>

      {/* ============================================================
          MAIN WORKSPACE: UNIFIED UPLOAD, DASHBOARD & CHATBOT
          ============================================================ */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
        {/* Header Bar */}
        <div
          style={{
            height: 56,
            borderBottom: "1px solid var(--hairline)",
            background: "var(--surface)",
            display: "flex",
            alignItems: "center",
            padding: "0 24px",
            justifyContent: "space-between",
            flexShrink: 0,
          }}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <span style={{ fontWeight: 700, fontSize: 15, color: "var(--ink)" }}>{activeThread?.title}</span>
            {activeThread?.attached && (
              <span
                className="mono"
                style={{
                  fontSize: 10.5,
                  background: "var(--surface-2)",
                  color: "var(--ink-faint)",
                  padding: "3px 8px",
                  borderRadius: 999,
                  display: "flex",
                  alignItems: "center",
                  gap: 4,
                }}
              >
                <FileText size={10} /> PDF Attached
              </span>
            )}
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
            <input
              type="file"
              ref={fileInputRef}
              style={{ display: "none" }}
              accept=".pdf"
              onChange={(e) => handleFileUpload(e.target.files[0])}
            />
          </div>
        </div>

        {/* Loading Indicator */}
        {loading && (
          <div
            style={{
              padding: 16,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: 12,
              background: "var(--primary-tint)",
              color: "var(--primary-deep)",
              fontWeight: 600,
              fontSize: 13.5,
            }}
          >
            <RefreshCw size={18} className="spinner" />
            <span>Processing pathology PDF report with EasyOCR & Clinical AI...</span>
          </div>
        )}

        {/* UNIFIED WORKSPACE CONTENT */}
        <div className="patho-scroll" style={{ flex: 1, overflowY: "auto", padding: 24 }}>
          <div style={{ maxWidth: 840, margin: "0 auto", display: "flex", flexDirection: "column", gap: 24 }}>
            {result && (
              /* POPULATED DASHBOARD CARDS */
              <>
                {/* Patient Profile & Executive Summary Grid */}
                <div style={{ display: "grid", gridTemplateColumns: "280px 1fr", gap: 20 }}>
                  <div className="patho-panel" style={{ padding: 18 }}>
                    <h3
                      className="mono"
                      style={{
                        fontSize: 11,
                        color: "var(--ink-faint)",
                        textTransform: "uppercase",
                        letterSpacing: "0.06em",
                        marginBottom: 14,
                      }}
                    >
                      Patient Demographics
                    </h3>
                    <div style={{ marginBottom: 12 }}>
                      <div style={{ fontSize: 12, color: "var(--ink-soft)" }}>Full Name</div>
                      <div style={{ fontWeight: 700, fontSize: 15, color: "var(--ink)" }}>{patientNameDisplay}</div>
                    </div>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 12 }}>
                      <div>
                        <div style={{ fontSize: 12, color: "var(--ink-soft)" }}>Age</div>
                        <div style={{ fontWeight: 600 }}>{patient.age ? `${patient.age} Yrs` : "Not Specified"}</div>
                      </div>
                      <div>
                        <div style={{ fontSize: 12, color: "var(--ink-soft)" }}>Sex</div>
                        <div style={{ fontWeight: 600 }}>{patient.sex || "Not Specified"}</div>
                      </div>
                    </div>
                    <div>
                      <div style={{ fontSize: 12, color: "var(--ink-soft)" }}>Category</div>
                      <div style={{ fontWeight: 600, color: "var(--primary)" }}>
                        {result.extraction?.document_category || "Laboratory"}
                      </div>
                    </div>
                  </div>

                  <div className="patho-panel" style={{ padding: 18 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
                      <Sparkles size={18} style={{ color: "var(--primary)" }} />
                      <h3 style={{ fontSize: 15, fontWeight: 700, color: "var(--primary-deep)" }}>
                        Executive Summary
                      </h3>
                    </div>
                    <p style={{ fontSize: 13.5, color: "var(--ink)", lineHeight: 1.6 }}>
                      {result.interpretation?.summary || "No summary available."}
                    </p>
                  </div>
                </div>

                {/* 3-Part Clinical Impression Panel */}
                <div className="patho-panel" style={{ padding: 22, background: "var(--surface)" }}>
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      gap: 10,
                      marginBottom: 16,
                      borderBottom: "1px solid var(--hairline)",
                      paddingBottom: 10,
                    }}
                  >
                    <Activity size={20} style={{ color: "var(--primary)" }} />
                    <h3 style={{ fontSize: 16, fontWeight: 700, color: "var(--primary-deep)" }}>
                      Laboratory Remarks & Clinical Interpretation
                    </h3>
                  </div>

                  <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
                    <div
                      style={{
                        background: "var(--surface-2)",
                        padding: "12px 16px",
                        borderRadius: "var(--radius-md)",
                        borderLeft: "4px solid var(--primary)",
                      }}
                    >
                      <div
                        style={{
                          fontWeight: 700,
                          fontSize: 13,
                          color: "var(--primary)",
                          marginBottom: 4,
                          display: "flex",
                          alignItems: "center",
                          gap: 6,
                        }}
                      >
                        <FlaskConical size={15} /> Clinical Interpretation
                      </div>
                      <div style={{ fontSize: 13.5, color: "var(--ink)", lineHeight: 1.6 }}>
                        {result.interpretation?.clinical_interpretation || result.interpretation?.summary}
                      </div>
                    </div>

                    <div
                      style={{
                        background: "var(--surface-2)",
                        padding: "12px 16px",
                        borderRadius: "var(--radius-md)",
                        borderLeft: "4px solid var(--high)",
                      }}
                    >
                      <div
                        style={{
                          fontWeight: 700,
                          fontSize: 13,
                          color: "var(--high)",
                          marginBottom: 4,
                          display: "flex",
                          alignItems: "center",
                          gap: 6,
                        }}
                      >
                        <Info size={15} /> Diagnostic & Pattern Impressions
                      </div>
                      <div style={{ fontSize: 13.5, color: "var(--ink)", lineHeight: 1.6 }}>
                        {result.interpretation?.pathologist_remarks || "No abnormal clinical pattern detected."}
                      </div>
                    </div>

                    <div
                      style={{
                        background: "var(--surface-2)",
                        padding: "12px 16px",
                        borderRadius: "var(--radius-md)",
                        borderLeft: "4px solid var(--normal)",
                      }}
                    >
                      <div
                        style={{
                          fontWeight: 700,
                          fontSize: 13,
                          color: "var(--normal)",
                          marginBottom: 4,
                          display: "flex",
                          alignItems: "center",
                          gap: 6,
                        }}
                      >
                        <FileCheck2 size={15} /> Suggested Clinical Correlation & Action Plan
                      </div>
                      <div style={{ fontSize: 13.5, color: "var(--ink)", lineHeight: 1.6 }}>
                        {result.interpretation?.suggested_clinical_correlation ||
                          "Maintain routine healthcare follow-up with primary care physician."}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Extracted Biomarkers Table */}
                <div className="patho-panel" style={{ padding: 20 }}>
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      marginBottom: 14,
                    }}
                  >
                    <h3 style={{ fontSize: 15, fontWeight: 700, color: "var(--ink)" }}>Extracted Biomarkers</h3>
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 8,
                        background: "var(--surface-2)",
                        padding: "5px 12px",
                        borderRadius: "var(--radius-md)",
                        border: "1px solid var(--hairline)",
                      }}
                    >
                      <Search size={13} style={{ color: "var(--ink-faint)" }} />
                      <input
                        value={tableFilter}
                        onChange={(e) => setTableFilter(e.target.value)}
                        placeholder="Filter analytes..."
                        style={{
                          border: "none",
                          outline: "none",
                          background: "transparent",
                          fontSize: 12,
                          color: "var(--ink)",
                        }}
                      />
                    </div>
                  </div>

                  <div className="patho-table-wrapper">
                    <table className="patho-table">
                      <thead>
                        <tr>
                          <th>Parameter</th>
                          <th>Value</th>
                          <th>Ref Range</th>
                          <th>Status</th>
                        </tr>
                      </thead>
                      <tbody>
                        {filteredTests.map((test, idx) => (
                          <tr key={idx}>
                            <td>
                              <div style={{ fontWeight: 600 }}>
                                {test.test_name_normalized || test.test_name_raw}
                              </div>
                            </td>
                            <td>
                              <span style={{ fontWeight: 700, fontSize: "1.05rem" }}>{test.value_raw}</span>
                              <span style={{ marginLeft: "0.35rem", fontSize: "0.85rem", color: "var(--ink-faint)" }}>
                                {test.unit_normalized || test.unit_raw}
                              </span>
                            </td>
                            <td style={{ fontSize: "0.85rem", color: "var(--ink-soft)" }}>
                              {test.reference_range_display || test.reference_range_raw || "-"}
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
              </>
            )}

            {/* CHAT MESSAGES DISPLAY */}
            <div style={{ display: "flex", flexDirection: "column", gap: 16, marginTop: 12 }}>
              {messages.map((m, idx) => (
                <div
                  key={idx}
                  className="patho-msg"
                  style={{
                    display: "flex",
                    justifyContent: m.sender === "user" ? "flex-end" : "flex-start",
                    gap: 10,
                  }}
                >
                  {m.sender === "ai" && (
                    <div
                      style={{
                        width: 30,
                        height: 30,
                        borderRadius: 8,
                        background: "var(--primary)",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        flexShrink: 0,
                        marginTop: 2,
                      }}
                    >
                      <Microscope size={15} color="white" />
                    </div>
                  )}
                  <div
                    style={{
                      background: m.sender === "user" ? "var(--primary-tint)" : "var(--surface)",
                      color: m.sender === "user" ? "var(--primary-deep)" : "var(--ink)",
                      border: m.sender === "user" ? "none" : "1px solid var(--hairline)",
                      borderRadius:
                        m.sender === "user" ? "16px 4px 16px 16px" : "4px 16px 16px 16px",
                      padding: "12px 16px",
                      maxWidth: 580,
                      fontSize: 13.5,
                      lineHeight: 1.6,
                    }}
                  >
                    {m.sender === "ai" ? (
                      <div className="patho-msg-content">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.text}</ReactMarkdown>
                      </div>
                    ) : (
                      m.text
                    )}
                  </div>
                </div>
              ))}

              {chatLoading && (
                <div className="patho-msg" style={{ display: "flex", gap: 10, alignItems: "center" }}>
                  <div
                    style={{
                      width: 30,
                      height: 30,
                      borderRadius: 8,
                      background: "var(--primary)",
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                    }}
                  >
                    <Microscope size={15} color="white" />
                  </div>
                  <div
                    style={{
                      background: "var(--surface)",
                      border: "1px solid var(--hairline)",
                      borderRadius: "4px 16px 16px 16px",
                      padding: "12px 16px",
                      display: "flex",
                      gap: 4,
                    }}
                  >
                    <span className="patho-dot" style={{ animationDelay: "0s" }} />
                    <span className="patho-dot" style={{ animationDelay: ".15s" }} />
                    <span className="patho-dot" style={{ animationDelay: ".3s" }} />
                  </div>
                </div>
              )}
              <div ref={chatBottomRef} />
            </div>
          </div>
        </div>

        {/* BOTTOM FIXED CHAT INPUT BAR */}
        <div style={{ padding: "12px 24px 20px", background: "var(--paper)", flexShrink: 0 }}>
          <div style={{ maxWidth: 840, margin: "0 auto" }}>
            {/* Prompt Chips */}
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 10 }}>
              {result ? (
                <>
                  <button
                    onClick={() => handleSendQuestion("Summarize my report results.")}
                    className="patho-chip"
                    style={{
                      border: "1px solid var(--hairline-strong)",
                      background: "var(--surface)",
                      color: "var(--ink-soft)",
                      fontSize: 11.5,
                      padding: "6px 12px",
                      borderRadius: 999,
                      cursor: "pointer",
                    }}
                  >
                    ✨ Executive Summary
                  </button>
                  <button
                    onClick={() => handleSendQuestion("Am I healthy?")}
                    className="patho-chip"
                    style={{
                      border: "1px solid var(--hairline-strong)",
                      background: "var(--surface)",
                      color: "var(--ink-soft)",
                      fontSize: 11.5,
                      padding: "6px 12px",
                      borderRadius: 999,
                      cursor: "pointer",
                    }}
                  >
                    🩺 Am I Healthy?
                  </button>
                  <button
                    onClick={() => handleSendQuestion("Which test values are out of range?")}
                    className="patho-chip"
                    style={{
                      border: "1px solid var(--hairline-strong)",
                      background: "var(--surface)",
                      color: "var(--ink-soft)",
                      fontSize: 11.5,
                      padding: "6px 12px",
                      borderRadius: 999,
                      cursor: "pointer",
                    }}
                  >
                    ⚠️ Abnormal Biomarkers
                  </button>
                  <button
                    onClick={() => handleSendQuestion("What diet and lifestyle recommendations should I follow?")}
                    className="patho-chip"
                    style={{
                      border: "1px solid var(--hairline-strong)",
                      background: "var(--surface)",
                      color: "var(--ink-soft)",
                      fontSize: 11.5,
                      padding: "6px 12px",
                      borderRadius: 999,
                      cursor: "pointer",
                    }}
                  >
                    🥗 Diet & Action Plan
                  </button>
                  <button
                    onClick={() => handleSendQuestion("What questions should I ask my doctor?")}
                    className="patho-chip"
                    style={{
                      border: "1px solid var(--hairline-strong)",
                      background: "var(--surface)",
                      color: "var(--ink-soft)",
                      fontSize: 11.5,
                      padding: "6px 12px",
                      borderRadius: 999,
                      cursor: "pointer",
                    }}
                  >
                    📋 Doctor Checklist
                  </button>
                  <button
                    onClick={() => handleSendQuestion("What are the clinical impressions?")}
                    className="patho-chip"
                    style={{
                      border: "1px solid var(--hairline-strong)",
                      background: "var(--surface)",
                      color: "var(--ink-soft)",
                      fontSize: 11.5,
                      padding: "6px 12px",
                      borderRadius: 999,
                      cursor: "pointer",
                    }}
                  >
                    🔬 Diagnostic Impression
                  </button>
                </>
              ) : (
                <>
                  <button
                    onClick={() => handleSendQuestion("What foods help increase hemoglobin and iron levels?")}
                    className="patho-chip"
                    style={{
                      border: "1px solid var(--hairline-strong)",
                      background: "var(--surface)",
                      color: "var(--ink-soft)",
                      fontSize: 11.5,
                      padding: "6px 12px",
                      borderRadius: 999,
                      cursor: "pointer",
                    }}
                  >
                    💡 Boost Hemoglobin
                  </button>
                  <button
                    onClick={() => handleSendQuestion("Explain HDL vs LDL cholesterol and heart risk.")}
                    className="patho-chip"
                    style={{
                      border: "1px solid var(--hairline-strong)",
                      background: "var(--surface)",
                      color: "var(--ink-soft)",
                      fontSize: 11.5,
                      padding: "6px 12px",
                      borderRadius: 999,
                      cursor: "pointer",
                    }}
                  >
                    🩸 Good vs Bad Cholesterol
                  </button>
                  <button
                    onClick={() => handleSendQuestion("Do I need to fast before a blood test? What are the rules?")}
                    className="patho-chip"
                    style={{
                      border: "1px solid var(--hairline-strong)",
                      background: "var(--surface)",
                      color: "var(--ink-soft)",
                      fontSize: 11.5,
                      padding: "6px 12px",
                      borderRadius: 999,
                      cursor: "pointer",
                    }}
                  >
                    🕒 Fasting & Water Rules
                  </button>
                  <button
                    onClick={() => handleSendQuestion("What are the most common causes of chronic fatigue?")}
                    className="patho-chip"
                    style={{
                      border: "1px solid var(--hairline-strong)",
                      background: "var(--surface)",
                      color: "var(--ink-soft)",
                      fontSize: 11.5,
                      padding: "6px 12px",
                      borderRadius: 999,
                      cursor: "pointer",
                    }}
                  >
                    ⚡ Causes of Fatigue
                  </button>
                  <button
                    onClick={() => handleSendQuestion("What is the difference between HbA1c and fasting blood sugar?")}
                    className="patho-chip"
                    style={{
                      border: "1px solid var(--hairline-strong)",
                      background: "var(--surface)",
                      color: "var(--ink-soft)",
                      fontSize: 11.5,
                      padding: "6px 12px",
                      borderRadius: 999,
                      cursor: "pointer",
                    }}
                  >
                    🩺 HbA1c vs Glucose
                  </button>
                </>
              )}
            </div>

            {/* Chat Input Textarea */}
            <div
              style={{
                display: "flex",
                alignItems: "flex-end",
                gap: 8,
                background: "var(--surface)",
                border: "1px solid var(--hairline-strong)",
                borderRadius: 16,
                padding: "8px 8px 8px 14px",
                boxShadow: "0 6px 18px -12px rgba(36,27,52,0.25)",
              }}
            >
              <button
                onClick={() => fileInputRef.current?.click()}
                title="Attach pathology report PDF"
                style={{
                  border: "none",
                  background: "transparent",
                  cursor: "pointer",
                  padding: 6,
                  flexShrink: 0,
                }}
              >
                <Paperclip size={17} style={{ color: "var(--ink-soft)" }} />
              </button>
              <textarea
                className="patho-textarea"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSendQuestion();
                  }
                }}
                placeholder={
                  result ? "Ask PathO about your report results..." : "Ask a medical question or upload a PDF report..."
                }
                rows={1}
                style={{
                  flex: 1,
                  border: "none",
                  outline: "none",
                  resize: "none",
                  fontSize: 13.5,
                  background: "transparent",
                  color: "var(--ink)",
                  maxHeight: 120,
                  padding: "6px 0",
                }}
              />
              <button
                onClick={() => handleSendQuestion()}
                disabled={!input.trim() || chatLoading}
                style={{
                  width: 32,
                  height: 32,
                  borderRadius: 10,
                  border: "none",
                  flexShrink: 0,
                  background: input.trim() ? "var(--primary)" : "var(--surface-2)",
                  cursor: input.trim() ? "pointer" : "default",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                }}
              >
                <ArrowUp size={16} style={{ color: input.trim() ? "white" : "var(--ink-faint)" }} />
              </button>
            </div>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                flexWrap: "wrap",
                gap: 8,
                marginTop: 8,
              }}
            >
              <div className="mono" style={{ fontSize: 10, color: "var(--ink-faint)" }}>
                PathO Clinical Assistant — Grounded AI decision support. Always consult a qualified medical doctor.
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <span
                  style={{
                    width: 7,
                    height: 7,
                    borderRadius: "50%",
                    background: engineInfo?.is_cloud_llm ? "#8b5cf6" : "#10b981",
                    display: "inline-block",
                  }}
                />
                <span className="mono" style={{ fontSize: 10, color: "var(--ink-soft)" }}>
                  {engineInfo ? `${engineInfo.model} (${engineInfo.provider})` : "Clinical Engine Ready"}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
