"use client";

// Streaming chat UI: connects to POST /chat/{workflow}/stream (SSE over a
// POST, parsed by hand in lib/chat/api.ts's streamChat -- native
// EventSource is GET-only). Ticket 09 (clarification interrupt()/resume)
// and tickets 11/12 (per-message 4-axis feedback) extend this. Ticket 10
// added the history sidebar (past threads, resuming a transcript, a
// separate "New conversation" action -- backend/api/routes/chat.py's
// stream_chat keeps chat_threads up to date as a side effect of every
// turn). This pass adds the third pane: clicking a citation chip opens the
// cited document's chunks on the right, scrolled to and highlighting the
// exact one the answer drew from (GET /documents/{id}/chunks) -- "the
// paragraph", in the sense of backend/ingestion/chunking.py's chunk
// boundaries, which is the finest-grained unit actually stored; there's no
// PDF page/position tracked to scroll a rendered original to instead.
import { useEffect, useRef, useState, type FormEvent } from "react";
import Link from "next/link";
import { getChatThread, listChatThreads, streamChat } from "../../lib/chat/api";
import { useAuth } from "../../lib/auth-context";
import type { ChatCitation, ChatMessage, ChatThread } from "../../lib/chat/types";
import { getDocumentChunks } from "../../lib/documents/api";
import type { DocumentChunk } from "../../lib/documents/types";

const STATUS_LABELS: Record<string, string> = {
  resolve_synonyms: "Checking terminology…",
  hybrid_retrieve: "Searching documents…",
  rerank: "Ranking results…",
  generate: "Writing answer…",
  self_eval: "Checking answer quality…",
  reformulate: "Refining search…",
  finalize: "Finishing up…",
};

function uid(): string {
  return Math.random().toString(36).slice(2);
}

function formatThreadDate(iso: string): string {
  const date = new Date(iso);
  const today = new Date();
  const sameDay =
    date.getFullYear() === today.getFullYear() &&
    date.getMonth() === today.getMonth() &&
    date.getDate() === today.getDate();
  return sameDay
    ? date.toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" })
    : date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

export default function ChatPage() {
  const { user, loading } = useAuth();
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [threads, setThreads] = useState<ChatThread[]>([]);
  const [activeThreadId, setActiveThreadId] = useState<string | undefined>(undefined);
  const [threadLoading, setThreadLoading] = useState(false);

  const [openCitation, setOpenCitation] = useState<ChatCitation | null>(null);
  const [docChunks, setDocChunks] = useState<DocumentChunk[] | null>(null);
  const [docPaneLoading, setDocPaneLoading] = useState(false);
  const [docPaneError, setDocPaneError] = useState<string | null>(null);
  const loadedDocumentIdRef = useRef<string | null>(null);
  const activeChunkRef = useRef<HTMLDivElement>(null);

  const threadIdRef = useRef<string | undefined>(undefined);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (!user) return;
    listChatThreads()
      .then(setThreads)
      .catch(() => setError("Couldn't load your past conversations"));
  }, [user]);

  useEffect(() => {
    activeChunkRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [openCitation, docChunks]);

  function startNewConversation() {
    if (sending) return;
    threadIdRef.current = undefined;
    setActiveThreadId(undefined);
    setMessages([]);
    setError(null);
    setInput("");
    closeCitation();
  }

  async function selectThread(threadId: string) {
    if (sending || threadId === activeThreadId) return;
    setThreadLoading(true);
    setError(null);
    closeCitation();
    try {
      const transcript = await getChatThread(threadId);
      threadIdRef.current = threadId;
      setActiveThreadId(threadId);
      setMessages(
        transcript.messages.map((m) => ({
          id: uid(),
          role: m.role,
          content: m.content,
          citations: m.citations,
        }))
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Couldn't load that conversation");
    } finally {
      setThreadLoading(false);
    }
  }

  function closeCitation() {
    setOpenCitation(null);
    setDocChunks(null);
    setDocPaneError(null);
    loadedDocumentIdRef.current = null;
  }

  async function openCitationPane(citation: ChatCitation) {
    setOpenCitation(citation);
    if (loadedDocumentIdRef.current === citation.document_id) return; // already loaded, just re-scroll/highlight
    setDocPaneLoading(true);
    setDocPaneError(null);
    try {
      const chunks = await getDocumentChunks(citation.document_id);
      loadedDocumentIdRef.current = citation.document_id;
      setDocChunks(chunks);
    } catch (err) {
      setDocChunks(null);
      setDocPaneError(err instanceof Error ? err.message : "Couldn't load that document");
    } finally {
      setDocPaneLoading(false);
    }
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    const text = input.trim();
    if (!text || sending) return;

    const userMessage: ChatMessage = { id: uid(), role: "user", content: text };
    const assistantId = uid();
    setMessages((prev) => [
      ...prev,
      userMessage,
      { id: assistantId, role: "assistant", content: "", pending: true, status: "Thinking…" },
    ]);
    setInput("");
    setSending(true);
    setError(null);

    function updateAssistant(patch: Partial<ChatMessage>) {
      setMessages((prev) =>
        prev.map((m) => (m.id === assistantId ? { ...m, ...patch } : m))
      );
    }

    try {
      for await (const evt of streamChat(text, threadIdRef.current)) {
        if (evt.event === "thread") {
          threadIdRef.current = evt.data.thread_id;
          setActiveThreadId(evt.data.thread_id);
        } else if (evt.event === "status") {
          updateAssistant({ status: STATUS_LABELS[evt.data.node] ?? evt.data.node });
        } else if (evt.event === "token") {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId
                ? { ...m, content: m.content + evt.data.content, status: undefined }
                : m
            )
          );
        } else if (evt.event === "done") {
          updateAssistant({
            content: evt.data.answer,
            citations: evt.data.citations,
            status: undefined,
            pending: false,
          });
        } else if (evt.event === "error") {
          updateAssistant({ pending: false, status: undefined });
          setError(evt.data.message);
        }
      }
    } catch (err) {
      updateAssistant({ pending: false, status: undefined });
      setError(err instanceof Error ? err.message : "Something went wrong");
    } finally {
      setSending(false);
      // Picks up the new/renamed/reordered thread this turn just created or
      // touched -- cheap enough to just refetch rather than reconcile by hand.
      listChatThreads()
        .then(setThreads)
        .catch(() => {});
    }
  }

  if (loading) return <main className="page"><p>Loading…</p></main>;

  if (!user) {
    return (
      <main className="page">
        <h1>Chat</h1>
        <p>
          <Link href="/login">Log in</Link> to ask OrthoMate a question.
        </p>
      </main>
    );
  }

  return (
    <main className="page-full">
      <span className="eyebrow">Ask OrthoMate</span>
      <h1>Chat</h1>
      <p className="lede">
        Ask about product specs, compatibility, or procedure requirements. Answers cite the
        source documents they&rsquo;re drawn from -- click a citation to open it.
      </p>

      <div className="chat-layout">
        <aside className="chat-sidebar">
          <button
            type="button"
            className="btn btn-ghost chat-new-thread"
            onClick={startNewConversation}
            disabled={sending}
          >
            + New conversation
          </button>
          <div className="chat-thread-list">
            {threads.length === 0 && (
              <div className="chat-thread-list-empty">No past conversations yet.</div>
            )}
            {threads.map((t) => (
              <button
                key={t.thread_id}
                type="button"
                className={`chat-thread-item${
                  t.thread_id === activeThreadId ? " chat-thread-item-active" : ""
                }`}
                onClick={() => selectThread(t.thread_id)}
                disabled={sending}
              >
                <span className="chat-thread-item-title">{t.title}</span>
                <span className="chat-thread-item-date">{formatThreadDate(t.updated_at)}</span>
              </button>
            ))}
          </div>
        </aside>

        <div className="chat-main">
          <div className="chat-thread">
            {threadLoading && <div className="empty-state">Loading conversation…</div>}
            {!threadLoading && messages.length === 0 && (
              <div className="empty-state">Ask a question about a product system to get started.</div>
            )}
            {!threadLoading &&
              messages.map((m) => (
                <div key={m.id} className={`chat-bubble chat-bubble-${m.role}`}>
                  {m.content && <div className="chat-bubble-text">{m.content}</div>}
                  {m.status && <div className="chat-status">{m.status}</div>}
                  {m.citations && m.citations.length > 0 && (
                    <div className="chat-citations">
                      {m.citations.map((c, i) => (
                        <button
                          key={`${c.document_id}#${c.chunk_index}-${i}`}
                          type="button"
                          className={`chat-citation${
                            openCitation?.document_id === c.document_id &&
                            openCitation?.chunk_index === c.chunk_index
                              ? " chat-citation-active"
                              : ""
                          }`}
                          onClick={() => openCitationPane(c)}
                        >
                          {c.filename}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            <div ref={bottomRef} />
          </div>

          {error && <p className="alert" role="alert">{error}</p>}

          <form onSubmit={handleSubmit} className="chat-input-row">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask a question…"
              disabled={sending}
              autoFocus
            />
            <button type="submit" className="btn btn-primary" disabled={sending || !input.trim()}>
              {sending ? "Sending…" : "Send"}
            </button>
          </form>
        </div>

        {openCitation && (
          <aside className="chat-doc-pane">
            <div className="chat-doc-pane-header">
              <span className="chat-doc-pane-title">{openCitation.filename}</span>
              <button
                type="button"
                className="chat-doc-pane-close"
                onClick={closeCitation}
                aria-label="Close document viewer"
              >
                ×
              </button>
            </div>
            {docPaneLoading && <div className="empty-state">Loading document…</div>}
            {docPaneError && <p className="alert" role="alert">{docPaneError}</p>}
            {docChunks && (
              <div className="chat-doc-pane-body">
                {docChunks.map((chunk) => {
                  const isActive = chunk.chunk_index === openCitation.chunk_index;
                  return (
                    <div
                      key={chunk.chunk_index}
                      ref={isActive ? activeChunkRef : undefined}
                      className={`chat-doc-chunk${isActive ? " chat-doc-chunk-active" : ""}`}
                    >
                      {chunk.section_title && (
                        <div className="chat-doc-chunk-section">{chunk.section_title}</div>
                      )}
                      <div className="chat-doc-chunk-text">{chunk.content}</div>
                    </div>
                  );
                })}
              </div>
            )}
          </aside>
        )}
      </div>
    </main>
  );
}
