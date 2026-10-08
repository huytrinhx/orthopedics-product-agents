"use client";

// Streaming chat UI: connects to POST /chat/stream (SSE over a POST,
// parsed by hand in lib/chat/api.ts -- native EventSource is GET-only).
// Tickets 11/12 (per-message 4-axis feedback) extend this. Ticket 10 added
// the history sidebar (past threads, resuming a transcript, a separate
// "New conversation" action). A citation chip opens the cited document's
// chunks on the right, scrolled to and highlighting the exact one the
// answer drew from (GET /documents/{id}/chunks) -- "the paragraph", in the
// sense of backend/ingestion/chunking.py's chunk boundaries, which is the
// finest-grained unit actually stored; there's no PDF page/position
// tracked to scroll a rendered original to instead.
//
// Ticket 09 (clarification interrupt()/resume): a "clarification" SSE
// event (consumeStream below) means detect_intent couldn't confidently
// tell which product system the query is about -- pendingClarification
// tracks the paused thread_id, and the next submit (a suggested option's
// button, or free text typed into the same input) calls resumeChat instead
// of starting a new streamChat turn. See answerClarification.
//
// `/chat?ask=<question>` (the Eval tab's Rerun) sends that question as the
// first turn of a fresh conversation on load, then strips the param so a
// refresh doesn't ask it again. Read from window.location rather than
// useSearchParams, which a static export would force behind a Suspense
// boundary for no benefit here.
import { useEffect, useRef, useState, type FormEvent } from "react";
import Link from "next/link";
import ReactMarkdown from "react-markdown";
import { getChatThread, listChatThreads, resumeChat, streamChat } from "../../lib/chat/api";
import { useAuth } from "../../lib/auth-context";
import { STATUS_LABELS, citationLabel, stripCitationMarkers } from "../../lib/chat/format";
import type { ChatCitation, ChatMessage, ChatStreamEvent, ChatThread } from "../../lib/chat/types";
import { getDocumentChunks, getDocumentFile } from "../../lib/documents/api";
import type { DocumentChunk } from "../../lib/documents/types";
import { MessageFeedback } from "./message-feedback";
import { PdfCitationViewer } from "./pdf-citation-viewer";

function isPdfFilename(filename: string): boolean {
  return filename.toLowerCase().endsWith(".pdf");
}

function uniqueId(): string {
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

  // Ticket 09: set while a clarification question is waiting on an answer --
  // routes the next submit (button click or free-text) to resumeChat against
  // this thread instead of starting a fresh streamChat turn.
  const [pendingClarification, setPendingClarification] = useState<string | null>(null);

  const [openCitation, setOpenCitation] = useState<ChatCitation | null>(null);
  const [documentChunks, setDocumentChunks] = useState<DocumentChunk[] | null>(null);
  const [documentPaneLoading, setDocumentPaneLoading] = useState(false);
  const [documentPaneError, setDocumentPaneError] = useState<string | null>(null);
  const loadedDocumentIdRef = useRef<string | null>(null);
  const activeChunkRef = useRef<HTMLDivElement>(null);

  // Ticket 26: the PDF file itself, fetched separately from its chunks
  // (an authenticated Blob fetch, not a bare <iframe src> -- see
  // lib/documents/api.ts's getDocumentFile) and only for .pdf citations.
  const [documentFileData, setDocumentFileData] = useState<{ data: Uint8Array } | null>(null);
  const [documentFileError, setDocumentFileError] = useState<string | null>(null);
  const loadedFileDocumentIdRef = useRef<string | null>(null);

  const threadIdRef = useRef<string | undefined>(undefined);
  // Guards the `?ask=` auto-send against running twice (React strict mode
  // double-invokes effects in dev).
  const askedFromUrlRef = useRef(false);
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
    if (!user || askedFromUrlRef.current) return;
    if (!user.is_admin && !user.is_active) return;
    const question = new URLSearchParams(window.location.search).get("ask")?.trim();
    if (!question) return;
    askedFromUrlRef.current = true;
    window.history.replaceState(null, "", window.location.pathname);
    sendQuestion(question);
    // sendQuestion is deliberately not a dependency -- this only ever runs
    // once, on the first render with a user.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user]);

  useEffect(() => {
    activeChunkRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [openCitation, documentChunks]);

  function startNewConversation() {
    if (sending) return;
    threadIdRef.current = undefined;
    setActiveThreadId(undefined);
    setMessages([]);
    setError(null);
    setInput("");
    setPendingClarification(null);
    closeCitation();
  }

  async function selectThread(threadId: string) {
    if (sending || threadId === activeThreadId) return;
    setThreadLoading(true);
    setError(null);
    setPendingClarification(null);
    closeCitation();
    try {
      const transcript = await getChatThread(threadId);
      threadIdRef.current = threadId;
      setActiveThreadId(threadId);
      setMessages(
        transcript.messages.map((message) => ({
          id: uniqueId(),
          role: message.role,
          content: message.content,
          citations: message.citations,
          messageId: message.message_id,
          feedback: message.feedback,
        }))
      );
    } catch (caughtError) {
      setError(caughtError instanceof Error ? caughtError.message : "Couldn't load that conversation");
    } finally {
      setThreadLoading(false);
    }
  }

  function closeCitation() {
    setOpenCitation(null);
    setDocumentChunks(null);
    setDocumentPaneError(null);
    loadedDocumentIdRef.current = null;
    setDocumentFileData(null);
    setDocumentFileError(null);
    loadedFileDocumentIdRef.current = null;
  }

  async function openCitationPane(citation: ChatCitation) {
    setOpenCitation(citation);
    const chunksAlreadyLoaded = loadedDocumentIdRef.current === citation.document_id;
    const needsFile =
      isPdfFilename(citation.filename) && loadedFileDocumentIdRef.current !== citation.document_id;

    if (!chunksAlreadyLoaded) {
      setDocumentPaneLoading(true);
      setDocumentPaneError(null);
    }
    if (needsFile) {
      setDocumentFileData(null);
      setDocumentFileError(null);
    }

    const chunksPromise = chunksAlreadyLoaded
      ? Promise.resolve(null)
      : getDocumentChunks(citation.document_id)
          .then((chunks) => {
            loadedDocumentIdRef.current = citation.document_id;
            setDocumentChunks(chunks);
          })
          .catch((caughtError) => {
            setDocumentChunks(null);
            setDocumentPaneError(caughtError instanceof Error ? caughtError.message : "Couldn't load that document");
          })
          .finally(() => setDocumentPaneLoading(false));

    const filePromise = needsFile
      ? getDocumentFile(citation.document_id)
          .then((blob) => blob.arrayBuffer())
          .then((buffer) => {
            loadedFileDocumentIdRef.current = citation.document_id;
            setDocumentFileData({ data: new Uint8Array(buffer) });
          })
          .catch((caughtError) => {
            setDocumentFileError(caughtError instanceof Error ? caughtError.message : "Couldn't load the PDF");
          })
      : Promise.resolve(null);

    await Promise.all([chunksPromise, filePromise]);
  }

  function updateMessage(id: string, patch: Partial<ChatMessage>) {
    setMessages((previous) => previous.map((message) => (message.id === id ? { ...message, ...patch } : message)));
  }

  // Shared by a fresh turn (streamChat) and a clarification answer
  // (resumeChat, ticket 09) -- both yield the same ChatStreamEvent shape
  // (backend/api/routes/chat.py's _stream_graph drives both), so status/
  // token/done/error handling only needs writing once. "clarification" is
  // the one event type only a fresh turn's detect_intent can produce (a
  // resume answer either resolves it or the graph runs straight to "done").
  async function consumeStream(eventStream: AsyncGenerator<ChatStreamEvent>, assistantId: string) {
    // deterministic's retry loop (generate -> self_eval -> reformulate ->
    // ... -> generate again) means "generate" can run more than once in a
    // turn, each one streaming full draft text -- react_agent's tool-
    // calling loop also revisits "generate" repeatedly, but its
    // intermediate rounds are pure tool calls with no text content, so
    // they never had this problem. No draft is worth showing live until
    // self_eval has accepted it -- streaming it (even with the per-status
    // content reset already in place) meant the user watched a full
    // possibly-wrong answer type out, then either vanish and get replaced
    // (a retry) or just sit there having never been vetted. So every
    // "generate" pass suppresses live tokens; the accepted answer is
    // revealed only once "done" (or "clarification") arrives.
    let sawSelfEval = false;
    let suppressTokens = false;
    for await (const streamEvent of eventStream) {
      if (streamEvent.event === "thread") {
        threadIdRef.current = streamEvent.data.thread_id;
        setActiveThreadId(streamEvent.data.thread_id);
      } else if (streamEvent.event === "status") {
        if (streamEvent.data.node === "self_eval") {
          sawSelfEval = true;
        }
        const isRetriedDraft = streamEvent.data.node === "generate" && sawSelfEval;
        suppressTokens = streamEvent.data.node === "generate";
        updateMessage(assistantId, {
          status: isRetriedDraft ? "Refining the answer…" : STATUS_LABELS[streamEvent.data.node] ?? streamEvent.data.node,
          ...(streamEvent.data.node === "generate" ? { content: "" } : {}),
        });
      } else if (streamEvent.event === "token") {
        if (suppressTokens) continue;
        setMessages((previous) =>
          previous.map((message) =>
            message.id === assistantId
              ? { ...message, content: message.content + streamEvent.data.content, status: undefined }
              : message
          )
        );
      } else if (streamEvent.event === "clarification") {
        setPendingClarification(streamEvent.data.thread_id);
        updateMessage(assistantId, {
          content: streamEvent.data.question,
          clarification: { question: streamEvent.data.question, options: streamEvent.data.options },
          status: undefined,
          pending: false,
        });
      } else if (streamEvent.event === "done") {
        setPendingClarification(null);
        updateMessage(assistantId, {
          content: streamEvent.data.answer,
          citations: streamEvent.data.citations,
          status: undefined,
          pending: false,
          messageId: streamEvent.data.message_id,
        });
      } else if (streamEvent.event === "error") {
        updateMessage(assistantId, { pending: false, status: undefined });
        setError(streamEvent.data.message);
      }
    }
  }

  async function runTurn(eventStream: AsyncGenerator<ChatStreamEvent>, assistantId: string) {
    setSending(true);
    setError(null);
    try {
      await consumeStream(eventStream, assistantId);
    } catch (caughtError) {
      updateMessage(assistantId, { pending: false, status: undefined });
      setError(caughtError instanceof Error ? caughtError.message : "Something went wrong");
    } finally {
      setSending(false);
      // Picks up the new/renamed/reordered thread this turn just created or
      // touched -- cheap enough to just refetch rather than reconcile by hand.
      listChatThreads()
        .then(setThreads)
        .catch(() => {});
    }
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    const text = input.trim();
    if (!text || sending) return;
    setInput("");

    if (pendingClarification) {
      await answerClarification(text);
      return;
    }

    await sendQuestion(text);
  }

  async function sendQuestion(text: string) {
    const userMessage: ChatMessage = { id: uniqueId(), role: "user", content: text };
    const assistantId = uniqueId();
    setMessages((previous) => [
      ...previous,
      userMessage,
      { id: assistantId, role: "assistant", content: "", pending: true, status: "Thinking…" },
    ]);
    await runTurn(streamChat(text, threadIdRef.current), assistantId);
  }

  // Ticket 09: answers a clarification question, either by clicking one of
  // its suggested options or by typing free text into the normal input
  // (handleSubmit routes here when pendingClarification is set). Adds a new
  // user/assistant bubble pair rather than mutating the clarification
  // message in place, so the exchange reads like the rest of the
  // conversation instead of the question silently turning into the answer.
  async function answerClarification(answer: string) {
    const threadId = pendingClarification;
    if (!threadId) return;
    setPendingClarification(null);

    const userMessage: ChatMessage = { id: uniqueId(), role: "user", content: answer };
    const assistantId = uniqueId();
    setMessages((previous) => [
      ...previous,
      userMessage,
      { id: assistantId, role: "assistant", content: "", pending: true, status: "Thinking…" },
    ]);
    await runTurn(resumeChat(threadId, answer), assistantId);
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

  // New signups start is_active=False (backend/auth/dependencies.py's
  // require_chat_access) until an admin enables them from the Users tab --
  // admins bypass this check entirely, so it never applies to them.
  if (!user.is_admin && !user.is_active) {
    return (
      <main className="page">
        <h1>Chat</h1>
        <p>Your account is pending admin approval. You&rsquo;ll be able to chat once an admin enables it.</p>
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
            {threads.map((thread) => (
              <button
                key={thread.thread_id}
                type="button"
                className={`chat-thread-item${
                  thread.thread_id === activeThreadId ? " chat-thread-item-active" : ""
                }`}
                onClick={() => selectThread(thread.thread_id)}
                disabled={sending}
              >
                <span className="chat-thread-item-title">{thread.title}</span>
                <span className="chat-thread-item-date">{formatThreadDate(thread.updated_at)}</span>
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
              messages.map((message) => (
                <div key={message.id} className={`chat-bubble chat-bubble-${message.role}`}>
                  {message.content && (
                    <div className="chat-bubble-text">
                      <ReactMarkdown>{stripCitationMarkers(message.content)}</ReactMarkdown>
                    </div>
                  )}
                  {message.status && <div className="chat-status">{message.status}</div>}
                  {message.clarification && (
                    <div className="chat-clarification">
                      {message.clarification.options.length > 0 && (
                        <div className="chat-clarification-options">
                          {message.clarification.options.map((option) => (
                            <button
                              key={option}
                              type="button"
                              className="btn btn-ghost chat-clarification-option"
                              onClick={() => answerClarification(option)}
                              disabled={sending || pendingClarification === null}
                            >
                              {option}
                            </button>
                          ))}
                        </div>
                      )}
                      <div className="chat-clarification-hint">Or type your answer below.</div>
                    </div>
                  )}
                  {message.citations && message.citations.length > 0 && (
                    <div className="chat-citations">
                      {message.citations.map((citation, index) => (
                        <button
                          key={`${citation.document_id}#${citation.chunk_index}-${index}`}
                          type="button"
                          className={`chat-citation${
                            openCitation?.document_id === citation.document_id &&
                            openCitation?.chunk_index === citation.chunk_index
                              ? " chat-citation-active"
                              : ""
                          }`}
                          onClick={() => openCitationPane(citation)}
                          title={citationLabel(citation)}
                        >
                          {citationLabel(citation)}
                        </button>
                      ))}
                    </div>
                  )}
                  {message.role === "assistant" && message.messageId && activeThreadId && (
                    <MessageFeedback
                      key={message.messageId}
                      threadId={activeThreadId}
                      messageId={message.messageId}
                      initialFeedback={message.feedback}
                      onSubmitted={(feedback) => updateMessage(message.id, { feedback })}
                    />
                  )}
                </div>
              ))}
            <div ref={bottomRef} />
          </div>

          {error && <p className="alert" role="alert">{error}</p>}

          <form onSubmit={handleSubmit} className="chat-input-row">
            <input
              value={input}
              onChange={(event) => setInput(event.target.value)}
              placeholder={pendingClarification ? "Type your answer…" : "Ask a question…"}
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
            {documentPaneLoading && <div className="empty-state">Loading document…</div>}
            {documentPaneError && <p className="alert" role="alert">{documentPaneError}</p>}
            {documentChunks && isPdfFilename(openCitation.filename) && (
              <>
                {documentFileError && <p className="alert" role="alert">{documentFileError}</p>}
                {!documentFileError && documentFileData && (
                  <PdfCitationViewer
                    file={documentFileData}
                    pageNumber={
                      documentChunks.find((chunk) => chunk.chunk_index === openCitation.chunk_index)
                        ?.page_number ?? 1
                    }
                    searchText={
                      documentChunks.find((chunk) => chunk.chunk_index === openCitation.chunk_index)
                        ?.content ?? ""
                    }
                    onError={() => setDocumentFileError("This PDF couldn't be displayed.")}
                  />
                )}
                {!documentFileError && !documentFileData && (
                  <div className="empty-state">Loading PDF…</div>
                )}
              </>
            )}
            {documentChunks && !isPdfFilename(openCitation.filename) && (
              <div className="chat-doc-pane-body">
                {documentChunks.map((chunk) => {
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
