"use client";

// Streaming chat UI: connects to POST /chat/stream (SSE over a POST,
// parsed by hand in lib/chat/api.ts -- native EventSource is GET-only).
// Tickets 11/12 (per-message 4-axis feedback) extend this. Ticket 10 added
// the history sidebar (past threads, resuming a transcript, a separate
// "New conversation" action). A citation chip opens the cited document in
// a modal (citation-viewer-modal.tsx): the whole PDF at the cited page with
// the passage highlighted, or a non-PDF's chunks with the cited one marked.
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
// boundary for no benefit here. A rerun with per-slot model overrides
// (issue #24) also carries `&models=<json>`; they go with that first turn
// (admin-only server-side) and stay with the conversation, and its answers
// show which models they ran on.
import { useEffect, useRef, useState, type FormEvent } from "react";
import Link from "next/link";
import ReactMarkdown from "react-markdown";
import { getChatThread, listChatThreads, resumeChat, streamChat } from "../../lib/chat/api";
import { useAuth } from "../../lib/auth-context";
import { STATUS_LABELS, citationLabel, stripCitationMarkers } from "../../lib/chat/format";
import type { ChatCitation, ChatMessage, ChatStreamEvent, ChatThread } from "../../lib/chat/types";
import type { ModelOverrides, SlotModelUsed } from "../../lib/model-assignments/types";
import { MessageFeedback } from "./message-feedback";
import { CitationViewerModal } from "./citation-viewer-modal";

function uniqueId(): string {
  return Math.random().toString(36).slice(2);
}

// `?models=` from the Eval tab's rerun. Anything that isn't a JSON object
// is ignored rather than sent.
function parseModelOverrides(raw: string | null): ModelOverrides | undefined {
  if (!raw) return undefined;
  try {
    const parsed: unknown = JSON.parse(raw);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? (parsed as ModelOverrides) : undefined;
  } catch {
    return undefined;
  }
}

const SLOT_LABELS: Record<string, string> = {
  intent_classification: "Intent",
  deterministic_answer: "Answer",
  react_agent: "ReAct agent",
};

function formatModelsUsed(models: Record<string, SlotModelUsed>): string {
  return Object.entries(models)
    .map(
      ([slot, used]) =>
        `${SLOT_LABELS[slot] ?? slot}: ${used.model_id} (${used.reasoning_effort ?? "default effort"}${
          used.source === "override" ? ", override" : ""
        })`
    )
    .join(" · ");
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
    const searchParams = new URLSearchParams(window.location.search);
    const question = searchParams.get("ask")?.trim();
    if (!question) return;
    askedFromUrlRef.current = true;
    window.history.replaceState(null, "", window.location.pathname);
    sendQuestion(question, parseModelOverrides(searchParams.get("models")));
    // sendQuestion is deliberately not a dependency -- this only ever runs
    // once, on the first render with a user.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user]);

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
          models: streamEvent.data.models,
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

  async function sendQuestion(text: string, modelOverrides?: ModelOverrides) {
    const userMessage: ChatMessage = { id: uniqueId(), role: "user", content: text };
    const assistantId = uniqueId();
    setMessages((previous) => [
      ...previous,
      userMessage,
      { id: assistantId, role: "assistant", content: "", pending: true, status: "Thinking…" },
    ]);
    await runTurn(streamChat(text, threadIdRef.current, modelOverrides), assistantId);
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
              <div className="empty-state">Ask a question about a product family to get started.</div>
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
                  {message.models && (
                    <div className="chat-models-used">Ran on: {formatModelsUsed(message.models)}</div>
                  )}
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
                          onClick={() => setOpenCitation(citation)}
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
          <CitationViewerModal
            key={`${openCitation.document_id}#${openCitation.chunk_index}`}
            citation={openCitation}
            onClose={closeCitation}
          />
        )}

      </div>
    </main>
  );
}
