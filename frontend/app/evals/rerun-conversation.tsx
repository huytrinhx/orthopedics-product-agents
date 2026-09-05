"use client";

// Ticket 15: renders one rerun attempt's conversation using the exact same
// bubble/citation/feedback markup and CSS classes as app/chat/page.tsx --
// per the grilling session that scoped this feature, a rerun is meant to
// look and behave like a real chat turn, not a stripped-down summary row.
//
// Two modes: "live" (just triggered -- streams the new turn, seeded first
// from the history graph.aupdate_state already wrote server-side) and
// "history" (reopening a past rerun -- a plain GET /chat/threads/{id}, no
// streaming). Both converge on the same local `messages: ChatMessage[]`
// state so the render below never needs to know which mode produced it.
import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { getChatThread } from "../../lib/chat/api";
import { citationLabel, STATUS_LABELS, stripCitationMarkers } from "../../lib/chat/format";
import type { ChatMessage, ChatStreamEvent } from "../../lib/chat/types";
import { rerunChat, resumeRerunChat } from "../../lib/feedback/api";
import { MessageFeedback } from "../chat/message-feedback";

function uid(): string {
  return Math.random().toString(36).slice(2);
}

type Props =
  | {
      mode: "live";
      originalThreadId: string;
      originalMessageId: string;
      workflowName: string;
      onThreadCreated: (threadId: string) => void;
    }
  | { mode: "history"; threadId: string; workflowName: string | null };

export function RerunConversation(props: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [threadId, setThreadId] = useState<string | null>(
    props.mode === "history" ? props.threadId : null
  );
  const [pendingClarification, setPendingClarification] = useState(false);
  const [replyInput, setReplyInput] = useState("");
  const [error, setError] = useState<string | null>(null);
  const startedRef = useRef(false);

  function updateMessage(id: string, patch: Partial<ChatMessage>) {
    setMessages((prev) => prev.map((m) => (m.id === id ? { ...m, ...patch } : m)));
  }

  // Mirrors app/chat/page.tsx's consumeStream (see its own comment for why
  // "generate" tokens are suppressed until self_eval has accepted a draft).
  async function consume(gen: AsyncGenerator<ChatStreamEvent>, assistantId: string) {
    let sawSelfEval = false;
    let suppressTokens = false;
    for await (const evt of gen) {
      if (evt.event === "thread") {
        setThreadId(evt.data.thread_id);
        if (props.mode === "live") props.onThreadCreated(evt.data.thread_id);
        const history = await getChatThread(evt.data.thread_id).catch(() => null);
        if (history) {
          setMessages([
            ...history.messages.map((m) => ({
              id: uid(),
              role: m.role,
              content: m.content,
              citations: m.citations,
              messageId: m.message_id,
              feedback: m.feedback,
            })),
            { id: assistantId, role: "assistant" as const, content: "", pending: true, status: "Thinking…" },
          ]);
        }
      } else if (evt.event === "status") {
        if (evt.data.node === "self_eval") sawSelfEval = true;
        const isRetriedDraft = evt.data.node === "generate" && sawSelfEval;
        suppressTokens = evt.data.node === "generate";
        updateMessage(assistantId, {
          status: isRetriedDraft ? "Refining the answer…" : STATUS_LABELS[evt.data.node] ?? evt.data.node,
          ...(evt.data.node === "generate" ? { content: "" } : {}),
        });
      } else if (evt.event === "token") {
        if (suppressTokens) continue;
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId ? { ...m, content: m.content + evt.data.content, status: undefined } : m
          )
        );
      } else if (evt.event === "clarification") {
        setPendingClarification(true);
        updateMessage(assistantId, {
          content: evt.data.question,
          clarification: { question: evt.data.question, options: evt.data.options },
          status: undefined,
          pending: false,
        });
      } else if (evt.event === "done") {
        setPendingClarification(false);
        updateMessage(assistantId, {
          content: evt.data.answer,
          citations: evt.data.citations,
          status: undefined,
          pending: false,
          messageId: evt.data.message_id,
        });
      } else if (evt.event === "error") {
        updateMessage(assistantId, { pending: false, status: undefined });
        setError(evt.data.message);
      }
    }
  }

  useEffect(() => {
    if (props.mode === "history") {
      getChatThread(props.threadId)
        .then((t) =>
          setMessages(
            t.messages.map((m) => ({
              id: uid(),
              role: m.role,
              content: m.content,
              citations: m.citations,
              messageId: m.message_id,
              feedback: m.feedback,
            }))
          )
        )
        .catch((err) => setError(err instanceof Error ? err.message : "Couldn't load this rerun"));
      return;
    }
    if (startedRef.current) return; // StrictMode double-invoke guard -- a rerun must fire exactly once
    startedRef.current = true;
    const assistantId = uid();
    consume(rerunChat(props.originalThreadId, props.originalMessageId, props.workflowName), assistantId).catch(
      (err) => setError(err instanceof Error ? err.message : "Rerun failed")
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function submitReply(e: React.FormEvent) {
    e.preventDefault();
    const text = replyInput.trim();
    const workflowName = props.workflowName;
    if (!text || !threadId || !workflowName) return;
    setReplyInput("");
    setPendingClarification(false);
    const userId = uid();
    const assistantId = uid();
    setMessages((prev) => [
      ...prev,
      { id: userId, role: "user", content: text },
      { id: assistantId, role: "assistant", content: "", pending: true, status: "Thinking…" },
    ]);
    await consume(resumeRerunChat(threadId, workflowName, text), assistantId);
  }

  return (
    <div className="rerun-conversation">
      {messages.map((m) => (
        <div key={m.id} className={`chat-bubble chat-bubble-${m.role}`}>
          {m.content && (
            <div className="chat-bubble-text">
              <ReactMarkdown>{stripCitationMarkers(m.content)}</ReactMarkdown>
            </div>
          )}
          {m.status && <div className="chat-status">{m.status}</div>}
          {m.citations && m.citations.length > 0 && (
            <div className="chat-citations">
              {m.citations.map((c, i) => (
                <span
                  key={`${c.document_id}#${c.chunk_index}-${i}`}
                  className="chat-citation"
                  style={{ cursor: "default" }}
                  title={citationLabel(c)}
                >
                  {citationLabel(c)}
                </span>
              ))}
            </div>
          )}
          {m.role === "assistant" && m.messageId && threadId && (
            <MessageFeedback
              threadId={threadId}
              messageId={m.messageId}
              initialFeedback={m.feedback}
              onSubmitted={(feedback) => updateMessage(m.id, { feedback })}
            />
          )}
        </div>
      ))}
      {pendingClarification && (
        <form className="rerun-reply-row chat-input-row" onSubmit={submitReply}>
          <input
            value={replyInput}
            onChange={(e) => setReplyInput(e.target.value)}
            placeholder="Answer the clarification to continue this rerun…"
            autoFocus
          />
          <button type="submit" className="btn btn-ghost">
            Reply
          </button>
        </form>
      )}
      {error && <p className="alert" role="alert">{error}</p>}
    </div>
  );
}
