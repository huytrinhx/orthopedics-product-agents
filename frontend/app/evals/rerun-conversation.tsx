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

function uniqueId(): string {
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
    setMessages((previous) => previous.map((message) => (message.id === id ? { ...message, ...patch } : message)));
  }

  // Mirrors app/chat/page.tsx's consumeStream (see its own comment for why
  // "generate" tokens are suppressed until self_eval has accepted a draft).
  async function consume(eventStream: AsyncGenerator<ChatStreamEvent>, assistantId: string) {
    let sawSelfEval = false;
    let suppressTokens = false;
    for await (const streamEvent of eventStream) {
      if (streamEvent.event === "thread") {
        setThreadId(streamEvent.data.thread_id);
        if (props.mode === "live") props.onThreadCreated(streamEvent.data.thread_id);
        const history = await getChatThread(streamEvent.data.thread_id).catch(() => null);
        if (history) {
          setMessages([
            ...history.messages.map((message) => ({
              id: uniqueId(),
              role: message.role,
              content: message.content,
              citations: message.citations,
              messageId: message.message_id,
              feedback: message.feedback,
            })),
            { id: assistantId, role: "assistant" as const, content: "", pending: true, status: "Thinking…" },
          ]);
        }
      } else if (streamEvent.event === "status") {
        if (streamEvent.data.node === "self_eval") sawSelfEval = true;
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
            message.id === assistantId ? { ...message, content: message.content + streamEvent.data.content, status: undefined } : message
          )
        );
      } else if (streamEvent.event === "clarification") {
        setPendingClarification(true);
        updateMessage(assistantId, {
          content: streamEvent.data.question,
          clarification: { question: streamEvent.data.question, options: streamEvent.data.options },
          status: undefined,
          pending: false,
        });
      } else if (streamEvent.event === "done") {
        setPendingClarification(false);
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

  useEffect(() => {
    if (props.mode === "history") {
      getChatThread(props.threadId)
        .then((transcript) =>
          setMessages(
            transcript.messages.map((message) => ({
              id: uniqueId(),
              role: message.role,
              content: message.content,
              citations: message.citations,
              messageId: message.message_id,
              feedback: message.feedback,
            }))
          )
        )
        .catch((caughtError) => setError(caughtError instanceof Error ? caughtError.message : "Couldn't load this rerun"));
      return;
    }
    if (startedRef.current) return; // StrictMode double-invoke guard -- a rerun must fire exactly once
    startedRef.current = true;
    const assistantId = uniqueId();
    consume(rerunChat(props.originalThreadId, props.originalMessageId, props.workflowName), assistantId).catch(
      (caughtError) => setError(caughtError instanceof Error ? caughtError.message : "Rerun failed")
    );
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function submitReply(event: React.FormEvent) {
    event.preventDefault();
    const text = replyInput.trim();
    const workflowName = props.workflowName;
    if (!text || !threadId || !workflowName) return;
    setReplyInput("");
    setPendingClarification(false);
    const userId = uniqueId();
    const assistantId = uniqueId();
    setMessages((previous) => [
      ...previous,
      { id: userId, role: "user", content: text },
      { id: assistantId, role: "assistant", content: "", pending: true, status: "Thinking…" },
    ]);
    await consume(resumeRerunChat(threadId, workflowName, text), assistantId);
  }

  return (
    <div className="rerun-conversation">
      {messages.map((message) => (
        <div key={message.id} className={`chat-bubble chat-bubble-${message.role}`}>
          {message.content && (
            <div className="chat-bubble-text">
              <ReactMarkdown>{stripCitationMarkers(message.content)}</ReactMarkdown>
            </div>
          )}
          {message.status && <div className="chat-status">{message.status}</div>}
          {message.citations && message.citations.length > 0 && (
            <div className="chat-citations">
              {message.citations.map((citation, index) => (
                <span
                  key={`${citation.document_id}#${citation.chunk_index}-${index}`}
                  className="chat-citation"
                  style={{ cursor: "default" }}
                  title={citationLabel(citation)}
                >
                  {citationLabel(citation)}
                </span>
              ))}
            </div>
          )}
          {message.role === "assistant" && message.messageId && threadId && (
            <MessageFeedback
              threadId={threadId}
              messageId={message.messageId}
              initialFeedback={message.feedback}
              onSubmitted={(feedback) => updateMessage(message.id, { feedback })}
            />
          )}
        </div>
      ))}
      {pendingClarification && (
        <form className="rerun-reply-row chat-input-row" onSubmit={submitReply}>
          <input
            value={replyInput}
            onChange={(event) => setReplyInput(event.target.value)}
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
