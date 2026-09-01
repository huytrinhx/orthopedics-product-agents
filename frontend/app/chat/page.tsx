// Streaming chat UI: connects to POST /chat/{workflow}/stream (SSE),
// surfaces interrupt() clarification prompts, and hosts the 4-axis
// feedback controls (faithfulness, relevance, style, citation) per
// message, submitted to POST /feedback/.
export default function ChatPage() {
  return (
    <main className="page">
      <span className="eyebrow">Ask OrthoMate</span>
      <h1>Chat</h1>
      <div className="empty-state">Streaming chat interface — coming in ticket 08.</div>
    </main>
  );
}
