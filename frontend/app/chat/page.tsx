// Streaming chat UI: connects to POST /chat/{workflow}/stream (SSE),
// surfaces interrupt() clarification prompts, and hosts the 4-axis
// feedback controls (faithfulness, relevance, style, citation) per
// message, submitted to POST /feedback/.
export default function ChatPage() {
  return (
    <main>
      <h1>Chat</h1>
      <p>TODO: streaming chat interface</p>
    </main>
  );
}
