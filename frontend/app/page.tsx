import Link from "next/link";

export default function Home() {
  return (
    <main>
      <h1>Orthopedics Product Agents</h1>
      <ul>
        <li>
          <Link href="/chat">Chat</Link>
        </li>
        <li>
          <Link href="/documents">Document Manager</Link>
        </li>
      </ul>
    </main>
  );
}
