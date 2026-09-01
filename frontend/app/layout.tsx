import type { ReactNode } from "react";
import { AuthProvider } from "../lib/auth-context";
import { Nav } from "./nav";
import "./globals.css";

export const metadata = {
  title: "OrthoMate",
  description: "Cited, page-traceable answers for foot & ankle surgeons.",
};

// Plain <link> tags rather than next/font/google: this is a static export
// (next.config.js) served by FastAPI (root Dockerfile) — the browser can
// fetch fonts from Google's CDN at runtime same as any other asset, with no
// build-time network dependency for `next build` (which self-hosting via
// next/font would require).
export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        {/* eslint-disable-next-line @next/next/no-page-custom-font -- this
        is the root layout, not a page under pages/_document.js; it renders
        once for the whole app router tree, so the "only loads for a single
        page" warning doesn't apply. */}
        <link
          rel="stylesheet"
          href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&family=IBM+Plex+Mono:wght@400;500;600&display=swap"
        />
      </head>
      <body>
        <AuthProvider>
          <Nav />
          {children}
        </AuthProvider>
      </body>
    </html>
  );
}
