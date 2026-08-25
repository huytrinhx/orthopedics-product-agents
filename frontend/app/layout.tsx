import type { ReactNode } from "react";

export const metadata = {
  title: "Orthopedics Product Agents",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
