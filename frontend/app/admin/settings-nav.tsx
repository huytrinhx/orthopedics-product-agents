"use client";

import type { MouseEvent } from "react";

export interface SettingsNavItem {
  id: string;
  label: string;
}

// The Settings page's section list: a vertical nav beside the selected
// section on a desktop viewport, a horizontally scrolling row above it on a
// narrow one (globals.css, .settings-nav). Each item is a real link to
// `?section=<id>`, so it can be opened in a new tab or copied; a plain click
// is handled in place by onSelect instead of reloading the page.
export function SettingsNav({
  items,
  selectedId,
  onSelect,
}: {
  items: SettingsNavItem[];
  selectedId: string;
  onSelect: (id: string) => void;
}) {
  function handleClick(event: MouseEvent<HTMLAnchorElement>, id: string) {
    const opensElsewhere =
      event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey;
    if (opensElsewhere) return;
    event.preventDefault();
    onSelect(id);
  }

  return (
    <nav className="settings-nav" aria-label="Settings sections">
      <ul>
        {items.map((item) => (
          <li key={item.id}>
            <a
              href={`?section=${item.id}`}
              aria-current={item.id === selectedId ? "page" : undefined}
              onClick={(event) => handleClick(event, item.id)}
            >
              {item.label}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  );
}
