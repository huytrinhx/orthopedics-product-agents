"use client";

import { useState, type FormEvent } from "react";
import type { Tag } from "../../lib/documents/tags/types";

const NEW_VALUE = "__new__";

// Shared by the upload form's two pickers and each table row's inline
// pickers (ticket 05: assign a tag at upload or by editing afterward) --
// same "pick an existing tag, or add a new one on the spot" behavior for
// both Systems and Document Types.
export function TagSelect({
  label,
  tags,
  value,
  onChange,
  onCreate,
}: {
  label: string;
  tags: Tag[];
  value: string;
  onChange: (id: string) => void;
  onCreate: (name: string) => Promise<Tag>;
}) {
  const [adding, setAdding] = useState(false);
  const [newName, setNewName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [creating, setCreating] = useState(false);

  function handleSelect(id: string) {
    if (id === NEW_VALUE) {
      setAdding(true);
      return;
    }
    onChange(id);
  }

  async function handleCreate(e: FormEvent) {
    e.preventDefault();
    const name = newName.trim();
    if (!name) return;
    setCreating(true);
    setError(null);
    try {
      const tag = await onCreate(name);
      onChange(tag.id);
      setAdding(false);
      setNewName("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "failed to create tag");
    } finally {
      setCreating(false);
    }
  }

  if (adding) {
    return (
      <form onSubmit={handleCreate} className="tag-add-row">
        <input
          value={newName}
          onChange={(e) => setNewName(e.target.value)}
          placeholder={`New ${label.toLowerCase()}`}
          autoFocus
        />
        <button type="submit" className="btn btn-ghost" disabled={creating}>
          {creating ? "Adding…" : "Add"}
        </button>
        <button
          type="button"
          className="btn-text"
          onClick={() => {
            setAdding(false);
            setError(null);
          }}
        >
          Cancel
        </button>
        {error && <span className="tag-add-error">{error}</span>}
      </form>
    );
  }

  return (
    <select value={value} onChange={(e) => handleSelect(e.target.value)} aria-label={label}>
      <option value="">{label}: none</option>
      {tags.map((tag) => (
        <option key={tag.id} value={tag.id}>
          {tag.name}
        </option>
      ))}
      <option value={NEW_VALUE}>+ New {label.toLowerCase()}…</option>
    </select>
  );
}
