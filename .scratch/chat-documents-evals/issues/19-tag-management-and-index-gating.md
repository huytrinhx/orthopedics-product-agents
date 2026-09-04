# 19: System/Document-Type tag management UI + Index gating

**What to build:** Two related Document Manager fixes, not scoped from the
original 17-ticket list — captured directly as a backlog item, not yet
designed or implemented.

1. **Delete a System tag / Document-Type tag.** Ticket 05 only ever added
   create ("no fixed/hardcoded enum") — there's no way to remove a tag once
   created. Needs a decision on what happens to documents currently assigned
   a tag that gets deleted (block the delete while in use? null out the FK?
   cascade?).
2. **A visual management UI for the two tag lists**, surfaced as two
   dropdowns at the top of the Document Manager page — distinct from the
   existing inline `TagSelect` pickers (`frontend/app/documents/tag-select.tsx`)
   used per-row/at-upload to *assign* a tag to one document. This is a
   dedicated place to view/manage the System and Document-Type lists
   themselves (rename? just list + delete, per above?).
3. **Only the Index button should trigger indexing — no other pathway.**
   Currently `PATCH /documents/{id}/tags` (`backend/api/routes/documents.py`)
   also queues a background re-index as a side effect of re-tagging (see its
   comment: "a re-tag has to re-run the same pipeline seam as upload so that
   metadata stays in sync"). That auto-trigger needs to go — re-tagging
   should just update the tag, leaving the document's indexed
   status/content alone until an admin explicitly hits Index/Reindex.

**Blocked by:** 05 (System and Document-Type tags, done)

**Status:** backlog — not yet scoped or started

- [ ] Admin can delete an existing System tag
- [ ] Admin can delete an existing Document-Type tag
- [ ] Decide + implement what happens to documents already carrying a
      deleted tag
- [ ] Document Manager page has a visible UI (two dropdowns at the top) for
      managing the System/Document-Type lists, separate from per-document
      tag assignment
- [ ] `PATCH /documents/{id}/tags` no longer queues a background re-index —
      indexing only fires from `POST /{document_id}/index`
