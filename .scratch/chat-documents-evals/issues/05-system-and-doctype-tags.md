# 05: System and Document-Type tag management

**What to build:** An admin can maintain two small lookup lists — product Systems (e.g. MIS, REFLEX) and Document Types (e.g. Brochure, Surgical Technique) — and tag each uploaded document with exactly one of each.

**Blocked by:** 04 (Document upload + list with status)

**Status:** done

- [x] An admin can create a new System tag and a new Document-Type tag from the Document Manager UI (no fixed/hardcoded enum)
- [x] Uploading (or editing) a document lets the admin assign exactly one System tag and one Document-Type tag to it
- [x] The document list shows each document's assigned tags
- [x] Existing tags can be reused across multiple documents (many documents, one tag each)

**Forward note (from ticket 07's schema design):** since System tags are
admin-defined (no hardcoded enum), nothing here needs to change today. But
the real master item file (`backend/evals/unite-master-csv.txt`) has ~27
distinct tray/instrument-set values (e.g. `REFLEX® HYBRID Implant System`,
`REFLEX® MINI Nitinol Staple System`) one level *below* the `MIS`/`REFLEX`
granularity System tags are actually being created at today. Ticket 07 models
that finer level as a separate `Tray` graph node under `ProductFamily` rather
than asking admins to tag documents at tray granularity — if tray-level
document tagging turns out to be needed too, that's a reason to revisit here.
