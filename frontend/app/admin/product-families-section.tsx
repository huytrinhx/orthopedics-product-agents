"use client";

import { Fragment, useEffect, useState, type FormEvent } from "react";
import {
  listProductFamilies,
  setFamilyDescription,
  setFamilyDisplayName,
  setTrayDisplayName,
} from "../../lib/catalog/api";
import type { ProductFamily } from "../../lib/catalog/types";

function errorMessage(caughtError: unknown, fallback: string): string {
  return caughtError instanceof Error ? caughtError.message : fallback;
}

// One family or tray row. Only the display name is editable; the catalog
// name beside it is the master file's, shown so an admin can see what a
// display name stands in for. Saving a blank display name resets it to the
// catalog name (server-side), as does Reset.
function DisplayNameRow({
  catalogName,
  displayName,
  kind,
  onSave,
}: {
  catalogName: string;
  displayName: string;
  kind: "family" | "tray";
  onSave: (nextDisplayName: string) => Promise<void>;
}) {
  const [editing, setEditing] = useState(false);
  const [draftName, setDraftName] = useState(displayName);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function startEditing() {
    setDraftName(displayName);
    setError(null);
    setEditing(true);
  }

  async function save(nextDisplayName: string) {
    setBusy(true);
    setError(null);
    try {
      await onSave(nextDisplayName);
      setEditing(false);
    } catch (caughtError) {
      setError(errorMessage(caughtError, `failed to update ${kind} display name`));
    } finally {
      setBusy(false);
    }
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    const nextDisplayName = draftName.trim();
    if (nextDisplayName === displayName) {
      setEditing(false);
      return;
    }
    await save(nextDisplayName);
  }

  const renamed = displayName !== catalogName;

  return (
    <tr className={kind === "tray" ? "settings-tree-tray" : "settings-tree-family"}>
      <td>
        {editing ? (
          <form onSubmit={handleSubmit} className="tag-add-row">
            <input
              value={draftName}
              onChange={(event) => setDraftName(event.target.value)}
              placeholder={catalogName}
              aria-label={`Display name for ${catalogName}`}
              autoFocus
            />
            <button type="submit" className="btn btn-ghost" disabled={busy}>
              {busy ? "Saving…" : "Save"}
            </button>
            <button type="button" className="btn-text" disabled={busy} onClick={() => setEditing(false)}>
              Cancel
            </button>
          </form>
        ) : (
          displayName
        )}
        {error && <div className="tag-add-error">{error}</div>}
      </td>
      <td className="settings-catalog-name">{renamed ? catalogName : "—"}</td>
      <td>
        {!editing && (
          <div className="actions-cell">
            <button type="button" className="btn-text" onClick={startEditing}>
              Edit
            </button>
            {renamed && (
              <button type="button" className="btn-text" disabled={busy} onClick={() => save("")}>
                Reset
              </button>
            )}
          </div>
        )}
      </td>
    </tr>
  );
}

// The family's description for the intent classifiers (issue #30), under its
// name. It's generated from the catalog's trays and Parts; an admin can
// replace it with their own, and Reset goes back to the generated text.
// Saving text identical to the generated description saves no edit.
function FamilyDescriptionRow({
  familyId,
  familyName,
  generatedDescription,
  descriptionOverride,
  onSave,
}: {
  familyId: string;
  familyName: string;
  generatedDescription: string;
  descriptionOverride: string | null;
  onSave: (nextDescription: string) => Promise<void>;
}) {
  const [editing, setEditing] = useState(false);
  const currentDescription = descriptionOverride ?? generatedDescription;
  const [draftDescription, setDraftDescription] = useState(currentDescription);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  function startEditing() {
    setDraftDescription(currentDescription);
    setError(null);
    setEditing(true);
  }

  async function save(nextDescription: string) {
    setBusy(true);
    setError(null);
    try {
      await onSave(nextDescription);
      setEditing(false);
    } catch (caughtError) {
      setError(errorMessage(caughtError, "failed to update the family description"));
    } finally {
      setBusy(false);
    }
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    const nextDescription = draftDescription.trim();
    if (nextDescription === currentDescription) {
      setEditing(false);
      return;
    }
    await save(nextDescription === generatedDescription ? "" : nextDescription);
  }

  const edited = descriptionOverride !== null;
  const textareaId = `family-description-${familyId}`;

  return (
    <tr className="settings-tree-description">
      <td colSpan={3}>
        <div className="family-description-head">
          <span className="family-description-label">Classifier description</span>
          <span className={edited ? "badge badge-processing" : "badge"}>{edited ? "Edited" : "Generated"}</span>
          {!editing && (
            <span className="actions-cell">
              <button type="button" className="btn-text" onClick={startEditing}>
                Edit
              </button>
              {edited && (
                <button type="button" className="btn-text" disabled={busy} onClick={() => save("")}>
                  Reset to generated
                </button>
              )}
            </span>
          )}
        </div>
        {editing ? (
          <form onSubmit={handleSubmit} className="family-description-form">
            <textarea
              id={textareaId}
              aria-label={`Classifier description for ${familyName}`}
              value={draftDescription}
              onChange={(event) => setDraftDescription(event.target.value)}
              placeholder={generatedDescription || "Describe the family's products, trays and brand."}
              rows={3}
              autoFocus
            />
            <div className="tag-add-row">
              <button type="submit" className="btn btn-ghost" disabled={busy}>
                {busy ? "Saving…" : "Save"}
              </button>
              <button type="button" className="btn-text" disabled={busy} onClick={() => setEditing(false)}>
                Cancel
              </button>
            </div>
          </form>
        ) : (
          <p className="family-description-text">
            {currentDescription || "No description: the classifiers see only the family's name."}
          </p>
        )}
        {edited && !editing && generatedDescription && (
          <p className="family-description-generated">Generated: {generatedDescription}</p>
        )}
        {error && <div className="tag-add-error">{error}</div>}
      </td>
    </tr>
  );
}

// Settings section: the master catalog's Product Family > Tray tree (issue
// #15). Which families and trays exist, and which tray sits under which
// family, come from the master file and are read-only here; the display
// name is the one editable field, and it's what users see everywhere. A
// catalog re-seed never overwrites it.
export function ProductFamiliesSection() {
  const [families, setFamilies] = useState<ProductFamily[]>([]);
  const [loaded, setLoaded] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    listProductFamilies()
      .then((fetchedFamilies) => {
        setFamilies(fetchedFamilies);
        setLoadError(null);
      })
      .catch((caughtError) => setLoadError(errorMessage(caughtError, "failed to load product families")))
      .finally(() => setLoaded(true));
  }, []);

  async function saveFamily(familyId: string, nextDisplayName: string) {
    const updated = await setFamilyDisplayName(familyId, nextDisplayName);
    setFamilies((previous) =>
      previous.map((family) =>
        family.id === updated.id ? { ...family, display_name: updated.display_name } : family
      )
    );
  }

  async function saveDescription(familyId: string, nextDescription: string) {
    const updated = await setFamilyDescription(familyId, nextDescription);
    setFamilies((previous) =>
      previous.map((family) =>
        family.id === updated.id ? { ...family, description_override: updated.description_override } : family
      )
    );
  }

  async function saveTray(familyId: string, trayId: string, nextDisplayName: string) {
    const updated = await setTrayDisplayName(trayId, nextDisplayName);
    setFamilies((previous) =>
      previous.map((family) =>
        family.id === familyId
          ? { ...family, trays: family.trays.map((tray) => (tray.id === updated.id ? updated : tray)) }
          : family
      )
    );
  }

  return (
    <section className="settings-section">
      <h2>Families &amp; trays</h2>
      <p className="settings-section-lede">
        The master catalog’s product families and their trays. The hierarchy comes from the
        catalog file and can’t be changed here, but each display name can — it’s what users
        see everywhere, and re-seeding the catalog keeps it. Clearing a display name goes back
        to the catalog name.
      </p>
      <p className="settings-section-lede">
        Each family’s classifier description is what the chat assistant reads when it works out
        which family a question is about. It’s generated from the family’s trays and parts. Edit
        it to add the product names reps use; the next question uses your wording, and
        re-seeding the catalog keeps it.
      </p>

      {loadError && <p className="alert" role="alert">{loadError}</p>}

      {loaded && !loadError && families.length === 0 ? (
        <div className="empty-state">No product families yet — seed the master catalog first.</div>
      ) : (
        families.length > 0 && (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Display name</th>
                  <th>Catalog name</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {families.map((family) => (
                  <Fragment key={family.id}>
                    <DisplayNameRow
                      catalogName={family.name}
                      displayName={family.display_name}
                      kind="family"
                      onSave={(nextDisplayName) => saveFamily(family.id, nextDisplayName)}
                    />
                    <FamilyDescriptionRow
                      familyId={family.id}
                      familyName={family.display_name}
                      generatedDescription={family.generated_description}
                      descriptionOverride={family.description_override}
                      onSave={(nextDescription) => saveDescription(family.id, nextDescription)}
                    />
                    {family.trays.map((tray) => (
                      <DisplayNameRow
                        key={tray.id}
                        catalogName={tray.name}
                        displayName={tray.display_name}
                        kind="tray"
                        onSave={(nextDisplayName) => saveTray(family.id, tray.id, nextDisplayName)}
                      />
                    ))}
                  </Fragment>
                ))}
              </tbody>
            </table>
          </div>
        )
      )}
    </section>
  );
}
