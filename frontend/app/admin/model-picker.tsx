"use client";

// A searchable model + reasoning-effort picker for one slot (issue #24).
// Options come from OpenRouter's catalog, already filtered server-side to
// models whose metadata covers the slot's needs, with dated ids preferred.
// A slot that accepts decision models (issue #23) also lists those, and a
// decision model takes two thresholds instead of a reasoning effort.
// Shared by the Settings section and the Eval tab's rerun overrides.
import { useEffect, useMemo, useState } from "react";
import { listSlotModels } from "../../lib/model-assignments/api";
import {
  DEFAULT_FAMILY_CONFIDENCE_THRESHOLD,
  DEFAULT_QUESTION_TYPE_MARGIN,
  REASONING_EFFORTS,
  type CatalogModel,
  type ModelChoice,
  type ReasoningEffort,
} from "../../lib/model-assignments/types";

function formatPrice(value: number | null): string {
  return value === null ? "?" : `$${value.toFixed(2)}`;
}

export function priceLabel(model: CatalogModel): string {
  return `${formatPrice(model.prompt_price_per_million)} in / ${formatPrice(
    model.completion_price_per_million
  )} out per M tokens`;
}

export function ModelPicker({
  slot,
  value,
  defaultReasoningEffort,
  onChange,
  unsetLabel,
}: {
  slot: string;
  value: ModelChoice | null;
  // Pre-filled when a reasoning model is first picked.
  defaultReasoningEffort: ReasoningEffort;
  onChange: (choice: ModelChoice | null) => void;
  // When set, a first option meaning "no choice" (value null), e.g. the
  // rerun panel's "Production setting".
  unsetLabel?: string;
}) {
  const [models, setModels] = useState<CatalogModel[] | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [search, setSearch] = useState("");

  useEffect(() => {
    listSlotModels(slot)
      .then((fetchedModels) => {
        setModels(fetchedModels);
        setLoadError(null);
      })
      .catch((caughtError) =>
        setLoadError(caughtError instanceof Error ? caughtError.message : "failed to load the model catalog")
      );
  }, [slot]);

  const selectedModel = models?.find((model) => model.model_id === value?.model_id) ?? null;

  const visibleModels = useMemo(() => {
    const query = search.trim().toLowerCase();
    const matching = (models ?? []).filter(
      (model) =>
        !query || model.model_id.toLowerCase().includes(query) || model.name.toLowerCase().includes(query)
    );
    // Keep the current choice visible while searching for something else.
    if (selectedModel && !matching.includes(selectedModel)) matching.unshift(selectedModel);
    return matching;
  }, [models, search, selectedModel]);

  function selectModel(modelId: string) {
    if (!modelId) {
      onChange(null);
      return;
    }
    const model = models?.find((candidate) => candidate.model_id === modelId);
    if (model?.model_kind === "decision" || (!model && value?.model_kind === "decision")) {
      // A decision model takes thresholds instead of an effort, kept when
      // switching between decision models.
      onChange({
        model_id: modelId,
        reasoning_effort: null,
        model_kind: "decision",
        family_confidence_threshold: value?.family_confidence_threshold ?? DEFAULT_FAMILY_CONFIDENCE_THRESHOLD,
        question_type_margin: value?.question_type_margin ?? DEFAULT_QUESTION_TYPE_MARGIN,
      });
      return;
    }
    // A model without reasoning support takes no effort at all.
    const reasoningEffort = model?.supports_reasoning
      ? value?.reasoning_effort ?? defaultReasoningEffort
      : null;
    onChange({ model_id: modelId, reasoning_effort: reasoningEffort });
  }

  function setThreshold(field: "family_confidence_threshold" | "question_type_margin", text: string) {
    if (!value) return;
    const parsed = Number.parseFloat(text);
    // Cleared or unreadable means the default, which the server fills in.
    onChange({ ...value, [field]: Number.isNaN(parsed) ? null : parsed });
  }

  function modelOption(model: CatalogModel) {
    return (
      <option key={model.model_id} value={model.model_id}>
        {model.name} — {model.model_id} ·{" "}
        {model.model_kind === "decision"
          ? `${formatPrice(model.prompt_price_per_million)} in per M tokens`
          : priceLabel(model)}
      </option>
    );
  }

  const chatModels = visibleModels.filter((model) => model.model_kind !== "decision");
  const decisionModels = visibleModels.filter((model) => model.model_kind === "decision");

  if (loadError) return <p className="alert" role="alert">{loadError}</p>;
  if (models === null) return <p className="model-picker-loading">Loading the model catalog…</p>;

  return (
    <div className="model-picker">
      <input
        type="search"
        value={search}
        onChange={(event) => setSearch(event.target.value)}
        placeholder={`Search ${models.length} compatible models…`}
        aria-label="Search models"
      />
      <select
        className="model-picker-list"
        size={Math.min(8, visibleModels.length + (unsetLabel ? 1 : 0)) || 2}
        value={value?.model_id ?? ""}
        onChange={(event) => selectModel(event.target.value)}
        aria-label="Model"
      >
        {unsetLabel && <option value="">{unsetLabel}</option>}
        {/* A saved id the catalog no longer lists (or lists only as an alias) still shows. */}
        {value && !selectedModel && <option value={value.model_id}>{value.model_id}</option>}
        {decisionModels.length === 0 ? (
          chatModels.map(modelOption)
        ) : (
          <>
            <optgroup label="Chat models">{chatModels.map(modelOption)}</optgroup>
            {/* Issue #23: answer intent classification as Choices, with probabilities. */}
            <optgroup label="Decision models">{decisionModels.map(modelOption)}</optgroup>
          </>
        )}
      </select>
      {visibleModels.length === 0 && <p className="model-picker-empty">No compatible model matches.</p>}
      {value?.model_kind === "decision" && (
        <div>
          <label className="model-picker-effort">
            Family confidence threshold
            <input
              type="number"
              min={0.05}
              max={1}
              step={0.05}
              value={value.family_confidence_threshold ?? ""}
              placeholder={String(DEFAULT_FAMILY_CONFIDENCE_THRESHOLD)}
              onChange={(event) => setThreshold("family_confidence_threshold", event.target.value)}
            />
          </label>
          <label className="model-picker-effort">
            Question-type margin
            <input
              type="number"
              min={0}
              max={0.95}
              step={0.05}
              value={value.question_type_margin ?? ""}
              placeholder={String(DEFAULT_QUESTION_TYPE_MARGIN)}
              onChange={(event) => setThreshold("question_type_margin", event.target.value)}
            />
          </label>
          <p className="model-picker-empty">
            Below the threshold the rep is asked which family; two question types within the
            margin are both offered. Starting values: tune them on the intent eval.
          </p>
        </div>
      )}
      {value && selectedModel?.supports_reasoning && (
        <label className="model-picker-effort">
          Reasoning effort
          <select
            value={value.reasoning_effort ?? ""}
            onChange={(event) =>
              onChange({
                ...value,
                reasoning_effort: (event.target.value || null) as ReasoningEffort | null,
              })
            }
          >
            <option value="">Model default</option>
            {REASONING_EFFORTS.map((effort) => (
              <option key={effort} value={effort}>
                {effort}
              </option>
            ))}
          </select>
        </label>
      )}
    </div>
  );
}
