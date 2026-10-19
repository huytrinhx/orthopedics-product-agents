// `name` is the master catalog's name (read-only, the seed's key);
// `display_name` is what users see, admin-editable on the Settings page and
// never overwritten by a re-seed (issue #15).
export interface Tray {
  id: string;
  name: string;
  display_name: string;
}

export interface ProductFamilySummary {
  id: string;
  name: string;
  display_name: string;
}

// `generated_description` is what the intent classifiers are told about the
// family, generated from the catalog (issue #30); `description_override` is
// an admin's replacement for it, or null to use the generated one.
export interface ProductFamily extends ProductFamilySummary {
  trays: Tray[];
  generated_description: string;
  description_override: string | null;
}

export interface FamilyDescription {
  id: string;
  description_override: string | null;
}
