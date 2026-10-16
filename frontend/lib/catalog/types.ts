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

export interface ProductFamily extends ProductFamilySummary {
  trays: Tray[];
}
