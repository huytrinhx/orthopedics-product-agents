"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { useAuth } from "../../lib/auth-context";
import { listUsers, setUserActive } from "../../lib/users/api";
import type { AuthUser } from "../../lib/auth/types";

// A single click flips the flag immediately -- no confirm step, since
// unlike a delete this is trivially reversible (click it again).
function ActiveToggle({
  user,
  toggling,
  onToggle,
}: {
  user: AuthUser;
  toggling: boolean;
  onToggle: (user: AuthUser) => void;
}) {
  if (user.is_admin) {
    // Read-only: is_admin comes from ADMIN_EMAILS (granted at signup or on a
    // later login) and backend/auth/dependencies.py's require_chat_access
    // never even consults is_active for an admin, so there's nothing to
    // toggle here.
    return <span className="users-readonly">—</span>;
  }
  return (
    <button type="button" className="btn-text" disabled={toggling} onClick={() => onToggle(user)}>
      {toggling ? "Saving…" : user.is_active ? "Disable" : "Enable"}
    </button>
  );
}

export default function UsersPage() {
  const { user, loading } = useAuth();
  const [users, setUsers] = useState<AuthUser[]>([]);
  const [listError, setListError] = useState<string | null>(null);
  const [togglingId, setTogglingId] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setUsers(await listUsers());
      setListError(null);
    } catch (err) {
      setListError(err instanceof Error ? err.message : "failed to load users");
    }
  }, []);

  useEffect(() => {
    if (!user?.is_admin) return;
    refresh();
  }, [user, refresh]);

  async function handleToggle(target: AuthUser) {
    setTogglingId(target.id);
    try {
      const updated = await setUserActive(target.id, !target.is_active);
      setUsers((prev) => prev.map((u) => (u.id === target.id ? updated : u)));
    } catch (err) {
      setListError(err instanceof Error ? err.message : "failed to update user");
    } finally {
      setTogglingId(null);
    }
  }

  if (loading) return <main className="page-wide"><p>Loading…</p></main>;

  if (!user || !user.is_admin) {
    return (
      <main className="page-wide">
        <h1>Users</h1>
        <p>This page is admin-only.</p>
        <Link href="/">Back home</Link>
      </main>
    );
  }

  return (
    <main className="page-wide">
      <span className="eyebrow">Access control</span>
      <h1>Users</h1>
      <p className="lede">
        A new signup can&rsquo;t chat until an admin enables them here. Admin accounts (set via
        the ADMIN_EMAILS allowlist) are always active and can&rsquo;t be disabled.
      </p>

      {listError && <p className="alert" role="alert">{listError}</p>}

      {users.length === 0 ? (
        <div className="empty-state">No users yet.</div>
      ) : (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Email</th>
                <th>Role</th>
                <th>Status</th>
                <th>Joined</th>
                <th>Last logged in at</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {users.map((u) => (
                <tr key={u.id}>
                  <td>{u.email}</td>
                  <td>{u.is_admin ? <span className="admin-tag">Admin</span> : "User"}</td>
                  <td>
                    <span className={`badge badge-${u.is_active ? "enabled" : "disabled"}`}>
                      {u.is_active ? "Enabled" : "Pending"}
                    </span>
                  </td>
                  <td>{new Date(u.created_at).toLocaleString()}</td>
                  <td>{u.last_login_at ? new Date(u.last_login_at).toLocaleString() : "Never"}</td>
                  <td>
                    <ActiveToggle user={u} toggling={togglingId === u.id} onToggle={handleToggle} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </main>
  );
}
