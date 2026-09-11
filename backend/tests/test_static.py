"""api.main.CleanUrlStaticFiles: `next build`'s static export writes each
page as a flat file (out/users.html, not out/users/index.html), so a plain
StaticFiles(html=True) mount 404s a browser's actual "/users" request --
only in-app client-side <Link> navigation avoided the gap, which is why it
went unnoticed until a direct/hard navigation to any non-root page 404'd in
production. Builds a throwaway static dir instead of depending on
frontend/out (CI's backend job never runs `npm run build`, and this
shouldn't need to).
"""
from starlette.applications import Starlette
from starlette.testclient import TestClient

from api.main import CleanUrlStaticFiles


def _client(tmp_path) -> TestClient:
    (tmp_path / "users.html").write_text("<title>Users</title>")
    (tmp_path / "index.html").write_text("<title>Home</title>")
    app = Starlette()
    app.mount("/", CleanUrlStaticFiles(directory=tmp_path, html=True), name="frontend")
    return TestClient(app)


def test_extensionless_path_resolves_to_matching_html_file(tmp_path):
    res = _client(tmp_path).get("/users")
    assert res.status_code == 200
    assert "Users" in res.text


def test_root_still_serves_index(tmp_path):
    res = _client(tmp_path).get("/")
    assert res.status_code == 200
    assert "Home" in res.text


def test_path_with_no_matching_html_file_still_404s(tmp_path):
    res = _client(tmp_path).get("/nonexistent")
    assert res.status_code == 404


def test_path_with_an_extension_is_not_mangled(tmp_path):
    (tmp_path / "logo.png").write_bytes(b"not-a-real-png")
    res = _client(tmp_path).get("/logo.png")
    assert res.status_code == 200
    assert res.content == b"not-a-real-png"
