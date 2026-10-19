"""Upload refuses a filename another document already has (issue #31), and
tests share one database for the whole run, so a test that uploads through
the API gives its file a name no other test uses."""
import uuid
from pathlib import PurePath


def unique_filename(filename: str) -> str:
    """`filename` with a random suffix before its extension:
    "brochure.txt" becomes "brochure-1a2b3c4d.txt"."""
    path = PurePath(filename)
    return f"{path.stem}-{uuid.uuid4().hex[:8]}{path.suffix}"
