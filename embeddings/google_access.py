# google_access.py
import os, re, requests
from typing import List, Tuple

class UnsupportedGoogleUrlError(Exception): ...
class NoPublicAccessError(Exception): ...

GOOGLE_DOCS_HOST = "docs.google.com"
DOCS_PATTERNS = (
    r"https?://docs\.google\.com/document/d/([a-zA-Z0-9_-]+)",
    r"https?://docs\.google\.com/presentation/d/([a-zA-Z0-9_-]+)",
)

EXPORT_SPEC = {
    "document": ("docx", ".docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    "presentation": ("pptx", ".pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation"),
}

def _infer_app_and_id(url: str) -> Tuple[str, str]:
    for pat in DOCS_PATTERNS:
        m = re.match(pat, url)
        if m:
            file_id = m.group(1)
            if "/document/" in url: return "document", file_id
            if "/presentation/" in url: return "presentation", file_id
    raise UnsupportedGoogleUrlError("Only public Google Docs & Slides are supported (not Sheets/Drive viewer).")

def _export_url(app: str, file_id: str, fmt: str) -> str:
    return f"https://{GOOGLE_DOCS_HOST}/{app}/d/{file_id}/export?format={fmt}"

def _looks_like_access_wall(text: str) -> bool:
    tl = text.lower()
    return any(x in tl for x in [
        "you need access","request access","sign in","you don't have access",
        "this file is in owner's trash","access denied"
    ])

def download_public_google_files(urls: List[str], out_dir: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0"})
    saved: List[str] = []

    for url in urls:
        app, file_id = _infer_app_and_id(url)
        fmt, ext, expected_mime = EXPORT_SPEC[app]
        exp_url = _export_url(app, file_id, fmt)

        r = s.get(exp_url, timeout=60, stream=True, allow_redirects=True)
        if r.status_code in (401, 403):
            raise NoPublicAccessError(f"No public access for: {url}")

        ctype = r.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if ctype == "text/html":
            head = next(r.iter_content(2048), b"")
            if _looks_like_access_wall(head.decode("utf-8", errors="ignore")):
                raise NoPublicAccessError(f"No public access for: {url}")
            raise NoPublicAccessError(f"Export did not return a file (HTML response). Check sharing: {url}")

        # (Optional) strict mime check:
        # if expected_mime not in ctype: raise NoPublicAccessError(f"Unexpected content type {ctype}")

        local_path = os.path.join(out_dir, f"{file_id}{ext}")
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(1024 * 128):
                if chunk: f.write(chunk)
        saved.append(local_path)
    return saved
