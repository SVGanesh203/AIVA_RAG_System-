#!/usr/bin/env python3
"""
Export a Confluence (IntouchCX Kbase) page:
- Fetch page storage (XHTML) + metadata
- Download all attachments (images + other files)
- Rewrite <ac:image> to point to local images/
- Convert entire page to Markdown (page.md); tables are inlined as Markdown
- Also write: page_storage.html, page_local.html, metadata.json, images/, attachments/

Auth: Bearer PAT (same as your working script)
TLS: VERIFY_SSL flag (set False for corp cert chains)

Usage: run and follow prompts to pick a space and page.
"""

import os
import re
import json
import time
import pathlib
import requests
import pandas as pd
from typing import Dict, Any, List, Tuple
from bs4 import BeautifulSoup

# Try Markdown converters (prefer Pandoc for fidelity)
_MD_BACKEND = None
try:
    import pypandoc  # type: ignore
    _MD_BACKEND = "pandoc"
except Exception:
    try:
        from markdownify import markdownify as _md_convert  # type: ignore
        _MD_BACKEND = "markdownify"
    except Exception:
        _MD_BACKEND = None

# ========= YOUR CONFIG =========
BASE_URL = "https://kbase.intouchcx.com"
ACCESS_TOKEN = "REPLACE_ME"   # <--- placeholder, add your PAT at runtime

VERIFY_SSL = False                 # False for internal/self-signed certs
PAGE_LIMIT = 200
# ===============================

API_ROOT = f"{BASE_URL.rstrip('/')}/rest/api"
WEB_ROOT = BASE_URL.rstrip('/')

session = requests.Session()
session.headers.update({
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "Accept": "application/json"
})
session.verify = VERIFY_SSL

def _err(msg: str):
    print(f"❌ {msg}")

def _ok(msg: str):
    print(f"✅ {msg}")

def _info(msg: str):
    print(f"• {msg}")

def sanitize_slug(s: str) -> str:
    s = re.sub(r"[^\w]+", "_", (s or "").strip().lower())  # non-word → _
    return re.sub(r"_{2,}", "_", s).strip("_") or "page"

def fetch_spaces() -> List[Tuple[str, str, str]]:
    url = f"{API_ROOT}/space"
    params = {"limit": PAGE_LIMIT}
    spaces = []
    while True:
        resp = session.get(url, params=params)
        if resp.status_code != 200:
            _err(f"Fetching spaces: {resp.status_code} - {resp.text}")
            break
        data = resp.json()
        for s in data.get("results", []):
            key = s.get("key")
            name = s.get("name")
            homepage = s.get("_links", {}).get("webui")
            homepage_url = f"{WEB_ROOT}{homepage}" if homepage else ""
            spaces.append((key, name, homepage_url))
        next_link = data.get("_links", {}).get("next")
        if not next_link:
            break
        url = f"{BASE_URL.rstrip('/')}{next_link}"
        params = {}
    return spaces

def fetch_pages(space_key: str) -> List[Tuple[str, str, str]]:
    url = f"{API_ROOT}/content"
    params = {"spaceKey": space_key, "type": "page", "limit": PAGE_LIMIT}
    pages = []
    while True:
        resp = session.get(url, params=params)
        if resp.status_code != 200:
            _err(f"Fetching pages for {space_key}: {resp.status_code} - {resp.text}")
            break
        data = resp.json()
        for item in data.get("results", []):
            pid = item.get("id")
            title = item.get("title")
            webui = item.get("_links", {}).get("webui")
            url = f"{WEB_ROOT}{webui}" if webui else f"{WEB_ROOT}/pages/{pid}"
            pages.append((pid, title, url))
        next_link = data.get("_links", {}).get("next")
        if not next_link:
            break
        url = f"{BASE_URL.rstrip('/')}{next_link}"
        params = {}
    return pages

def get_page(page_id: str) -> Dict[str, Any]:
    url = f"{API_ROOT}/content/{page_id}"
    params = {
        "expand": ",".join([
            "body.storage",
            "body.export_view",          # <-- add this
            "version",
            "metadata.labels",
            "ancestors",
            "space"
        ])
    }
    r = session.get(url, params=params)
    r.raise_for_status()
    return r.json()

def confluence_html_to_markdown(html: str) -> str:
    """
    Convert Confluence-rendered HTML to Markdown.
    Tries pypandoc first; falls back to markdownify; last resort: plain text.
    """
    html = html or ""
    # Try pypandoc if available
    try:
        import pypandoc
        return pypandoc.convert_text(html, "gfm", format="html")  # GitHub-flavored MD
    except Exception:
        pass

    # Fallback: markdownify
    try:
        from markdownify import markdownify as md
        # strip=["style","script"] keeps text; heading_style="ATX" gives #, ##, …
        return md(
            html,
            heading_style="ATX",
            strip=["style", "script"]
        ).strip()
    except Exception:
        # Last resort: plain text via BeautifulSoup
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")  # use lxml for robust parsing
        for t in soup(["script", "style"]): t.decompose()
        return soup.get_text("\n", strip=True)

def get_attachments(page_id: str) -> List[Dict[str, Any]]:
    url = f"{API_ROOT}/content/{page_id}/child/attachment"
    params = {"limit": 200, "start": 0, "expand": "metadata"}
    attachments = []
    while True:
        r = session.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        attachments.extend(data.get("results", []))
        if data.get("_links", {}).get("next"):
            url = f"{BASE_URL.rstrip('/')}{data['_links']['next']}"
            params = {}
        else:
            break
    return attachments

def download_attachment_by_link(download_href: str, dest_path: str):
    url = download_href
    if download_href.startswith("/"):
        url = f"{BASE_URL.rstrip('/')}{download_href}"
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with session.get(url, stream=True, timeout=180) as res:
        res.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in res.iter_content(8192):
                if chunk:
                    f.write(chunk)
    return dest_path

def storage_html_to_text(xhtml: str) -> str:
    soup = BeautifulSoup(xhtml or "", "html.parser")
    for tag in soup(["script", "style"]): tag.decompose()
    for a in soup.find_all("a", href=True):
        text = a.get_text(" ", strip=True)
        href = a["href"]
        a.replace_with(f"{text} ({href})" if text else href)
    for blk in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "li", "pre"]):
        if not blk.text.endswith("\n"):
            blk.append("\n")
    text = soup.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _dataframes_from_table_tag(table_tag) -> List[pd.DataFrame]:
    # pandas.read_html expects a full HTML doc or a table string
    try:
        dfs = pd.read_html(str(table_tag))
        return dfs
    except Exception:
        return []

def _replace_tables_with_markdown(html: str) -> str:
    """
    Replace each <table> with a <pre class="md-table">MARKDOWN</pre>
    so later HTML->MD conversion keeps the markdown as-is.
    """
    soup = BeautifulSoup(html, "html.parser")
    changed = False
    for table in soup.find_all("table"):
        dfs = _dataframes_from_table_tag(table)
        if not dfs:
            continue
        # join multiple tables in one tag if present
        md_parts = []
        for df in dfs:
            md_parts.append(df.to_markdown(index=False))
        md_text = "\n\n".join(md_parts)

        pre = soup.new_tag("pre")
        pre["class"] = "md-table"
        pre.string = md_text  # keep raw markdown text
        table.replace_with(pre)
        changed = True
    return str(soup) if changed else html

def html_to_markdown(html: str) -> str:
    """
    Convert HTML to Markdown using the best available backend.
    - If Pandoc (pypandoc) is available -> GitHub-Flavored Markdown
    - Else fallback to markdownify
    """
    if _MD_BACKEND == "pandoc":
        try:
            return pypandoc.convert_text(
                html,
                "gfm",
                format="html",
                extra_args=[
                    "--wrap=none",
                    "--atx-headers"
                ]
            )
        except Exception as e:
            _info(f"Pandoc conversion failed ({e}); falling back to markdownify…")

    if _MD_BACKEND == "markdownify":
        return _md_convert(
            html,
            heading_style="ATX",
            bullets="-",
            code_language_detection=False,
            strip=["style", "script"]
        )

    # Last resort: dump plain text
    _info("No Markdown backend found; writing plain text instead.")
    return storage_html_to_text(html)

def rewrite_images_and_download(xhtml: str, attachments: List[Dict[str, Any]], images_dir: str) -> str:
    """
    Find <ac:image> with <ri:attachment ri:filename="..."> or <ri:url ri:value="...">
    Download files to images_dir and rewrite <ac:image> → <img src="images/....">
    Returns modified HTML (string).
    """
    os.makedirs(images_dir, exist_ok=True)

    # index attachments by filename for quick lookup
    by_filename = {}
    for att in attachments:
        fname = att.get("title") or att.get("metadata", {}).get("mediaType") or ""
        if fname:
            by_filename[fname] = att

    # parse as XML to keep namespaces
    soup = BeautifulSoup(xhtml, "xml")

    for ac_img in soup.find_all("ac:image"):
        local_path = None

        # 1) ri:attachment
        ri_att = ac_img.find("ri:attachment")
        if ri_att and ri_att.has_attr("ri:filename"):
            fname = ri_att["ri:filename"]
            att = by_filename.get(fname)
            if att and att.get("_links", {}).get("download"):
                dl = att["_links"]["download"]
                dest = os.path.join(images_dir, fname)
                try:
                    download_attachment_by_link(dl, dest)
                    local_path = dest
                except Exception as e:
                    print(f"   • image '{fname}' download failed: {e}")

        # 2) ri:url (external or absolute)
        if local_path is None:
            ri_url = ac_img.find("ri:url")
            if ri_url and ri_url.has_attr("ri:value"):
                href = ri_url["ri:value"]
                fname = sanitize_slug(os.path.basename(href)) or "image"
                dest = os.path.join(images_dir, fname)
                try:
                    download_attachment_by_link(href, dest)
                    local_path = dest
                except Exception as e:
                    print(f"   • external image download failed: {href} -> {e}")

        # Replace the whole <ac:image> with standard <img>
        if local_path:
            img = soup.new_tag("img")
            img["src"] = os.path.join(os.path.basename(images_dir), os.path.basename(local_path))
            if ac_img.has_attr("ac:alt"):
                img["alt"] = ac_img["ac:alt"]
            ac_img.replace_with(img)

    return str(soup)

def export_page(page_id: str, base_out_dir: str):
    page = get_page(page_id)
    title = page.get("title", f"page-{page_id}")
    slug = sanitize_slug(title)
    out_dir = os.path.join(base_out_dir, f"{slug}_{page_id}")
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n📝 Exporting: {title} (id={page_id})")
    print(f"   → {out_dir}")

    # ---- body: storage (XHTML) + export_view (rendered HTML) ----
    storage = (page.get("body") or {}).get("storage") or {}
    xhtml = storage.get("value", "")

    export_view = (page.get("body") or {}).get("export_view") or {}
    rendered_html = export_view.get("value") or ""

    # 1) save raw storage XHTML (debug/reference)
    raw_path = os.path.join(out_dir, "page_storage.html")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(xhtml)
    _info("saved storage HTML")

    # also save rendered HTML so you can diff if needed
    rendered_path = os.path.join(out_dir, "page_rendered.html")
    with open(rendered_path, "w", encoding="utf-8") as f:
        f.write(rendered_html or "<!-- no export_view available -->")
    _info("saved rendered HTML (export_view)")

    # 2) attachments (download everything)
    atts = get_attachments(page_id)
    _info(f"{len(atts)} attachment(s)")
    att_dir = os.path.join(out_dir, "attachments")
    os.makedirs(att_dir, exist_ok=True)
    for a in atts:
        a_title = a.get("title") or "file"
        dl = a.get("_links", {}).get("download")
        if not dl:
            continue
        dest = os.path.join(att_dir, a_title)
        try:
            download_attachment_by_link(dl, dest)
        except Exception as e:
            print(f"     - attachment '{a_title}' failed: {e}")

    # 3) rewrite images inside the STORAGE HTML (so local HTML has working <img>)
    img_dir = os.path.join(out_dir, "images")
    local_html = rewrite_images_and_download(xhtml, atts, img_dir)

    # also rewrite tables in LOCAL HTML (so you have pre-wrapped MD blocks)
    # assumes you already have this helper; if not, either define it or remove this line.
    local_html_with_table_md = _replace_tables_with_markdown(local_html)

    local_path = os.path.join(out_dir, "page_local.html")
    with open(local_path, "w", encoding="utf-8") as f:
        f.write(local_html_with_table_md)
    _info("saved page_local.html (images + tables prepared)")

    # 4) full-page Markdown (PRIMARY): use RENDERED HTML
    #    (This expands Confluence macros; if missing, fall back to storage HTML.)
    markdown_body = confluence_html_to_markdown(rendered_html if rendered_html else xhtml)

    # Optional: if you want to try to rewrite image URLs in markdown_body to your local images/,
    # you can add a small post-process here (regex replace on src= or direct links).

    # 4b) YAML front matter
    labels = [(l or {}).get("name") for l in (page.get("metadata") or {}).get("labels", {}).get("results", [])]
    safe_title = (title or "").replace('"', "'")
    labels_yaml = json.dumps(labels, ensure_ascii=False)
    fm = [
        "---",
        f'title: "{safe_title}"',
        f"space: {(page.get('space') or {}).get('key')}",
        f"version: {(page.get('version') or {}).get('number')}",
        f"labels: {labels_yaml}",
        f"web_url: {WEB_ROOT}{(page.get('_links') or {}).get('webui','')}",
        f"exported_at: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "---",
        ""
    ]

    md_full = "\n".join(fm) + markdown_body + "\n"
    md_path = os.path.join(out_dir, "page.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_full)
    _ok("wrote page.md")

    # 5) metadata.json (for programmatic uses)
    meta = {
        "page_id": page_id,
        "title": page.get("title"),
        "space": (page.get("space") or {}).get("key"),
        "version": (page.get("version") or {}).get("number"),
        "labels": labels,
        "ancestors": [{"id": a.get("id"), "title": a.get("title")} for a in page.get("ancestors", [])],
        "web_url": f"{WEB_ROOT}{(page.get('_links') or {}).get('webui','')}",
        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    _info("saved metadata.json")

def main():
    print("=" * 80)
    print("📚 IntouchCX Kbase – Export Page → Markdown (images + tables inline)")
    print("=" * 80)

    if _MD_BACKEND is None:
        _info("No Markdown converter found. Install `pypandoc` (best) or `markdownify` for better output.")

    print("\n🔍 Fetching spaces…")
    spaces = fetch_spaces()
    if not spaces:
        _err("No spaces found or access denied.")
        return
    for i, (key, name, url) in enumerate(spaces, 1):
        print(f"{i:2d}. [{key}] {name}")

    space_key = input("\nEnter SPACE KEY: ").strip()
    if not any(s[0] == space_key for s in spaces):
        _err(f"Space '{space_key}' not found.")
        return

    print(f"\n📄 Listing pages in [{space_key}] …")
    pages = fetch_pages(space_key)
    if not pages:
        _err("No pages found in that space (or access denied).")
        return
    for i, (pid, title, url) in enumerate(pages, 1):
        print(f"{i:3d}. {title}  ({pid})")

    choice = input("\nEnter PAGE ID (exact) to export: ").strip()
    if not any(p[0] == choice for p in pages):
        _err(f"Page ID '{choice}' not in list.")
        return

    out_root = os.path.join(os.getcwd(), "kbase_exports")
    os.makedirs(out_root, exist_ok=True)
    export_page(choice, out_root)

    print("\n✅ Done!  (See kbase_exports/<slug>-<id>/page.md)")
    if _MD_BACKEND:
        print(f"   Markdown engine: {_MD_BACKEND}")
    else:
        print("   Markdown engine: plain-text fallback (install pypandoc or markdownify)")

if __name__ == "__main__":
    main()
