#!/usr/bin/env python3
"""Check the static project page before previewing or deploying it.

Checks every HTML page under website/:
  * local href/src targets exist and are relative (no leading "/"), so the
    site works under a repository sub-path such as /X-STAR/;
  * in-page anchors (#id) resolve;
  * every <img> has non-empty alt text and width/height, and the declared
    aspect ratio matches the image file (no stretching);
  * no served text file (HTML, CSS, JS, JSON) contains machine paths or
    credentials-like strings.

With --external, also requests each external link (HEAD, then GET) and reports
non-2xx/3xx answers. Exit status 1 if any check fails.
"""

import argparse
import re
import sys
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

SITE = Path(__file__).resolve().parents[1] / "docs"
SERVED = {".html", ".css", ".js", ".json", ".svg", ".txt", ".xml"}
FORBIDDEN = [re.compile(p) for p in (r"/data\d*/", r"/storage/", r"/home/", r"\.venv",
                                     r"file://", r"ghp_[A-Za-z0-9]{10,}", r"hf_[A-Za-z0-9]{10,}")]


class Page(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links, self.ids, self.images = [], set(), []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.add(a["id"])
        for key in ("href", "src"):
            if a.get(key):
                self.links.append((tag, a[key]))
        if a.get("srcset"):
            self.links += [(tag, part.strip().split()[0]) for part in a["srcset"].split(",")]
        if tag == "img":
            self.images.append(a)


def image_size(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.size


def check_external(url):
    for method in ("HEAD", "GET"):
        req = urllib.request.Request(url, method=method, headers={"User-Agent": "xstar-site-check"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.status
        except Exception as exc:  # noqa: BLE001 - report any failure
            status = getattr(exc, "code", None) or str(exc)
    return status


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--external", action="store_true", help="also request external links")
    args = ap.parse_args()
    problems, external = [], set()

    for page_path in sorted(p for p in SITE.rglob("*.html") if "tools" not in p.parts):
        rel = page_path.relative_to(SITE)
        page = Page()
        page.feed(page_path.read_text(encoding="utf-8"))
        for tag, link in page.links:
            parts = urlsplit(link)
            if parts.scheme in ("http", "https"):
                external.add(link)
                continue
            if parts.scheme == "data":       # inline resources (e.g. the empty favicon)
                continue
            if parts.scheme or link.startswith("//"):
                problems.append(f"{rel}: unexpected link scheme {link!r}")
                continue
            if link.startswith("/"):
                problems.append(f"{rel}: absolute path {link!r} breaks under a repository sub-path")
                continue
            if parts.path:
                target = (page_path.parent / unquote(parts.path)).resolve()
                if not target.exists():
                    problems.append(f"{rel}: missing local target {link!r}")
            elif parts.fragment and parts.fragment not in page.ids:
                problems.append(f"{rel}: anchor #{parts.fragment} not found")
        for img in page.images:
            src = img.get("src", "")
            if not (img.get("alt") or "").strip():
                problems.append(f"{rel}: <img src={src!r}> has no alt text")
            if not (img.get("width") and img.get("height")):
                problems.append(f"{rel}: <img src={src!r}> lacks width/height")
                continue
            file = page_path.parent / src
            if file.exists():
                w, h = image_size(file)
                declared = int(img["width"]) / int(img["height"])
                if abs(declared - w / h) > 0.01:
                    problems.append(f"{rel}: {src} declared {img['width']}x{img['height']}, file is {w}x{h}")

    for f in SITE.rglob("*"):
        if f.is_file() and f.suffix in SERVED and "tools" not in f.relative_to(SITE).parts:
            text = f.read_text(encoding="utf-8", errors="replace")
            for pat in FORBIDDEN:
                if pat.search(text):
                    problems.append(f"{f.relative_to(SITE)}: contains forbidden pattern {pat.pattern!r}")

    if args.external:
        for url in sorted(external):
            status = check_external(url)
            ok = isinstance(status, int) and 200 <= status < 400
            print(f"{'ok ' if ok else 'BAD'} {status} {url}")
            if not ok:
                problems.append(f"external link {url} answered {status}")
    else:
        print(f"{len(external)} external links not requested (use --external)")

    for p in problems:
        print("FAIL", p)
    print("site check:", "FAILED" if problems else "passed")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
