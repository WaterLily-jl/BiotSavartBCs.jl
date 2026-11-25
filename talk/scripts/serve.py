"""Serve the deck locally, with HTTP Range support so videos can seek.

    python scripts/serve.py          # http://localhost:8000
    python scripts/serve.py 8080

Python's built-in `http.server` ignores Range requests, so a browser cannot
jump into the middle of a video it has not downloaded yet; trimmed clips
(data-start / data-end) then stall at 0 s. GitHub Pages supports ranges, so
this only matters locally.
"""
import os
import re
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class RangeHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        rng = self.headers.get("Range")
        path = self.translate_path(self.path)
        m = re.match(r"bytes=(\d*)-(\d*)$", rng or "")
        if not m or not os.path.isfile(path):
            return super().send_head()
        size = os.path.getsize(path)
        start = int(m[1]) if m[1] else max(0, size - int(m[2] or 0))
        end = min(int(m[2]), size - 1) if m[1] and m[2] else size - 1
        if start >= size:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{size}")
            self.end_headers()
            return None
        f = open(path, "rb")
        f.seek(start)
        self._remaining = end - start + 1
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(self._remaining))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        return f

    def copyfile(self, source, outputfile):
        remaining = getattr(self, "_remaining", None)
        if remaining is None:
            return super().copyfile(source, outputfile)
        try:
            while remaining > 0:
                chunk = source.read(min(1 << 16, remaining))
                if not chunk:
                    break
                outputfile.write(chunk)
                remaining -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser often drops a range request once it has what it needs
        finally:
            self._remaining = None

    def end_headers(self):
        if not self.headers.get("Range"):
            self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")   # always pick up edits while rehearsing
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    handler = partial(RangeHandler, directory=str(ROOT))
    print(f"Serving {ROOT} at http://localhost:{port}")
    ThreadingHTTPServer(("", port), handler).serve_forever()
