"""Full-page screenshots of every app page into docs/screenshots/, using headless Chrome.

Usage:  make screenshots                      (after `make backtest`)
        make screenshots QUERY="stress=70&week=5"   (pin the plan page to the demo week)

Starts the Streamlit app on a spare port, drives Chrome over the DevTools protocol, waits until
the app has finished computing, and saves one PNG per page. Needs Google Chrome or Chromium.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import signal
import urllib.parse
import urllib.request
from pathlib import Path

from websockets.sync.client import connect  # ships with Streamlit's server stack

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    shutil.which("google-chrome") or "",
    shutil.which("chromium") or "",
    shutil.which("chromium-browser") or "",
]
READY_JS = """(() => {
  const main = document.querySelector('[data-testid="stMain"]') || document.querySelector('section.main');
  if (!main || !document.querySelector('h1')) return 0;
  if (document.querySelector('[data-testid="stStatusWidget"]')) return 0;
  if (document.querySelector('[data-testid="stSpinner"]')) return 0;
  return Math.ceil(main.scrollHeight);
})()"""


def wait_http(url: str, seconds: int) -> None:
    end = time.time() + seconds
    while time.time() < end:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(0.5)
    raise SystemExit(f"timed out waiting for {url}")


def capture(debug_port: int, url: str, out: Path, width: int, timeout: int) -> None:
    req = urllib.request.Request(f"http://127.0.0.1:{debug_port}/json/new?{urllib.parse.quote(url, safe='')}", method="PUT")
    with urllib.request.urlopen(req, timeout=10) as r:
        target = json.loads(r.read())
    with connect(target["webSocketDebuggerUrl"], max_size=200 * 1024 * 1024, open_timeout=20) as ws:
        counter = 0

        def call(method: str, **params) -> dict:
            nonlocal counter
            counter += 1
            ws.send(json.dumps({"id": counter, "method": method, "params": params}))
            while True:
                msg = json.loads(ws.recv(timeout=60))
                if msg.get("id") == counter:
                    return msg.get("result", {})

        call("Emulation.setDeviceMetricsOverride", width=width, height=1200, deviceScaleFactor=1, mobile=False)
        end, height, stable = time.time() + timeout, 0, 0
        while time.time() < end:
            res = call("Runtime.evaluate", expression=READY_JS, returnByValue=True)
            h = int(res.get("result", {}).get("value") or 0)
            stable = stable + 1 if h and h == height else 0
            height = h
            if stable >= 3:  # same non-zero height three times in a row: the page has settled
                break
            time.sleep(1.0)
        if not height:
            raise SystemExit(f"app did not finish rendering {url} within {timeout}s")
        call("Emulation.setDeviceMetricsOverride", width=width, height=height + 120, deviceScaleFactor=1, mobile=False)
        time.sleep(2.5)
        shot = call("Page.captureScreenshot", format="png")
        out.write_bytes(base64.b64decode(shot["data"]))
        call("Page.close")
    print(f"wrote {out}  ({width} x {height + 120})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "docs" / "screenshots"))
    ap.add_argument("--query", default=os.environ.get("QUERY", ""), help='extra URL parameters for the plan page, e.g. "stress=70&week=5"')
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--width", type=int, default=1500)
    ap.add_argument("--timeout", type=int, default=180)
    args = ap.parse_args()
    chrome = os.environ.get("CHROME") or next((c for c in CANDIDATES if c and Path(c).exists()), None)
    if not chrome:
        raise SystemExit("Chrome/Chromium not found. Set CHROME=/path/to/chrome, or take the screenshots by hand.")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    debug_port = args.port + 1
    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "streamlit_app.py"), "--server.headless", "true", "--server.port", str(args.port), "--browser.gatherUsageStats", "false"],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    )
    profile = tempfile.mkdtemp(prefix="cspa_chrome_")
    browser = subprocess.Popen(
        [chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run", f"--remote-debugging-port={debug_port}", f"--user-data-dir={profile}", "--remote-allow-origins=*", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    )
    try:
        wait_http(f"http://localhost:{args.port}/", 90)
        wait_http(f"http://127.0.0.1:{debug_port}/json/version", 30)
        base = f"http://localhost:{args.port}/?"
        q = f"&{args.query}" if args.query else ""
        pages = [("01_plan", f"page=plan{q}"), ("02_plan_compare", f"page=plan&compare=1{q}"), ("03_backtest", "page=backtest"), ("04_about", "page=about")]
        for name, query in pages:
            capture(debug_port, base + query, out / f"{name}.png", args.width, args.timeout)
    finally:
        for proc in (browser, server):  # each runs in its own process group, so helpers go too
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    main()
