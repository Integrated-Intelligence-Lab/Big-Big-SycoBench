
from __future__ import annotations

import urllib.error
import urllib.request

from common import INPUTS, ensure_dirs


def download(url: str, path) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            content = response.read()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        print(f"Could not download {url}")
        print(f"Reason: {exc}")
        return False

    path.write_bytes(content)
    print(f"Wrote {path} ({len(content):,} bytes)")
    return True


def main() -> None:
    ensure_dirs()
    missing = []

    for name, meta in INPUTS.items():
        path = meta["path"]
        if path.exists() and path.stat().st_size > 0:
            print(f"Already present: {path}")
            continue
        ok = download(meta["url"], path)
        if not ok:
            missing.append((name, path, meta["url"]))

    if missing:
        print("\nManual download needed for:")
        for name, path, url in missing:
            print(f"- {name}: {url}")
            print(f"  save as: {path}")
        raise SystemExit(1)

    print("\nAll inputs are ready.")


if __name__ == "__main__":
    main()

