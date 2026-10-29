"""Download the public OSF project k8ph9 (Sales, Prihar, Gagnon-Bartsch & Heffernan) via the official OSF API.

- Walks the osfstorage tree recursively.
- Resumable downloads (HTTP Range) into data/raw/osf_k8ph9/<path>.
- Verifies byte size against the OSF-reported size, computes SHA-256,
  and writes data/DATA_MANIFEST.md plus data/manifest.json.

Usage:  python src/00_download.py [--only-small]
"""
import argparse
import datetime as dt
import hashlib
import json
import pathlib
import sys
import time

import requests

NODE = "k8ph9"
API_ROOT = f"https://api.osf.io/v2/nodes/{NODE}/files/osfstorage/"
ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / f"osf_{NODE}"
SESSION = requests.Session()
SESSION.headers["User-Agent"] = "ET610-research-replication (python-requests)"


def get_json(url):
    for attempt in range(5):
        r = SESSION.get(url, timeout=60)
        if r.status_code == 200:
            return r.json()
        time.sleep(2 ** attempt)
    raise RuntimeError(f"GET {url} failed with {r.status_code}")


def walk(url, prefix=""):
    """Yield file records (dicts) from the OSF storage tree."""
    while url:
        js = get_json(url)
        for item in js["data"]:
            a = item["attributes"]
            if a["kind"] == "folder":
                sub = item["relationships"]["files"]["links"]["related"]["href"]
                yield from walk(sub, prefix + a["name"] + "/")
            else:
                yield {
                    "path": prefix + a["name"],
                    "osf_id": item["id"],
                    "size": a["size"],
                    "date_modified": a.get("date_modified"),
                    "version": (a.get("extra") or {}).get("version") or a.get("current_version"),
                    "osf_sha256": ((a.get("extra") or {}).get("hashes") or {}).get("sha256"),
                    "download": item["links"]["download"],
                }
        url = js["links"].get("next")


def download(rec):
    dest = RAW / rec["path"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    if have == rec["size"]:
        return dest
    if have > rec["size"]:
        dest.unlink()
        have = 0
    for attempt in range(8):
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with SESSION.get(rec["download"], headers=headers, stream=True, timeout=120) as r:
                if have and r.status_code != 206:  # server ignored range: restart
                    have = 0
                    mode = "wb"
                else:
                    mode = "ab" if have else "wb"
                r.raise_for_status()
                with open(dest, mode) as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
        except (requests.RequestException, OSError) as e:
            print(f"  retry {attempt} for {rec['path']}: {e}", file=sys.stderr)
            time.sleep(min(60, 2 ** attempt))
        have = dest.stat().st_size if dest.exists() else 0
        if have == rec["size"]:
            return dest
    raise RuntimeError(f"Incomplete download {rec['path']}: {have} of {rec['size']} bytes")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-small", action="store_true", help="skip files > 100 MB")
    args = ap.parse_args()
    retrieved = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    records = list(walk(API_ROOT))
    print(f"{len(records)} files listed on OSF")
    for rec in records:
        if args.only_small and rec["size"] > 100e6:
            rec["status"] = "skipped (--only-small)"
            continue
        print(f"downloading {rec['path']} ({rec['size']:,} bytes)")
        dest = download(rec)
        rec["local_size"] = dest.stat().st_size
        rec["sha256"] = sha256(dest)
        rec["size_match"] = rec["local_size"] == rec["size"]
        rec["osf_hash_match"] = (rec["osf_sha256"] == rec["sha256"]) if rec["osf_sha256"] else None
        rec["status"] = "ok" if rec["size_match"] and rec["osf_hash_match"] is not False else "MISMATCH"
        rec["retrieved_utc"] = retrieved
    (ROOT / "data" / "manifest.json").write_text(json.dumps(records, indent=2))
    lines = [
        "# Data manifest — OSF project k8ph9",
        "",
        f"Source: https://osf.io/{NODE}/ (official OSF API, `{API_ROOT}`). Retrieved {retrieved}.",
        "Raw files live in `data/raw/osf_k8ph9/` and are excluded from git (ASSISTments Data ToU: no redistribution).",
        "Integrity: local byte size compared with the OSF-reported size; SHA-256 computed locally and compared",
        "with the OSF-stored SHA-256 where OSF provides one. Parseability is checked separately in the audit (src/01).",
        "",
        "| Path | OSF file id | OSF size (bytes) | Modified (OSF) | Version | SHA-256 (local) | Size match | OSF hash match | Status |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        lines.append(
            f"| `{r['path']}` | {r['osf_id']} | {r['size']:,} | {r['date_modified']} | {r.get('version')} | "
            f"`{r.get('sha256', '-')}` | {r.get('size_match', '-')} | {r.get('osf_hash_match', '-')} | {r['status']} |"
        )
    (ROOT / "data" / "DATA_MANIFEST.md").write_text("\n".join(lines) + "\n")
    bad = [r for r in records if r["status"] == "MISMATCH"]
    if bad:
        sys.exit(f"Integrity failure: {[r['path'] for r in bad]}")


if __name__ == "__main__":
    main()
