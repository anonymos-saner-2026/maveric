import argparse
import time
import json
from pathlib import Path
import requests

# Default: publicly available ETSI PAS PDFs that adopt O-RAN Alliance specs.
# You can add/remove URLs freely.
DEFAULT_URLS = [
    # WG1 Architecture + Slicing
    ("ETSI_TS_103_982_O-RAN_Architecture_Description_v08.00.00", "https://www.etsi.org/deliver/etsi_ts/103900_103999/103982/08.00.00_60/ts_103982v080000p.pdf"),
    ("ETSI_TS_104_041_O-RAN_Slicing_Architecture_v11.00.00", "https://www.etsi.org/deliver/etsi_ts/104000_104099/104041/11.00.00_60/ts_104041v110000p.pdf"),

    # A1 interface suite (WG2)
    ("ETSI_TS_103_983_A1_General_Aspects_v03.01.00", "https://www.etsi.org/deliver/etsi_ts/103900_103999/103983/03.01.00_60/ts_103983v030100p.pdf"),
    ("ETSI_TS_103_985_A1_Use_Cases_Requirements_v01.01.00", "https://www.etsi.org/deliver/etsi_ts/103900_103999/103985/01.01.00_60/ts_103985v010100p.pdf"),
    ("ETSI_TS_103_986_A1_Transport_Protocol_v02.01.00", "https://www.etsi.org/deliver/etsi_ts/103900_103999/103986/02.01.00_60/ts_103986v020100p.pdf"),
    ("ETSI_TS_103_987_A1_Application_Protocol_v04.00.00", "https://www.etsi.org/deliver/etsi_ts/103900_103999/103987/04.00.00_60/ts_103987v040000p.pdf"),
    ("ETSI_TS_103_988_A1_Type_Definitions_v05.00.00", "https://www.etsi.org/deliver/etsi_ts/103900_103999/103988/05.00.00_60/ts_103988v050000p.pdf"),
    ("ETSI_TS_103_989_A1_Test_Specification_v03.00.00", "https://www.etsi.org/deliver/etsi_ts/103900_103999/103989/03.00.00_60/ts_103989v030000p.pdf"),

    # Other ETSI PAS O-RAN docs (WG9/WG10/WG11)
    ("ETSI_TS_104_023_O-RAN_Fronthaul_Management_Plane_v12.00.01", "https://www.etsi.org/deliver/etsi_ts/104000_104099/104023/12.00.01_60/ts_104023v120001p.pdf"),
    ("ETSI_TS_104_107_O-RAN_Security_Protocols_v09.00.00", "https://www.etsi.org/deliver/etsi_ts/104100_104199/104107/09.00.00_60/ts_104107v090000p.pdf"),

    # Technical reports from O-RAN Alliance (high-level analysis)
    ("ETSI_TR_104_037_O-RAN_Use_Cases_Analysis_Report_v12.00.00", "https://www.etsi.org/deliver/etsi_tr/104000_104099/104037/12.00.00_60/tr_104037v120000p.pdf"),
]


def read_url_file(path: str):
    """
    url_file supports either:
      - json list: [{"name": "...", "url": "..."}, ...]
      - plain text: one url per line (name will be derived)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)

    if p.suffix.lower() == ".json":
        items = json.loads(p.read_text(encoding="utf-8"))
        out = []
        for it in items:
            out.append((it.get("name") or Path(it["url"]).name, it["url"]))
        return out

    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = Path(line.split("?")[0]).name or f"doc_{len(out):03d}"
        out.append((name, line))
    return out


def download_one(name: str, url: str, out_dir: Path, timeout=60, retries=3):
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "._-+" else "_" for c in name)
    if not safe.lower().endswith(".pdf"):
        safe = safe + ".pdf"
    out_path = out_dir / safe

    if out_path.exists() and out_path.stat().st_size > 1000:
        print(f"✓ exists: {out_path.name}")
        return

    headers = {"User-Agent": "Mozilla/5.0 (oran-slm-qa; research)"}
    for attempt in range(1, retries + 1):
        try:
            print(f"↓ downloading ({attempt}/{retries}): {safe}")
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            out_path.write_bytes(r.content)
            print(f"✓ saved: {out_path} ({out_path.stat().st_size/1024:.1f} KB)")
            return
        except Exception as e:
            print(f"✗ failed: {url}\n  {e}")
            if attempt < retries:
                time.sleep(2.0 * attempt)
            else:
                print("  giving up.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/raw_specs", help="Output directory")
    ap.add_argument("--url_file", type=str, default=None, help="Optional URL list file (txt or json)")
    ap.add_argument("--timeout", type=int, default=60)
    args = ap.parse_args()

    out_dir = Path(args.out)
    urls = list(DEFAULT_URLS)
    if args.url_file:
        urls.extend(read_url_file(args.url_file))

    print(f"Will download {len(urls)} documents into: {out_dir}")
    for name, url in urls:
        download_one(name, url, out_dir, timeout=args.timeout)

    print("Done.")


if __name__ == "__main__":
    main()
