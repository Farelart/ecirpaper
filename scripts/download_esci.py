"""Download the official ESCI release at a fixed Git revision (stdlib only)."""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "7916cdf6ab75a462e77f20ab40428a10923998d5"
REPOSITORY = "amazon-science/esci-data"
FILES = {
    "shopping_queries_dataset_examples.parquet": (51286808, "4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a"),
    "shopping_queries_dataset_products.parquet": (1108857465, "25124442d064d64b26f74082d6fa09438d679efc0c183cf28d19064a2b65a265"),
    "shopping_queries_dataset_sources.csv": (1682802, None),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url, destination):
    temporary = destination.with_suffix(destination.suffix + ".part")
    subprocess.run([
        "curl", "--fail", "--location", "--silent", "--show-error",
        "--retry", "3", "--connect-timeout", "30", "--output", str(temporary), url,
    ], check=True)
    return temporary


def main():
    raw = ROOT / "data/raw/esci"
    raw.mkdir(parents=True, exist_ok=True)
    records = []
    for name, (expected_bytes, expected_hash) in FILES.items():
        url = (f"https://media.githubusercontent.com/media/{REPOSITORY}/{REVISION}/"
               f"shopping_queries_dataset/{name}") if name.endswith(".parquet") else (
                   f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/shopping_queries_dataset/{name}")
        target = raw / name
        if target.exists() and target.stat().st_size == expected_bytes:
            actual_hash = sha256(target)
            if expected_hash and actual_hash != expected_hash:
                raise RuntimeError(f"Existing {target} failed checksum verification")
            print(f"Verified existing {name}", flush=True)
        else:
            print(f"Downloading {name} ({expected_bytes / 1e6:.1f} MB)", flush=True)
            temporary = fetch(url, target)
            actual_hash = sha256(temporary)
            if temporary.stat().st_size != expected_bytes:
                raise RuntimeError(f"Size mismatch for {name}")
            if expected_hash and actual_hash != expected_hash:
                raise RuntimeError(f"Checksum mismatch for {name}")
            temporary.replace(target)
            print(f"Verified {name}", flush=True)
        records.append({"name": name, "url": url, "bytes": expected_bytes,
                        "sha256": actual_hash, "upstream_lfs_sha256": expected_hash})
    sources = ROOT / "data/source"
    sources.mkdir(parents=True, exist_ok=True)
    for name in ["README.md", "LICENSE", "NOTICE", "ranking/prepare_trec_eval_files.py"]:
        target = sources / name.replace("/", "_")
        temporary = fetch(f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/{name}", target)
        temporary.replace(target)
    manifest = {"repository": f"https://github.com/{REPOSITORY}", "revision": REVISION,
                "downloaded_at_utc": datetime.now(timezone.utc).isoformat(), "files": records}
    (ROOT / "data/esci_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Saved data/esci_manifest.json", flush=True)


if __name__ == "__main__":
    main()
