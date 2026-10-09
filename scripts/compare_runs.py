import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    args = ap.parse_args()
    rows = []
    for run in args.runs:
        p = Path(run)
        summary = json.loads((p / "summary.json").read_text(encoding="utf-8"))
        manifest = json.loads((p / "run_manifest.json").read_text(encoding="utf-8"))
        rows.append({"run_id": manifest["run_id"], "model": manifest["model_name"], **summary})
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
