from __future__ import annotations

from statistics import mean


def aggregate_predictions(rows: list[dict]) -> dict:
    ok = [r for r in rows if r.get("status") == "ok"]
    latencies = [float(r["latency_seconds"]) for r in ok if r.get("latency_seconds") is not None]
    in_tok = [int(r["input_tokens"]) for r in ok if r.get("input_tokens") is not None]
    out_tok = [int(r["output_tokens"]) for r in ok if r.get("output_tokens") is not None]
    tps = [float(r["tokens_per_second"]) for r in ok if r.get("tokens_per_second") is not None]
    vram = [float(r["peak_vram_mb"]) for r in ok if r.get("peak_vram_mb") is not None]
    return {
        "records": len(rows),
        "ok": len(ok),
        "errors": len(rows) - len(ok),
        "success_rate": len(ok) / len(rows) if rows else 0.0,
        "latency_mean_seconds": mean(latencies) if latencies else None,
        "input_tokens_mean": mean(in_tok) if in_tok else None,
        "output_tokens_mean": mean(out_tok) if out_tok else None,
        "tokens_per_second_mean": mean(tps) if tps else None,
        "peak_vram_mb_max": max(vram) if vram else None,
    }
