
import json
import ast
import re
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def load_sft_history(source):
    """
    Đọc training history từ:
      - Thư mục run có checkpoint-*/trainer_state.json
      - File trainer_state.json
      - ZIP chứa checkpoint và trainer_state.json
      - File log .txt, .log hoặc .md
    """
    source = Path(source)

    if source.is_dir():
        paths = list(source.rglob("trainer_state.json"))
        if not paths:
            raise FileNotFoundError(
                f"Không tìm thấy trainer_state.json trong {source}"
            )

        states = [
            (json.loads(p.read_text(encoding="utf-8")), str(p))
            for p in paths
        ]
        state, selected = max(
            states,
            key=lambda x: x[0].get("global_step", -1)
        )
        records = state["log_history"]

    elif source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as z:
            names = [
                n for n in z.namelist()
                if n.endswith("trainer_state.json")
            ]
            if not names:
                raise ValueError(
                    f"ZIP không chứa trainer_state.json: {source}"
                )

            states = [
                (json.loads(z.read(n)), n)
                for n in names
            ]
            state, selected = max(
                states,
                key=lambda x: x[0].get("global_step", -1)
            )
            records = state["log_history"]

    elif source.suffix.lower() == ".json":
        state = json.loads(source.read_text(encoding="utf-8"))
        records = state["log_history"]
        selected = str(source)

    else:
        raw = source.read_text(
            encoding="utf-8", errors="replace"
        )
        records = []

        for match in re.finditer(r"\{[^{}]*\}", raw):
            try:
                item = ast.literal_eval(match.group())
            except (ValueError, SyntaxError):
                continue

            if (
                isinstance(item, dict)
                and "epoch" in item
                and ("loss" in item or "eval_loss" in item)
            ):
                records.append(item)

        selected = str(source)

    if not records:
        raise ValueError(f"Không có dữ liệu loss: {source}")

    df = pd.DataFrame(records)

    for col in ["epoch", "step", "loss", "eval_loss", "grad_norm"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print(f"Đã đọc: {selected}")
    print(f"  Training logs: {df['loss'].notna().sum()}")
    print(f"  Validation logs: {df['eval_loss'].notna().sum()}")

    return df


def prepare_sft_metrics(df):
    """Tính trung bình Training Loss và Gradient Norm mỗi epoch."""
    train = df[df["loss"].notna() & df["epoch"].notna()].copy()
    val = df[df["eval_loss"].notna() & df["epoch"].notna()].copy()

    if train.empty or val.empty:
        raise ValueError("Thiếu Training Loss hoặc Validation Loss")

    train["epoch_id"] = np.ceil(
        train["epoch"] - 1e-8
    ).astype(int)

    train_epoch = (
        train.groupby("epoch_id")
        .agg(
            train_loss=("loss", "mean"),
            grad_norm=("grad_norm", "mean")
        )
        .reset_index()
        .rename(columns={"epoch_id": "epoch"})
    )

    val["epoch"] = val["epoch"].round().astype(int)

    val_epoch = (
        val[["epoch", "eval_loss"]]
        .drop_duplicates(subset="epoch", keep="last")
        .sort_values("epoch")
    )

    metrics = (
        train_epoch.merge(val_epoch, on="epoch", how="outer")
        .sort_values("epoch")
        .reset_index(drop=True)
    )

    return train, metrics


def plot_sft_comparison(models, output_dir="reports/figures/sft"):
    """
    models: dict {model_name: path}
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    data = {}
    summaries = []

    for name, source in models.items():
        history = load_sft_history(source)
        train, metrics = prepare_sft_metrics(history)

        data[name] = {
            "train": train,
            "metrics": metrics
        }

        valid = metrics.dropna(subset=["eval_loss"])
        best = valid.loc[valid["eval_loss"].idxmin()]

        summaries.append({
            "model": name,
            "best_epoch": int(best["epoch"]),
            "best_eval_loss": float(best["eval_loss"]),
            "final_eval_loss": float(valid.iloc[-1]["eval_loss"])
        })

        filename = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
        metrics.to_csv(
            output / f"{filename}_epoch_metrics.csv",
            index=False
        )

    # 4 biểu đồ tổng hợp
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax_train = axes[0, 0]
    ax_val = axes[0, 1]
    ax_relative = axes[1, 0]
    ax_grad = axes[1, 1]

    for name, item in data.items():
        m = item["metrics"]
        v = m.dropna(subset=["eval_loss"])

        ax_train.plot(
            m["epoch"], m["train_loss"],
            marker="o", label=name
        )

        ax_val.plot(
            v["epoch"], v["eval_loss"],
            marker="o", label=name
        )

        best = v.loc[v["eval_loss"].idxmin()]
        ax_val.scatter(
            best["epoch"], best["eval_loss"], s=80
        )
        ax_val.annotate(
            f"E{int(best['epoch'])}: {best['eval_loss']:.3f}",
            (best["epoch"], best["eval_loss"]),
            xytext=(5, -16),
            textcoords="offset points",
            fontsize=9
        )

        relative = (
            (v["eval_loss"] / v.iloc[0]["eval_loss"]) - 1
        ) * 100

        ax_relative.plot(
            v["epoch"], relative,
            marker="o", label=name
        )

        ax_grad.plot(
            m["epoch"], m["grad_norm"],
            marker="o", label=name
        )

    titles = [
        (ax_train, "Training Loss (Mean per Epoch)", "Loss"),
        (ax_val, "Validation Loss", "Loss"),
        (ax_relative, "Relative Validation Loss Change", "%"),
        (ax_grad, "Mean Gradient Norm", "Gradient Norm"),
    ]

    for ax, title, ylabel in titles:
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()

    ax_relative.axhline(0, linestyle="--", alpha=0.5)

    fig.suptitle(
        "SFT Training Analysis: Qwen3 vs Gemma3",
        fontsize=16, fontweight="bold"
    )

    fig.tight_layout()

    fig.savefig(
        output / "sft_loss_comparison.png",
        dpi=250, bbox_inches="tight"
    )
    fig.savefig(
        output / "sft_loss_comparison.svg",
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    # Biểu đồ Training Loss chi tiết theo step
    fig, ax = plt.subplots(figsize=(12, 5))

    for name, item in data.items():
        train = item["train"]
        steps = np.arange(1, len(train) + 1)

        smoothed = train["loss"].rolling(
            window=10, min_periods=1
        ).mean()

        ax.plot(steps, smoothed, label=f"{name} (MA10)")

    ax.set_title("SFT Training Loss by Step (Moving Average 10)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Training Loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        output / "sft_train_loss_steps.png",
        dpi=250, bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    summary = pd.DataFrame(summaries)
    summary.to_csv(output / "sft_summary.csv", index=False)

    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))
    print("\nĐã lưu biểu đồ và CSV tại:", output.resolve())

    return summary
