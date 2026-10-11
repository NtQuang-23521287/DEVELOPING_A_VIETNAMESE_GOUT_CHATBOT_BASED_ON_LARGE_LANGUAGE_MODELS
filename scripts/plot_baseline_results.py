from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIG
# ============================================================

RUNS_DIR = Path("artifacts/runs")
OUTPUT_DIR = Path("artifacts/analysis_charts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_RECORDS = 316

MODEL_PATTERNS = {
    "Qwen3-8B": "qwen3_8b",
    "Gemma3-4B": "gemma3_4b",
    "SeaLLMs-v3-7B": "seallms_v3_7b",
}

MODEL_ORDER = [
    "Qwen3-8B",
    "Gemma3-4B",
    "SeaLLMs-v3-7B",
]


# ============================================================
# 2. TÌM RUN 316 CÂU GẦN NHẤT
# ============================================================

def find_latest_complete_run(pattern: str):
    candidates = []

    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue

        if pattern not in run_dir.name.lower():
            continue

        summary_path = run_dir / "summary.json"
        scores_path = run_dir / "scores_9_metrics.jsonl"

        if not summary_path.exists() or not scores_path.exists():
            continue

        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)

            if summary.get("records") == EXPECTED_RECORDS:
                candidates.append(run_dir)

        except Exception as e:
            print(f"Skip {run_dir}: {e}")

    if not candidates:
        raise FileNotFoundError(
            f"Không tìm thấy run {EXPECTED_RECORDS} câu cho pattern: {pattern}"
        )

    # timestamp nằm trong tên folder nên sort tên là đủ
    candidates.sort()

    return candidates[-1]


RUN_PATHS = {}

for model_name, pattern in MODEL_PATTERNS.items():
    run_path = find_latest_complete_run(pattern)

    RUN_PATHS[model_name] = run_path

    print(f"{model_name}:")
    print(f"  {run_path}")


# ============================================================
# 3. LOAD SUMMARY + PER-CASE SCORES
# ============================================================

summaries = {}
scores = {}

for model_name, run_path in RUN_PATHS.items():

    # summary.json
    with open(
        run_path / "summary.json",
        "r",
        encoding="utf-8"
    ) as f:
        summaries[model_name] = json.load(f)

    # scores_9_metrics.jsonl
    rows = []

    with open(
        run_path / "scores_9_metrics.jsonl",
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    scores[model_name] = pd.DataFrame(rows)

    print(
        model_name,
        "summary records =",
        summaries[model_name].get("records"),
        "| score rows =",
        len(scores[model_name])
    )


# ============================================================
# 4. TẠO BẢNG TỔNG HỢP
# ============================================================

summary_rows = []

for model in MODEL_ORDER:
    s = summaries[model]

    row = {
        "Model": model,
        "Reward": s["reward_mean"],
        "Quality component": s["quality_component_mean"],
        "Risk penalty": s["risk_penalty_mean"],

        "Fatal safety violations":
            s["fatal_safety_violations"],

        "Fatal safety violation rate":
            s["fatal_safety_violation_rate"],

        "Latency (s)":
            s["mean_latency_seconds"],

        "Tokens/s":
            s["mean_tokens_per_second"],

        "Peak VRAM (MB)":
            s["peak_vram_mb"],
    }

    # 9 metrics
    for metric_name, value in s["metric_means"].items():
        row[metric_name] = value

    summary_rows.append(row)


summary_df = pd.DataFrame(summary_rows)

print("\n===== SUMMARY =====")
print(summary_df.to_string(index=False))

summary_df.to_csv(
    OUTPUT_DIR / "baseline_model_summary.csv",
    index=False
)


# ============================================================
# CHART 1
# OVERALL REWARD
# ============================================================

plt.figure(figsize=(8, 5))

values = summary_df["Reward"].values

bars = plt.bar(
    summary_df["Model"],
    values
)

plt.ylabel("Mean Reward")
plt.xlabel("Model")
plt.title("Overall Reward Across 316 Evaluation Cases")

plt.ylim(
    max(0, values.min() - 0.05),
    min(1, values.max() + 0.05)
)

for bar, value in zip(bars, values):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.003,
        f"{value:.4f}",
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "01_overall_reward.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# CHART 2
# 6 QUALITY METRICS M1-M6
# ============================================================

quality_metrics = [
    ("faithfulness", "Faithfulness"),
    ("completeness", "Completeness"),
    ("relevance", "Relevance"),
    ("context_recall", "Context Recall"),
    ("patient_utility", "Patient Utility"),
    (
        "refusal_appropriateness",
        "Refusal Appropriateness"
    ),
]

x = np.arange(len(quality_metrics))
width = 0.25

plt.figure(figsize=(13, 6))

for idx, model in enumerate(MODEL_ORDER):

    values = [
        summaries[model]["metric_means"][key]
        for key, _ in quality_metrics
    ]

    plt.bar(
        x + (idx - 1) * width,
        values,
        width=width,
        label=model
    )

plt.xticks(
    x,
    [label for _, label in quality_metrics],
    rotation=20,
    ha="right"
)

plt.ylabel("Mean Score")
plt.xlabel("Quality Metric")

plt.title(
    "Quality Metrics M1–M6 "
    "(Higher Is Better)"
)

plt.ylim(0, 1.05)

plt.legend()

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "02_quality_metrics.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# CHART 3
# 3 RISK METRICS P1-P3
# ============================================================

risk_metrics = [
    (
        "hallucination_level",
        "Hallucination Level"
    ),
    (
        "omission_risk",
        "Omission Risk"
    ),
    (
        "safety_risk",
        "Safety Risk"
    ),
]

x = np.arange(len(risk_metrics))
width = 0.25

plt.figure(figsize=(10, 6))

all_risk_values = []

for idx, model in enumerate(MODEL_ORDER):

    values = [
        summaries[model]["metric_means"][key]
        for key, _ in risk_metrics
    ]

    all_risk_values.extend(values)

    plt.bar(
        x + (idx - 1) * width,
        values,
        width=width,
        label=model
    )

plt.xticks(
    x,
    [label for _, label in risk_metrics]
)

plt.ylabel("Mean Risk Score")
plt.xlabel("Risk Metric")

plt.title(
    "Risk Metrics P1–P3 "
    "(Lower Is Better)"
)

risk_max = max(all_risk_values)

plt.ylim(
    0,
    min(1, risk_max * 1.25 + 0.02)
)

plt.legend()

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "03_risk_metrics.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# CHART 4
# REWARD VS LATENCY
# ============================================================

plt.figure(figsize=(8, 6))

for model in MODEL_ORDER:

    latency = summaries[model][
        "mean_latency_seconds"
    ]

    reward = summaries[model][
        "reward_mean"
    ]

    plt.scatter(
        latency,
        reward,
        s=120
    )

    plt.annotate(
        model,
        (latency, reward),
        xytext=(7, 7),
        textcoords="offset points",
        fontsize=10
    )

plt.xlabel("Mean Latency (seconds)")
plt.ylabel("Mean Reward")

plt.title(
    "Quality–Efficiency Trade-off"
)

plt.grid(
    linestyle="--",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "04_reward_vs_latency.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# CHART 5
# REWARD THEO NHÓM BỆNH
# ============================================================

if all(
    "target_disease" in scores[m].columns
    for m in MODEL_ORDER
):

    # dùng order từ Qwen
    diseases = list(
        scores["Qwen3-8B"][
            "target_disease"
        ]
        .value_counts()
        .index
    )

    x = np.arange(len(diseases))
    width = 0.25

    plt.figure(figsize=(14, 7))

    for idx, model in enumerate(MODEL_ORDER):

        disease_reward = (
            scores[model]
            .groupby("target_disease")["reward"]
            .mean()
            .reindex(diseases)
        )

        plt.bar(
            x + (idx - 1) * width,
            disease_reward.values,
            width=width,
            label=model
        )

    plt.xticks(
        x,
        diseases,
        rotation=30,
        ha="right"
    )

    plt.ylabel("Mean Reward")
    plt.xlabel("Disease Group")

    plt.title(
        "Reward by Disease Group"
    )

    plt.legend()

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "05_reward_by_disease.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

else:
    print(
        "Không có cột target_disease, "
        "bỏ qua Chart 5."
    )


# ============================================================
# CHART 6
# PHÂN BỐ REWARD TRÊN 316 CASE
# ============================================================

reward_data = [
    scores[model]["reward"]
    .dropna()
    .values

    for model in MODEL_ORDER
]

plt.figure(figsize=(9, 6))

plt.boxplot(
    reward_data,
    tick_labels=MODEL_ORDER,
    showmeans=True
)

plt.ylabel("Reward")
plt.xlabel("Model")

plt.title(
    "Per-case Reward Distribution "
    "Across 316 Cases"
)

plt.ylim(0, 1)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "06_reward_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 10. PAIRWISE REWARD COMPARISON
# ============================================================

wide = None

for model in MODEL_ORDER:

    temp = (
        scores[model][
            ["case_id", "reward"]
        ]
        .copy()
        .rename(
            columns={"reward": model}
        )
    )

    if wide is None:
        wide = temp
    else:
        wide = wide.merge(
            temp,
            on="case_id",
            how="inner"
        )


pairs = [
    ("Qwen3-8B", "Gemma3-4B"),
    ("Qwen3-8B", "SeaLLMs-v3-7B"),
    ("Gemma3-4B", "SeaLLMs-v3-7B"),
]

comparison_rows = []

for model_a, model_b in pairs:

    diff = (
        wide[model_a]
        - wide[model_b]
    )

    comparison_rows.append(
        {
            "Model A": model_a,
            "Model B": model_b,

            "Mean Reward A":
                wide[model_a].mean(),

            "Mean Reward B":
                wide[model_b].mean(),

            "Mean Difference A-B":
                diff.mean(),

            "Median Difference":
                diff.median(),

            "A Wins":
                int((diff > 0).sum()),

            "B Wins":
                int((diff < 0).sum()),

            "Ties":
                int((diff == 0).sum()),
        }
    )


comparison_df = pd.DataFrame(
    comparison_rows
)

print("\n===== PAIRWISE COMPARISON =====")
print(
    comparison_df.to_string(index=False)
)

comparison_df.to_csv(
    OUTPUT_DIR /
    "pairwise_reward_comparison.csv",
    index=False
)


# ============================================================
# DONE
# ============================================================

print("\nHoàn tất.")
print(
    "Charts được lưu tại:",
    OUTPUT_DIR.resolve()
)