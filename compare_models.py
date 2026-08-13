"""
compare_models.py
-----------------
Aggregates metrics from all trained models under results/, generates:
1. results/comparison_table.csv
2. results/comparison_chart.png (Comprehensive multi-panel visualization)
3. Formatted terminal summary
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def compare_models(results_dir: str = "results") -> pd.DataFrame:
    results_path = Path(results_dir)
    data = []

    model_display_names = {
        "tinyvgg": "TinyVGG (Scratch)",
        "resnet18": "ResNet18 (Pretrained)",
        "efficientnet_b0": "EfficientNet-B0 (Pretrained)",
        "efficientnet_b7": "EfficientNet-B7 (Pretrained)",
        "mobilenet_v3_small": "MobileNetV3-Small (Pretrained)"
    }

    # Desired ordering
    ordered_models = ["tinyvgg", "resnet18", "efficientnet_b0", "efficientnet_b7", "mobilenet_v3_small"]

    for model_key in ordered_models:
        metric_file = results_path / model_key / "metrics.json"
        if not metric_file.exists():
            print(f"[Warning] Metrics file not found for '{model_key}': {metric_file}")
            continue

        with open(metric_file, "r", encoding="utf-8") as f:
            m = json.load(f)

        data.append({
            "Model Key": model_key,
            "Model": model_display_names.get(model_key, model_key),
            "Test Accuracy (%)": m.get("final_test_accuracy", 0.0),
            "Test Loss": m.get("final_test_loss", 0.0),
            "Training Time (s)": m.get("training_time_seconds", 0.0),
            "Total Parameters": m.get("total_params", 0),
            "Trainable Parameters": m.get("trainable_params", 0),
            "Frozen Parameters": m.get("frozen_params", 0)
        })

    df = pd.DataFrame(data)

    # 1. Save CSV
    csv_path = results_path / "comparison_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"[SUCCESS] Saved comparison table to: {csv_path}")

    # 2. Print Summary Table
    print("\n" + "=" * 95)
    print("MODEL PERFORMANCE COMPARISON SUMMARY")
    print("=" * 95)
    display_df = df[["Model", "Test Accuracy (%)", "Training Time (s)", "Total Parameters", "Trainable Parameters"]]
    print(display_df.to_string(index=False))
    print("=" * 95)

    # 3. Create Multi-Panel Visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=200)
    plt.subplots_adjust(hspace=0.35, wspace=0.25)
    sns.set_theme(style="whitegrid")

    palette = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12"]

    # --- Plot 1: Test Accuracy Bar Chart ---
    ax1 = axes[0, 0]
    bars1 = ax1.bar(df["Model"], df["Test Accuracy (%)"], color=palette, edgecolor="black", alpha=0.85, width=0.6)
    ax1.set_title("1. Test Accuracy Comparison (%)", fontsize=13, fontweight="bold", pad=10)
    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.set_ylim([0, 110])
    ax1.tick_params(axis="x", rotation=25)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 1.5, f"{yval:.2f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # --- Plot 2: Training Time Bar Chart ---
    ax2 = axes[0, 1]
    bars2 = ax2.bar(df["Model"], df["Training Time (s)"], color=palette, edgecolor="black", alpha=0.85, width=0.6)
    ax2.set_title("2. Total Training Time (Seconds - 10 Epochs)", fontsize=13, fontweight="bold", pad=10)
    ax2.set_ylabel("Time (seconds)", fontsize=11)
    ax2.tick_params(axis="x", rotation=25)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, yval + 5, f"{yval:.1f}s", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # --- Plot 3: Total vs Trainable Parameters (Log Scale) ---
    ax3 = axes[1, 0]
    x = range(len(df))
    width = 0.35
    ax3.bar([i - width/2 for i in x], df["Total Parameters"], width=width, label="Total Parameters", color="#34495e", alpha=0.85)
    ax3.bar([i + width/2 for i in x], df["Trainable Parameters"], width=width, label="Trainable Parameters", color="#e67e22", alpha=0.85)
    ax3.set_yscale("log")
    ax3.set_xticks(x)
    ax3.set_xticklabels(df["Model"], rotation=25, ha="right", fontsize=9)
    ax3.set_title("3. Total vs Trainable Parameters (Log Scale)", fontsize=13, fontweight="bold", pad=10)
    ax3.set_ylabel("Parameter Count (Log Scale)", fontsize=11)
    ax3.legend(fontsize=10)

    # --- Plot 4: Accuracy vs Efficiency (Bubble Plot) ---
    ax4 = axes[1, 1]
    scatter_sizes = [max(100, p / 100000) for p in df["Total Parameters"]]
    scatter = ax4.scatter(
        df["Training Time (s)"],
        df["Test Accuracy (%)"],
        s=scatter_sizes,
        c=palette,
        alpha=0.7,
        edgecolors="black",
        linewidths=1.5
    )
    for i, txt in enumerate(df["Model Key"]):
        ax4.annotate(
            txt,
            (df["Training Time (s)"][i], df["Test Accuracy (%)"][i]),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            fontweight="bold",
            fontsize=9
        )
    ax4.set_title("4. Accuracy vs Training Time Trade-off", fontsize=13, fontweight="bold", pad=10)
    ax4.set_xlabel("Training Time (seconds)", fontsize=11)
    ax4.set_ylabel("Test Accuracy (%)", fontsize=11)
    ax4.set_ylim([50, 105])

    plt.tight_layout()
    chart_path = results_path / "comparison_chart.png"
    plt.savefig(chart_path)
    plt.close()
    print(f"[SUCCESS] Saved comparison chart to: {chart_path}")

    return df


if __name__ == "__main__":
    compare_models()