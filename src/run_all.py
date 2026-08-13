"""
run_all.py
----------
Orchestrator script that sequentially trains all 5 architectures on the same dataset
with identical hyperparameters (10 epochs, Adam optimizer, lr=0.001) for fair comparison.
Reports progress and metrics at each stage.
"""

import argparse
import sys
import time
from typing import List
import torch

from train import run_training


MODELS_TO_COMPARE = [
    "tinyvgg",
    "resnet18",
    "efficientnet_b0",
    "efficientnet_b7",
    "mobilenet_v3_small"
]


def run_all(
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 0.001,
    data_dir: str = "data",
    results_dir: str = "results",
    use_amp: bool = True,
    seed: int = 42
):
    print("=" * 80)
    print("TRANSFER LEARNING BENCHMARK: 5 ARCHITECTURES COMPARISON")
    print(f"Models: {', '.join(MODELS_TO_COMPARE)}")
    print(f"Epochs per model: {epochs} | Mixed Precision (AMP): {use_amp}")
    print("=" * 80)

    total_pipeline_start = time.time()
    summary_results = []

    for idx, model_name in enumerate(MODELS_TO_COMPARE, 1):
        print(f"\n>>>>>>>> PROGRESS: [{idx}/{len(MODELS_TO_COMPARE)}] Training '{model_name}' >>>>>>>>")
        model_start = time.time()

        metrics = run_training(
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            data_dir=data_dir,
            results_dir=results_dir,
            use_amp=use_amp,
            seed=seed,
            num_workers=2
        )

        model_elapsed = time.time() - model_start
        status = metrics.get("status", "unknown")
        test_acc = metrics.get("final_test_accuracy", "N/A")
        train_time = metrics.get("training_time_seconds", model_elapsed)

        print(f"<<<<<<<< FINISHED [{idx}/{len(MODELS_TO_COMPARE)}] '{model_name}' in {model_elapsed:.2f}s | Status: {status} | Test Acc: {test_acc}% <<<<<<<<\n")
        summary_results.append(metrics)

    total_pipeline_time = time.time() - total_pipeline_start
    print("=" * 80)
    print(f"ALL 5 MODELS PROCESSED in {total_pipeline_time:.2f}s ({total_pipeline_time/60:.2f} minutes)!")
    print("=" * 80)
    return summary_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full 5-model benchmark")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per model (default: 10)")
    parser.add_argument("--batch_size", type=int, default=32, help="Base batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (default: 0.001)")
    parser.add_argument("--no_amp", action="store_true", help="Disable AMP")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    args = parser.parse_args()

    run_all(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_amp=not args.no_amp,
        seed=args.seed
    )