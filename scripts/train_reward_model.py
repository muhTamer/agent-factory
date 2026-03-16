#!/usr/bin/env python3
"""
Train the neural reward model MLP on generated training data.

Usage:
    python scripts/train_reward_model.py [--data DATA_PATH] [--output MODEL_PATH] [--epochs N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from app.orchestration.reward_model_trainer import RewardModelTrainer


def main():
    parser = argparse.ArgumentParser(description="Train reward model")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/training_data/reward_training.json"),
        help="Path to training data JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/reward_mlp.pt"),
        help="Path to save trained model weights",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split fraction")
    args = parser.parse_args()

    if not args.data.exists():
        print(f"Error: training data not found at {args.data}")
        print("Run scripts/generate_training_data.py first.")
        return

    trainer = RewardModelTrainer()
    metadata = trainer.train(
        data_path=args.data,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_split=args.val_split,
    )

    print(f"\nFinal train loss: {metadata['final_train_loss']:.4f}")
    if metadata.get("final_val_loss") is not None:
        print(f"Final val loss:   {metadata['final_val_loss']:.4f}")


if __name__ == "__main__":
    main()
