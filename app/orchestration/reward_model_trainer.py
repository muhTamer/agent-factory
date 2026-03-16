# app/orchestration/reward_model_trainer.py
"""
Reward Model Trainer for the Neural Solvability Estimator.

Trains a 3-layer MLP on (subtask_embedding, agent_embedding) → score
training data.  The sentence embedding model is FROZEN — only the MLP
layers are updated.

Training specs (per AOP paper):
  - Epochs: 50
  - Batch size: 32
  - Learning rate: 1e-3
  - Loss: MSE
  - Optimiser: Adam
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from app.orchestration.neural_solvability_estimator import RewardMLP, _get_embedder

# ── Dataset ─────────────────────────────────────────────────────────


class RewardDataset(Dataset):
    """Pre-computes embeddings and pairs them with target scores."""

    def __init__(self, data: List[Dict[str, Any]], embedder):
        subtasks = [d["subtask"] for d in data]
        agents = [d["agent_description"] for d in data]

        # Batch-encode all texts upfront (FROZEN embedder)
        self.subtask_embs = embedder.encode(
            subtasks, convert_to_tensor=True, show_progress_bar=False
        )
        self.agent_embs = embedder.encode(
            agents, convert_to_tensor=True, show_progress_bar=False
        )
        self.scores = torch.tensor(
            [float(d["score"]) for d in data], dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.scores)

    def __getitem__(self, idx: int):
        concat = torch.cat([self.subtask_embs[idx], self.agent_embs[idx]])
        return concat, self.scores[idx]


# ── Trainer ─────────────────────────────────────────────────────────


class RewardModelTrainer:
    """Train the RewardMLP on generated training data."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
    ):
        self.embedding_model = embedding_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = _get_embedder(embedding_model)

    def train(
        self,
        data_path: Path,
        output_path: Path = Path("models/reward_mlp.pt"),
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        val_split: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Train the MLP on training data.

        Args:
            data_path: Path to training data JSON.
            output_path: Where to save trained weights.
            epochs: Number of training epochs.
            batch_size: Mini-batch size.
            lr: Learning rate.
            val_split: Fraction of data for validation.

        Returns:
            Training metadata dict with loss history.
        """
        # Load data
        raw = json.loads(data_path.read_text(encoding="utf-8"))
        if not raw:
            raise ValueError(f"No training data found in {data_path}")

        print(f"[Trainer] Loaded {len(raw)} examples from {data_path}")

        # Train/val split
        split_idx = max(1, int(len(raw) * (1 - val_split)))
        train_data = raw[:split_idx]
        val_data = raw[split_idx:] if split_idx < len(raw) else []

        print(f"[Trainer] Train: {len(train_data)}, Val: {len(val_data)}")

        # Build datasets
        train_dataset = RewardDataset(train_data, self.embedder)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if val_data:
            val_dataset = RewardDataset(val_data, self.embedder)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Initialise model
        model = RewardMLP(input_dim=768).to(self.device)
        criterion = nn.MSELoss()
        optimiser = torch.optim.Adam(model.parameters(), lr=lr)

        # Training loop
        history: List[Dict[str, float]] = []
        t0 = time.time()

        for epoch in range(1, epochs + 1):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                predictions = model(batch_x)
                loss = criterion(predictions, batch_y)

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(1, n_batches)

            # Validation
            avg_val_loss = None
            if val_loader:
                model.eval()
                val_loss = 0.0
                val_batches = 0
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx = vx.to(self.device)
                        vy = vy.to(self.device)
                        vpred = model(vx)
                        val_loss += criterion(vpred, vy).item()
                        val_batches += 1
                avg_val_loss = val_loss / max(1, val_batches)

            entry = {"epoch": epoch, "train_loss": avg_train_loss}
            if avg_val_loss is not None:
                entry["val_loss"] = avg_val_loss
            history.append(entry)

            if epoch % 10 == 0 or epoch == 1:
                val_str = f", val_loss={avg_val_loss:.4f}" if avg_val_loss else ""
                print(
                    f"  Epoch {epoch:3d}/{epochs}: "
                    f"train_loss={avg_train_loss:.4f}{val_str}"
                )

        elapsed = time.time() - t0
        print(f"[Trainer] Training complete in {elapsed:.1f}s")

        # Save model
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_path)
        print(f"[Trainer] Model saved to {output_path}")

        # Save metadata
        metadata = {
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "train_examples": len(train_data),
            "val_examples": len(val_data),
            "final_train_loss": history[-1]["train_loss"],
            "final_val_loss": history[-1].get("val_loss"),
            "training_time_s": round(elapsed, 1),
            "embedding_model": self.embedding_model,
            "device": self.device,
            "history": history,
        }
        meta_path = output_path.with_name("training_metadata.json")
        meta_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[Trainer] Metadata saved to {meta_path}")

        return metadata
