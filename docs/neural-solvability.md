# Neural Solvability Estimator

The neural solvability estimator replaces TF-IDF cosine similarity with dense sentence embeddings and a trained MLP reward model for scoring (subtask, agent) pairs during AOP orchestration.

**Motivation:** TF-IDF struggles with lexical gaps — when a user's phrasing differs from an agent's description but the intent aligns semantically (e.g., "get my money back" vs "refund processing specialist"). Dense embeddings capture these semantic relationships.

---

## Architecture

### Scoring Formula

```
score = α × neural_similarity + β × historical_performance
```

- `α = 0.6` — Weight for embedding-based similarity
- `β = 0.4` — Weight for historical agent success rate from `PerformanceStore`

### Neural Similarity

The embedding similarity is computed using:

1. **Encoder:** `all-MiniLM-L6-v2` (sentence-transformers, 384-dimensional vectors, runs locally — no API calls)
2. **Inputs:** Subtask description concatenated with agent description text (built identically to the TF-IDF estimator via `_build_agent_text()`)
3. **MLP:** Takes the concatenated embedding pair (768d → 256 → 64 → 1) and outputs a solvability probability

### RewardMLP Architecture

```python
nn.Sequential(
    nn.Linear(768, 256),   # Concat of two 384d embeddings
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(256, 64),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(64, 1),
    nn.Sigmoid()           # Output ∈ [0, 1]
)
```

Trained weights are stored at `models/reward_mlp.pt`. If the file is absent, the MLP runs with random initialization (untrained baseline).

### Data Structures

```python
@dataclass
class NeuralSolvabilityScore:
    agent_id: str
    neural_sim: float          # MLP output
    historical_perf: float     # PerformanceStore average
    combined: float            # α × neural_sim + β × historical_perf
    agent_text: str            # Agent description used for scoring

@dataclass
class NeuralSolvabilityResult:
    subtask_scores: Dict[str, List[NeuralSolvabilityScore]]
    assignments: Dict[str, str]    # subtask → best agent
    coverage_ratio: float          # Fraction assigned
    missing: List[str]             # Unassigned subtasks
```

The result interface matches `SolvabilityResult` from the TF-IDF estimator, making them interchangeable.

---

## Training Pipeline

Training follows a three-stage process: generate data → train model → evaluate.

### Stage 1: Training Data Generation

**Script:** `scripts/generate_training_data.py`
**Module:** `app/orchestration/training_data_generator.py`

```powershell
PYTHONPATH=. python scripts/generate_training_data.py
```

The generator:
1. Decomposes 30 built-in queries using AOP task decomposition
2. Pre-ranks candidate agents using TF-IDF solvability (top-l per subtask)
3. Executes each candidate agent on the subtask
4. Scores responses using `LLMScorer` (gpt-4o-mini) on three dimensions:
   - **Correctness** — Is the answer factually accurate?
   - **Relevance** — Does it address the subtask?
   - **Completeness** — Is the response thorough?
5. Saves `(subtask_text, agent_description, average_score)` triples to `data/training_data/reward_training.json`

### Stage 2: Model Training

**Script:** `scripts/train_reward_model.py`
**Module:** `app/orchestration/reward_model_trainer.py`

```powershell
PYTHONPATH=. python scripts/train_reward_model.py
```

Training configuration:
- **Epochs:** 50
- **Batch size:** 32
- **Learning rate:** 1e-3
- **Optimizer:** Adam
- **Loss:** MSE
- **Validation split:** 10%

The trainer pre-computes all embeddings using `all-MiniLM-L6-v2` before training. Outputs:
- `models/reward_mlp.pt` — Trained model weights
- `models/training_metadata.json` — Training stats (epochs, final loss, sample count)

### Stage 3: Evaluation

**Script:** `scripts/run_comparison.py`
**Module:** `evaluation/solvability_comparison.py`

```powershell
PYTHONPATH=. python scripts/run_comparison.py
```

Compares TF-IDF and Neural estimators on 8 test scenarios with ground-truth agent assignments. Metrics:
- **Accuracy** — Correct top-1 agent assignment rate
- **Agreement rate** — How often both estimators pick the same agent
- **Lexical gap performance** — Accuracy on paraphrased queries
- **Latency** — Per-query inference time

Results are saved to `evaluation/comparison_results.json`.

---

## Runtime Integration

### Default Estimator Selection

The `AOPCoordinator` uses neural by default with automatic TF-IDF fallback:

```python
@staticmethod
def _default_estimator(store: PerformanceStore) -> _Estimator:
    try:
        from app.orchestration.neural_solvability_estimator import NeuralSolvabilityEstimator
        return NeuralSolvabilityEstimator(store)
    except Exception as exc:
        print(f"[AOP] WARNING: Failed to load neural estimator: {exc}")
        return SolvabilityEstimator(store)
```

If `torch` or `sentence-transformers` are not installed, the system falls back to TF-IDF with a warning.

### Hot-Swap API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/solvability-estimator` | Returns `{"kind": "neural", "options": ["neural", "tfidf"]}` |
| `PATCH` | `/solvability-estimator` | Switches estimator: `{"kind": "tfidf"}` → returns `{"kind": "tfidf"}` |

Switching takes effect immediately for subsequent requests.

### Frontend Toggle

The `EstimatorTogglePanel` component in the Explainability sidebar provides a UI for switching estimators:
- Always visible (not gated by message selection)
- Shows active estimator badge (Neural MLP / TF-IDF)
- Toggle buttons with active/inactive states
- Amber warning when backend is unreachable
- Refresh button to re-fetch from backend

---

## Scripts Reference

| Script | Purpose | Usage |
|--------|---------|-------|
| `scripts/generate_training_data.py` | Generate (subtask, agent, score) triples | `PYTHONPATH=. python scripts/generate_training_data.py` |
| `scripts/train_reward_model.py` | Train the RewardMLP on generated data | `PYTHONPATH=. python scripts/train_reward_model.py` |
| `scripts/run_comparison.py` | Compare Neural vs TF-IDF on test scenarios | `PYTHONPATH=. python scripts/run_comparison.py` |
| `scripts/_bootstrap.py` | Shared helper: loads agents from factory spec | Imported by other scripts |

All scripts require `PYTHONPATH=.` to resolve `app.*` imports.

---

## Dependencies

| Package | Required? | Purpose |
|---------|-----------|---------|
| `torch` | Optional | MLP training and inference |
| `sentence-transformers` | Optional | all-MiniLM-L6-v2 embeddings |

Both are optional. Without them, the system automatically falls back to TF-IDF solvability estimation. Install with:

```powershell
pip install torch sentence-transformers
```
