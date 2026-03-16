#!/usr/bin/env python3
"""
Generate training data for the neural reward model.

Usage:
    python scripts/generate_training_data.py [--queries QUERIES_FILE] [--output OUTPUT_PATH]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.orchestration.aop_coordinator import AOPCoordinator
from app.orchestration.scorer import LLMScorer
from app.orchestration.training_data_generator import TrainingDataGenerator
from scripts._bootstrap import bootstrap_registry

# Default diverse queries for training data generation.
DEFAULT_QUERIES = [
    # Informational
    "What is your refund policy?",
    "How do I cancel my subscription?",
    "Tell me about your return policy for electronics",
    "What documents do I need to open an account?",
    "How long does it take to process a refund?",
    "What are the fees for international transfers?",
    "Can you explain your privacy policy?",
    "What are your business hours?",
    "How do I reset my password?",
    "What payment methods do you accept?",
    "Tell me about your loyalty program",
    "What is the maximum withdrawal limit?",
    "How do I update my billing address?",
    "What are the requirements for a mortgage?",
    "Explain your dispute resolution process",
    # Action
    "I want a refund for order #12345",
    "Process a refund for my broken laptop, order ORD-9876",
    "Cancel my subscription effective immediately",
    "I need to dispute a charge of $50 on my statement",
    "Transfer $500 to account 123456789",
    "Close my savings account",
    "I want to upgrade my plan to premium",
    "Schedule a callback from customer service",
    "File a complaint about my recent service experience",
    "I need to report a fraudulent transaction on my card",
    # Mixed / multi-intent
    "What is your refund policy and can I get a refund for order #567?",
    "Tell me about premium features and upgrade my account",
    "I need help understanding my bill and disputing a charge",
    "How do I file a complaint? Also, I want to cancel order #789",
    "What are your shipping options and track my order ORD-2024",
]


def main():
    parser = argparse.ArgumentParser(description="Generate training data")
    parser.add_argument(
        "--queries",
        type=Path,
        default=None,
        help="JSON file with a list of query strings (uses built-in defaults if omitted)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/training_data/reward_training.json"),
        help="Output path for training data JSON",
    )
    parser.add_argument(
        "--agents-per-subtask",
        type=int,
        default=None,
        help="Number of agents to evaluate per subtask (default: half registry)",
    )
    args = parser.parse_args()

    # Load queries
    if args.queries and args.queries.exists():
        queries = json.loads(args.queries.read_text(encoding="utf-8"))
        print(f"Loaded {len(queries)} queries from {args.queries}")
    else:
        queries = DEFAULT_QUERIES
        print(f"Using {len(queries)} built-in default queries")

    # Initialise components
    registry, store = bootstrap_registry()
    aop = AOPCoordinator(registry=registry, performance_store=store)
    scorer = LLMScorer()

    generator = TrainingDataGenerator(
        aop_coordinator=aop,
        registry=registry,
        scorer=scorer,
        num_agents_per_subtask=args.agents_per_subtask,
    )

    data = generator.generate_from_queries(
        queries=queries,
        output_path=args.output,
    )
    print(f"\nDone! Generated {len(data)} training examples.")


if __name__ == "__main__":
    main()
