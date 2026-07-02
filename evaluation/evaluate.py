"""
Recall@10 Evaluation Engine for SHL Assessment Recommender.

Parses public conversation traces C1-C10, replays the turns against the local
Orchestrator, extracts recommended tests, and computes Mean Recall@10.

Design Decision: Automatic markdown parser to extract expected shortlists.
Improves: Evaluation rigor (behavioral validation, regression prevention).
"""

import os
import re
import sys
from pathlib import Path
from typing import Any

# Add project root to python path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from app.catalog.loader import CatalogStore
from app.retrieval.embeddings import generate_embeddings
from app.retrieval.vector_store import VectorStore
from app.retrieval.keyword_search import KeywordSearchEngine
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.ranker import Ranker
from app.agents.orchestrator import Orchestrator
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


def parse_conversation_file(filepath: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Parse markdown conversation trace to extract user inputs and expected shortlist.

    Args:
        filepath: Path to conversation md file.

    Returns:
        Tuple of (messages list, expected_names list).
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Parse turns to get user messages
    turns = content.split("### Turn ")
    messages: list[dict[str, str]] = []
    expected_names: list[str] = []

    for turn in turns[1:]:
        # Find User block
        user_match = re.search(r"\*\*User\*\*\s*\n*\s*>\s*(.*)", turn)
        if user_match:
            user_text = user_match.group(1).strip()
            # Handle multi-line blockquotes if user has them
            lines = [user_text]
            # check subsequent lines of blockquote
            turn_lines = turn.split("\n")
            for idx, line in enumerate(turn_lines):
                if "**User**" in line:
                    for sub_line in turn_lines[idx+1:]:
                        if sub_line.strip().startswith(">"):
                            sub_text = sub_line.replace(">", "").strip()
                            if sub_text and sub_text not in lines:
                                lines.append(sub_text)
                        elif sub_line.strip() == "" or "**Agent**" in sub_line:
                            break
            user_text = " ".join(lines).strip()
            messages.append({"role": "user", "content": user_text})

        # Find Agent block
        agent_match = re.search(r"\*\*Agent\*\*\s*\n*\s*(.*)", turn)
        if agent_match:
            agent_text = agent_match.group(1).strip()
            # We don't replay the exact agent replies since API is stateless,
            # but we need to keep track of assistant turns in messages list
            messages.append({"role": "assistant", "content": agent_text})

    # 2. Extract final shortlist table names
    # Matches rows in the markdown table e.g. | 1 | Smart Interview Live Coding | ...
    table_rows = re.findall(r"\|\s*\d+\s*\|\s*([^|]+)\s*\|", content)
    for row in table_rows:
        name = row.strip()
        if name and name not in expected_names:
            expected_names.append(name)

    # Let's clean expected_names (e.g. remove abbreviations or format mismatch)
    # Match names exactly from catalog
    return messages, expected_names


def run_evaluation():
    logger.info("Starting automated evaluation on 10 public conversation traces...")

    # Load store
    catalog = CatalogStore()
    catalog.load("data/enriched_catalog.json")

    # Build search engines
    texts = catalog.get_retrieval_texts()
    embeddings = generate_embeddings(texts)

    vector_store = VectorStore()
    vector_store.build(embeddings)

    keyword_engine = KeywordSearchEngine()
    keyword_engine.build(texts)

    retriever = HybridRetriever(catalog, vector_store, keyword_engine)
    ranker = Ranker()

    orchestrator = Orchestrator(catalog, retriever, ranker)

    traces_dir = Path("sample_conversations/GenAI_SampleConversations")
    if not traces_dir.exists():
        logger.error("Traces directory not found at: %s", traces_dir)
        sys.exit(1)

    trace_files = sorted(list(traces_dir.glob("C*.md")), key=lambda p: int(p.stem[1:]))

    recalls = []
    logger.info("-" * 80)
    logger.info(f"{'Trace':<10} | {'Expected Count':<15} | {'Matched Count':<15} | {'Recall@10':<12}")
    logger.info("-" * 80)

    for file in trace_files:
        messages, expected_names = parse_conversation_file(file)

        # We replay the user turns only. For a stateless API, we feed history.
        # Find the last turn where user says something and agent provides final shortlist.
        # We can construct the full history up to that point.
        # Let's extract the last user message turn
        replay_history = []
        for msg in messages:
            # We append user and assistant messages
            replay_history.append(msg)
            if msg["role"] == "user":
                # We can run orchestrator on this prefix
                pass

        # Since it is stateless, let's execute orchestrator on the full conversation history.
        # The orchestrator will output its reply, recommendations, and end_of_conversation.
        result = orchestrator.process_message(replay_history)
        recommendations = result.get("recommendations", [])

        # Match predicted names to expected names
        predicted_names = [rec["name"] for rec in recommendations]

        # Calculate Recall@10
        # Recall = (|expected intersection predicted|) / |expected|
        matched = 0
        for name in expected_names:
            # Check fuzzy or substring match
            name_norm = name.lower().strip()
            # Clean names like OPQ32r or SVAR to match catalog names
            matched_flag = False
            for pred in predicted_names:
                pred_norm = pred.lower().strip()
                if name_norm in pred_norm or pred_norm in name_norm:
                    matched_flag = True
                    break
            if matched_flag:
                matched += 1

        recall = matched / len(expected_names) if expected_names else 0.0
        recalls.append(recall)

        logger.info(f"{file.name:<10} | {len(expected_names):<15} | {matched:<15} | {recall:.4f}")

    mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
    logger.info("-" * 80)
    logger.info(f"Mean Recall@10: {mean_recall:.4f}")
    logger.info("-" * 80)


if __name__ == "__main__":
    run_evaluation()
