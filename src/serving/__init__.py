"""Scoring service (Part E): application schema, scorer, reason codes and latency benchmark."""
from .batch import score_frame
from .latency import SAMPLE_APPLICATION, benchmark_latency
from .reasons import REASON_TEMPLATES, borrower_reasons, explain
from .schema import REQUIRED_APPLICATION_FIELDS, exchange_rate, prepare_single_application
from .scorer import decide, load_bundle, review_flags, score_application

__all__ = [
    "REASON_TEMPLATES",
    "REQUIRED_APPLICATION_FIELDS",
    "SAMPLE_APPLICATION",
    "benchmark_latency",
    "borrower_reasons",
    "decide",
    "exchange_rate",
    "explain",
    "load_bundle",
    "prepare_single_application",
    "review_flags",
    "score_application",
    "score_frame",
]
