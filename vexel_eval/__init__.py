"""Vexel Evaluation Harness: High-precision quantitative SLM benchmarking library."""

from vexel_eval.schemas import (
    CatalystSignalSchema,
    CatalystType,
    Chain,
    DeFiSwapIntentSchema,
    DexRouter,
    DirectionBias,
    ExecutionAction,
    PolymarketDeltaSchema,
    SCHEMA_REGISTRY,
    get_schema,
    get_system_instruction,
)
from vexel_eval.runner import (
    BenchmarkRunner,
    BenchmarkSample,
    ModelBenchmarkMetrics,
    SampleResult,
    extract_json_candidate,
    evaluate_field_accuracy,
)
from vexel_eval.reporter import BenchmarkReporter

__version__ = "0.1.0"

__all__ = [
    "CatalystSignalSchema",
    "CatalystType",
    "Chain",
    "DeFiSwapIntentSchema",
    "DexRouter",
    "DirectionBias",
    "ExecutionAction",
    "PolymarketDeltaSchema",
    "SCHEMA_REGISTRY",
    "get_schema",
    "get_system_instruction",
    "BenchmarkRunner",
    "BenchmarkSample",
    "ModelBenchmarkMetrics",
    "SampleResult",
    "extract_json_candidate",
    "evaluate_field_accuracy",
    "BenchmarkReporter",
]
