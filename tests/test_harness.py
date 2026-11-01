"""Pytest test suite for vexel-eval-harness: schemas, latency math, runner, and reporter."""

from __future__ import annotations

import json
import pytest
from pathlib import Path
from pydantic import ValidationError

from vexel_eval.schemas import (
    CatalystSignalSchema,
    CatalystType,
    Chain,
    DeFiSwapIntentSchema,
    DexRouter,
    DirectionBias,
    ExecutionAction,
    PolymarketDeltaSchema,
    get_schema,
)
from vexel_eval.runner import (
    BenchmarkRunner,
    BenchmarkSample,
    evaluate_field_accuracy,
    extract_json_candidate,
)
from vexel_eval.reporter import BenchmarkReporter


class TestSchemas:
    """Verify strict Pydantic v2 validation constraints."""

    def test_catalyst_schema_valid(self):
        valid_data = {
            "ticker": "eth",  # should normalize to ETH
            "catalyst_type": "REGULATORY_RULING",
            "direction_bias": "LONG",
            "urgency_score": 0.85,
            "confidence": 0.95,
            "reasoning_60char_max": "SEC approves spot ETF with immediate effect.",
        }
        obj = CatalystSignalSchema.model_validate(valid_data)
        assert obj.ticker == "ETH"
        assert obj.catalyst_type == CatalystType.REGULATORY_RULING
        assert obj.urgency_score == 0.85

    def test_catalyst_schema_reasoning_length_constraint(self):
        long_reasoning = "This is a very long chain-of-thought explanation that exceeds the sixty character limit set by the quant schema."
        data = {
            "ticker": "BTC",
            "catalyst_type": "FED_RATE_DECISION",
            "direction_bias": "LONG",
            "urgency_score": 0.9,
            "confidence": 0.9,
            "reasoning_60char_max": long_reasoning,
        }
        with pytest.raises(ValidationError):
            CatalystSignalSchema.model_validate(data)

    def test_catalyst_schema_urgency_bounds(self):
        data = {
            "ticker": "SOL",
            "catalyst_type": "MAINNET_LAUNCH",
            "direction_bias": "LONG",
            "urgency_score": 1.25,  # Invalid: > 1.0
            "confidence": 0.9,
            "reasoning_60char_max": "Firedancer live.",
        }
        with pytest.raises(ValidationError):
            CatalystSignalSchema.model_validate(data)

    def test_polymarket_delta_schema(self):
        valid = {
            "market_id": "WILL-FED-CUT-IN-DEC",
            "resolution_conflict_risk": False,
            "delta_yes_prob": 0.15,
            "execution_action": "BUY_YES",
            "confidence": 0.92,
        }
        obj = PolymarketDeltaSchema.model_validate(valid)
        assert obj.market_id == "will-fed-cut-in-dec"
        assert obj.execution_action == ExecutionAction.BUY_YES

    def test_polymarket_delta_bounds(self):
        invalid = {
            "market_id": "test",
            "resolution_conflict_risk": False,
            "delta_yes_prob": 1.5,  # Out of bounds (> 1.0)
            "execution_action": "BUY_YES",
            "confidence": 0.92,
        }
        with pytest.raises(ValidationError):
            PolymarketDeltaSchema.model_validate(invalid)

    def test_defi_swap_schema(self):
        valid = {
            "chain": "BASE",
            "dex_router": "AERODROME",
            "token_in": "USDC",
            "token_out": "WETH",
            "amount_in_wei": "50000000000",
            "max_slippage_bps": 50,
            "deadline_sec": 120,
        }
        obj = DeFiSwapIntentSchema.model_validate(valid)
        assert obj.chain == Chain.BASE
        assert obj.dex_router == DexRouter.AERODROME
        assert obj.amount_in_wei == "50000000000"

    def test_defi_swap_invalid_amount(self):
        invalid = {
            "chain": "BASE",
            "dex_router": "AERODROME",
            "token_in": "USDC",
            "token_out": "WETH",
            "amount_in_wei": "-1000",  # non-positive
            "max_slippage_bps": 50,
            "deadline_sec": 120,
        }
        with pytest.raises(ValidationError):
            DeFiSwapIntentSchema.model_validate(invalid)


class TestJsonExtractionAndAccuracy:
    """Verify parsing and deterministic field evaluation."""

    def test_extract_pure_raw_json(self):
        text = '{"ticker": "BTC", "direction_bias": "LONG"}'
        parsed, is_valid, is_raw = extract_json_candidate(text)
        assert is_valid is True
        assert is_raw is True
        assert parsed["ticker"] == "BTC"

    def test_extract_markdown_fenced_json(self):
        text = "```json\n{\"ticker\": \"ETH\", \"direction_bias\": \"SHORT\"}\n```"
        parsed, is_valid, is_raw = extract_json_candidate(text)
        assert is_valid is True
        assert is_raw is False
        assert parsed["ticker"] == "ETH"

    def test_extract_malformed_json(self):
        text = "This is not json at all."
        parsed, is_valid, is_raw = extract_json_candidate(text)
        assert is_valid is False
        assert parsed is None

    def test_field_accuracy_tolerance(self):
        gt = {"ticker": "BTC", "urgency_score": 0.85, "direction_bias": "LONG"}
        pred = {"ticker": "btc", "urgency_score": 0.87, "direction_bias": "LONG"}
        matches = evaluate_field_accuracy(pred, gt)
        assert matches["ticker"] is True
        assert matches["urgency_score"] is True  # within 0.05
        assert matches["direction_bias"] is True

        # Beyond tolerance
        pred2 = {"ticker": "BTC", "urgency_score": 0.95, "direction_bias": "SHORT"}
        matches2 = evaluate_field_accuracy(pred2, gt)
        assert matches2["urgency_score"] is False  # 0.10 diff > 0.05
        assert matches2["direction_bias"] is False


class TestRunnerAndReporter:
    """Verify benchmark execution simulation and report writing."""

    def test_simulate_benchmark_run_sync(self, tmp_path):
        sample = BenchmarkSample(
            id="test_001",
            input_text="Federal Reserve unexpectedly lowers rates.",
            ground_truth={
                "ticker": "BTC",
                "catalyst_type": "FED_RATE_DECISION",
                "direction_bias": "LONG",
                "urgency_score": 0.95,
                "confidence": 0.98,
                "reasoning_60char_max": "Emergency rate cut fuels macro liquidity.",
            },
        )
        runner = BenchmarkRunner(simulate=True)
        metrics = runner.run_benchmark_sync(
            samples=[sample],
            schema_cls=CatalystSignalSchema,
            profile="vexel-fine-tuned",
        )

        assert metrics.total_samples == 1
        assert metrics.valid_json_count == 1
        assert metrics.schema_compliant_count == 1
        assert metrics.ttft_p50_ms > 0
        assert metrics.latency_p50_ms > metrics.ttft_p50_ms

        # Report generation test
        reporter = BenchmarkReporter()
        report_file = tmp_path / "TEST_REPORT.md"
        generated_path = reporter.generate_markdown_report(metrics, output_path=report_file)

        assert generated_path.exists()
        content = generated_path.read_text(encoding="utf-8")
        assert "VexelQuant Research" in content
        assert "Executive Summary" in content
        assert "Latency & Token Economics" in content
