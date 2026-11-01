"""Built-in Pydantic v2 quantitative trading schemas for Vexel benchmark tracks.

Tracks:
1. CatalystSignalSchema: High-urgency breaking news & macro/crypto event parsing.
2. PolymarketDeltaSchema: Prediction market probability delta & orderbook actioning.
3. DeFiSwapIntentSchema: Deterministic multi-chain on-chain DEX execution intent.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, Type
from pydantic import BaseModel, ConfigDict, Field, field_validator


# =====================================================================
# TRACK 1: Catalyst Signal Schema
# =====================================================================

class CatalystType(str, Enum):
    """Categorical classification of high-impact market catalysts."""
    REGULATORY_RULING = "REGULATORY_RULING"
    PROTOCOL_EXPLOIT = "PROTOCOL_EXPLOIT"
    TOKENOMICS_UPGRADE = "TOKENOMICS_UPGRADE"
    FED_RATE_DECISION = "FED_RATE_DECISION"
    ETF_FLOW = "ETF_FLOW"
    EXCHANGE_LISTING_DELISTING = "EXCHANGE_LISTING_DELISTING"
    GOVERNANCE_PROPOSAL = "GOVERNANCE_PROPOSAL"
    MAINNET_LAUNCH = "MAINNET_LAUNCH"
    PARTNERSHIP_ACQUISITION = "PARTNERSHIP_ACQUISITION"
    MACRO_LIQUIDITY_SHIFT = "MACRO_LIQUIDITY_SHIFT"


class DirectionBias(str, Enum):
    """Directional positioning bias for quantitative trading strategies."""
    LONG = "LONG"
    SHORT = "SHORT"
    NEUTRAL = "NEUTRAL"


class CatalystSignalSchema(BaseModel):
    """Schema for extracting high-frequency directional signals from breaking news."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    ticker: str = Field(
        ...,
        description="Standardized crypto asset symbol or ticker in uppercase (e.g. 'BTC', 'ETH', 'SOL').",
        min_length=1,
        max_length=12,
    )
    catalyst_type: CatalystType = Field(
        ...,
        description="Standardized classification of the breaking news catalyst.",
    )
    direction_bias: DirectionBias = Field(
        ...,
        description="Trade bias: LONG, SHORT, or NEUTRAL.",
    )
    urgency_score: float = Field(
        ...,
        description="Execution immediacy score from 0.0 (low) to 1.0 (sub-minute execution required).",
        ge=0.0,
        le=1.0,
    )
    confidence: float = Field(
        ...,
        description="Statistical confidence in directional signal extraction, from 0.0 to 1.0.",
        ge=0.0,
        le=1.0,
    )
    reasoning_60char_max: str = Field(
        ...,
        description="Strictly ultra-compact justification limited to 60 characters maximum.",
        max_length=60,
    )

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str) -> str:
        v_clean = v.strip().upper()
        if not v_clean:
            raise ValueError("Ticker cannot be empty")
        return v_clean

    @field_validator("reasoning_60char_max")
    @classmethod
    def validate_reasoning_len(cls, v: str) -> str:
        clean = v.strip()
        if len(clean) > 60:
            raise ValueError(f"Reasoning exceeds 60 characters: {len(clean)} > 60")
        return clean


# =====================================================================
# TRACK 2: Polymarket Delta Schema
# =====================================================================

class ExecutionAction(str, Enum):
    """Binary prediction market orderbook execution directive."""
    BUY_YES = "BUY_YES"
    BUY_NO = "BUY_NO"
    SELL_YES = "SELL_YES"
    SELL_NO = "SELL_NO"
    HOLD_NO_ACTION = "HOLD_NO_ACTION"


class PolymarketDeltaSchema(BaseModel):
    """Schema for prediction market CLOB arbitrage and delta re-pricing."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    market_id: str = Field(
        ...,
        description="Polymarket condition ID, market slug, or token identifier.",
        min_length=3,
        max_length=128,
    )
    resolution_conflict_risk: bool = Field(
        ...,
        description="Flag indicating risk of ambiguous UMA oracle dispute or resolution delay.",
    )
    delta_yes_prob: float = Field(
        ...,
        description="Estimated probability delta on the 'YES' token relative to CLOB mid-market, from -1.0 to +1.0.",
        ge=-1.0,
        le=1.0,
    )
    execution_action: ExecutionAction = Field(
        ...,
        description="Action to execute on the Polymarket CLOB.",
    )
    confidence: float = Field(
        ...,
        description="Model confidence in probability revision, from 0.0 to 1.0.",
        ge=0.0,
        le=1.0,
    )

    @field_validator("market_id")
    @classmethod
    def normalize_market_id(cls, v: str) -> str:
        v_clean = v.strip().lower()
        if not v_clean:
            raise ValueError("Market ID cannot be empty")
        return v_clean


# =====================================================================
# TRACK 3: DeFi Swap Intent Schema
# =====================================================================

class Chain(str, Enum):
    """Supported execution blockchains."""
    ETHEREUM = "ETHEREUM"
    BASE = "BASE"
    ARBITRUM = "ARBITRUM"
    SOLANA = "SOLANA"
    POLYGON = "POLYGON"
    OPTIMISM = "OPTIMISM"


class DexRouter(str, Enum):
    """Deterministic routing protocol on the target blockchain."""
    UNISWAP_V3 = "UNISWAP_V3"
    UNISWAP_V4 = "UNISWAP_V4"
    JUPITER = "JUPITER"
    CURVE = "CURVE"
    AERODROME = "AERODROME"
    RAYDIUM = "RAYDIUM"


class DeFiSwapIntentSchema(BaseModel):
    """Schema for intent-based deterministic on-chain liquidity routing."""
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    chain: Chain = Field(
        ...,
        description="Execution blockchain network.",
    )
    dex_router: DexRouter = Field(
        ...,
        description="DEX protocol router targeted for the swap.",
    )
    token_in: str = Field(
        ...,
        description="Input token symbol or contract/mint address.",
        min_length=1,
        max_length=64,
    )
    token_out: str = Field(
        ...,
        description="Output token symbol or contract/mint address.",
        min_length=1,
        max_length=64,
    )
    amount_in_wei: str = Field(
        ...,
        description="Unsigned integer amount in base units / wei represented as a string to avoid IEEE-754 precision loss.",
    )
    max_slippage_bps: int = Field(
        ...,
        description="Maximum tolerated slippage in basis points (1 bps = 0.01%, 50 bps = 0.50%).",
        ge=0,
        le=10000,
    )
    deadline_sec: int = Field(
        ...,
        description="Transaction validity deadline in seconds from current block timestamp.",
        ge=1,
        le=86400,
    )

    @field_validator("amount_in_wei")
    @classmethod
    def validate_amount_wei(cls, v: str) -> str:
        v_clean = str(v).strip()
        if not v_clean.isdigit():
            raise ValueError(f"amount_in_wei must be a non-negative integer string, got: '{v}'")
        if int(v_clean) <= 0:
            raise ValueError(f"amount_in_wei must be greater than zero, got: '{v}'")
        return v_clean


# =====================================================================
# SCHEMA REGISTRY & HELPER UTILITIES
# =====================================================================

SCHEMA_REGISTRY: Dict[str, Type[BaseModel]] = {
    "catalyst": CatalystSignalSchema,
    "catalyst_signal": CatalystSignalSchema,
    "polymarket": PolymarketDeltaSchema,
    "polymarket_delta": PolymarketDeltaSchema,
    "defi_swap": DeFiSwapIntentSchema,
    "defi_intent": DeFiSwapIntentSchema,
}


def get_schema(name: str) -> Type[BaseModel]:
    """Retrieve schema class by canonical or alias name."""
    norm = name.strip().lower().replace("-", "_")
    if norm not in SCHEMA_REGISTRY:
        valid_keys = ", ".join(sorted(SCHEMA_REGISTRY.keys()))
        raise KeyError(f"Unknown schema track '{name}'. Available: {valid_keys}")
    return SCHEMA_REGISTRY[norm]


def get_system_instruction(schema_cls: Type[BaseModel]) -> str:
    """Generate deterministic system instruction forcing raw JSON conforming to schema."""
    json_schema = schema_cls.model_json_schema()
    return (
        "You are VexelQuant's high-precision quantitative execution model. "
        "Analyze the input signal and respond ONLY with a raw JSON object matching this schema. "
        "Do NOT output markdown backticks (no ```json). Do NOT output conversational text. "
        f"Schema: {json_schema}"
    )
