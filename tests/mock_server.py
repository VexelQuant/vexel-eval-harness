"""Ultra-lightweight mock OpenAI/vLLM/llama.cpp HTTP server for live socket testing.

Zero external dependencies - uses standard library http.server.
Supports both SSE streaming (stream=True) and standard JSON completions.
"""

from __future__ import annotations

import json
import re
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional


SAMPLE_CATALYST_MAP = {
    "ethena": {"ticker": "ENA", "catalyst_type": "TOKENOMICS_UPGRADE", "direction_bias": "LONG", "urgency_score": 0.88, "confidence": 0.94, "reasoning_60char_max": "USDe reserves verified 100% liquid; short squeeze likely."},
    "ethereum etf": {"ticker": "ETH", "catalyst_type": "REGULATORY_RULING", "direction_bias": "LONG", "urgency_score": 0.95, "confidence": 0.98, "reasoning_60char_max": "Spot ETH options approved by SEC; institutional inflows."},
    "hyperliquid": {"ticker": "HYPE", "catalyst_type": "MAINNET_LAUNCH", "direction_bias": "LONG", "urgency_score": 0.82, "confidence": 0.91, "reasoning_60char_max": "Hyperliquid EVM mainnet live; record DEX fees collected."},
    "crvusd": {"ticker": "CRV", "catalyst_type": "MACRO_LIQUIDITY_SHIFT", "direction_bias": "NEUTRAL", "urgency_score": 0.65, "confidence": 0.89, "reasoning_60char_max": "crvUSD stabilized above 0.998; short delta neutralized."},
    "federal reserve": {"ticker": "BTC", "catalyst_type": "FED_RATE_DECISION", "direction_bias": "LONG", "urgency_score": 0.98, "confidence": 0.97, "reasoning_60char_max": "50bps emergency rate cut expands dollar liquidity pool."},
    "aave": {"ticker": "AAVE", "catalyst_type": "PROTOCOL_EXPLOIT", "direction_bias": "SHORT", "urgency_score": 0.99, "confidence": 0.95, "reasoning_60char_max": "Aave pauses market on exploit fear; immediate downside."},
    "binance": {"ticker": "XMR", "catalyst_type": "EXCHANGE_LISTING_DELISTING", "direction_bias": "SHORT", "urgency_score": 0.96, "confidence": 0.99, "reasoning_60char_max": "Binance privacy delisting removes 70% spot liquidity."},
    "blackrock": {"ticker": "SOL", "catalyst_type": "PARTNERSHIP_ACQUISITION", "direction_bias": "LONG", "urgency_score": 0.85, "confidence": 0.93, "reasoning_60char_max": "BlackRock BUIDL expands to Solana; major TVL expansion."},
    "monad": {"ticker": "MONAD", "catalyst_type": "MAINNET_LAUNCH", "direction_bias": "LONG", "urgency_score": 0.78, "confidence": 0.87, "reasoning_60char_max": "Monad devnet reaches 10k TPS; layer-1 competition spikes."},
    "uniswap": {"ticker": "UNI", "catalyst_type": "GOVERNANCE_PROPOSAL", "direction_bias": "LONG", "urgency_score": 0.92, "confidence": 0.96, "reasoning_60char_max": "CFTC settlement clears UNI fee switch; revenue distribution."},
    "jump": {"ticker": "ETH", "catalyst_type": "MACRO_LIQUIDITY_SHIFT", "direction_bias": "SHORT", "urgency_score": 0.95, "confidence": 0.94, "reasoning_60char_max": "Jump liquidates massive stETH holding; market dumped."},
    "tether": {"ticker": "USDT", "catalyst_type": "TOKENOMICS_UPGRADE", "direction_bias": "NEUTRAL", "urgency_score": 0.40, "confidence": 0.92, "reasoning_60char_max": "Tether $5.2B profit confirms ultra-safe reserve backing."},
    "berachain": {"ticker": "BERA", "catalyst_type": "MAINNET_LAUNCH", "direction_bias": "LONG", "urgency_score": 0.86, "confidence": 0.90, "reasoning_60char_max": "Berachain mainnet launch locks $1.8B protocol liquidity."},
    "eigenlayer": {"ticker": "EIGEN", "catalyst_type": "TOKENOMICS_UPGRADE", "direction_bias": "SHORT", "urgency_score": 0.90, "confidence": 0.93, "reasoning_60char_max": "EIGEN token transferability unlocked; sell supply surge."},
    "makerdao": {"ticker": "MKR", "catalyst_type": "GOVERNANCE_PROPOSAL", "direction_bias": "NEUTRAL", "urgency_score": 0.60, "confidence": 0.85, "reasoning_60char_max": "Maker/Sky rebrand unpegs old DAI; protocol confusion."},
}


def match_payload(prompt: str) -> Dict[str, Any]:
    """Inspect prompt and match ground-truth or schema-conforming payload."""
    lower = prompt.lower()

    # 1. Polymarket orderbook check (check before catalyst to avoid headline keyword collisions)
    if "market_id" in lower or "clob_mid_price" in lower or "polymarket" in lower or "polymarketdeltasignal" in lower:
        return {
            "market_id": "will-fed-cut-rates",
            "resolution_conflict_risk": False,
            "delta_yes_prob": 0.15,
            "execution_action": "BUY_YES",
            "confidence": 0.95,
        }

    # 2. DeFi swap check
    if "defiswapintent" in lower or "dex_router" in lower or "amount_in_wei" in lower or "token_in" in lower or "aerodrome" in lower or "jupiter" in lower:
        return {
            "chain": "BASE",
            "dex_router": "AERODROME",
            "token_in": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
            "token_out": "0x4200000000000000000000000000000000000006",
            "amount_in_wei": "1000000000",
            "max_slippage_bps": 50,
            "deadline_sec": 300,
        }

    # 3. Catalyst keyword match
    for keyword, response in SAMPLE_CATALYST_MAP.items():
        if keyword in lower:
            return response

    # Fallback catalyst
    return {
        "ticker": "BTC",
        "catalyst_type": "FED_RATE_DECISION",
        "direction_bias": "LONG",
        "urgency_score": 0.95,
        "confidence": 0.98,
        "reasoning_60char_max": "Fed easing cycle drives macro risk asset expansion.",
    }


class MockLLMHandler(BaseHTTPRequestHandler):
    """Handles OpenAI-compatible /v1/chat/completions requests."""

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress standard access logs to avoid terminal noise."""
        pass

    def do_GET(self) -> None:
        if self.path in ("/health", "/v1/health", "/"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok", "service": "vexel-mock-llm"}')
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self) -> None:
        if not (self.path.endswith("/chat/completions") or self.path == "/v1/chat/completions"):
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        try:
            req_data = json.loads(body.decode("utf-8"))
        except Exception:
            req_data = {}

        stream = req_data.get("stream", False)
        messages = req_data.get("messages", [])
        prompt_content = " ".join(m.get("content", "") for m in messages if isinstance(m, dict))

        target_payload = match_payload(prompt_content)
        model_name = req_data.get("model", "vexel-qwen-2.5-1.5b-q4_k_m")

        # Simulate model-specific latency and markdown behaviors
        if "Base" in model_name:
            ttft_delay = 0.035
            chunk_delay = 0.004
            # 65% chance of markdown fence wrapper
            if hash(prompt_content) % 3 != 0:
                json_output = f"```json\n{json.dumps(target_payload)}\n```"
            else:
                json_output = json.dumps(target_payload)
        elif "Frontier" in model_name or "gpt-4o" in model_name.lower():
            ttft_delay = 0.100
            chunk_delay = 0.008
            json_output = json.dumps(target_payload)
        else:
            ttft_delay = 0.012
            chunk_delay = 0.002
            json_output = json.dumps(target_payload)

        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()

            # Emulate model TTFT
            time.sleep(ttft_delay)

            # Stream character chunks or token chunks
            chunk_size = max(4, len(json_output) // 6)
            for i in range(0, len(json_output), chunk_size):
                token = json_output[i : i + chunk_size]
                chunk_obj = {
                    "id": f"chatcmpl-mock-{int(time.time()*1000)}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": token},
                            "finish_reason": None,
                        }
                    ],
                }
                line = f"data: {json.dumps(chunk_obj)}\n\n"
                self.wfile.write(line.encode("utf-8"))
                self.wfile.flush()
                time.sleep(chunk_delay)

            # Final DONE chunk
            done_chunk = {
                "id": "chatcmpl-mock-done",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            self.wfile.write(f"data: {json.dumps(done_chunk)}\n\n".encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            time.sleep(ttft_delay)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            resp_obj = {
                "id": f"chatcmpl-mock-{int(time.time()*1000)}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": json_output},
                        "finish_reason": "stop",
                    }
                ],
            }
            self.wfile.write(json.dumps(resp_obj).encode("utf-8"))


def run_mock_server(port: int = 8765) -> ThreadingHTTPServer:
    """Start threaded mock HTTP server on 127.0.0.1:port."""
    server = ThreadingHTTPServer(("127.0.0.1", port), MockLLMHandler)
    return server


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    server = run_mock_server(port)
    print(f"Vexel Mock LLM Server listening on http://127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
