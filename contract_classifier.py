"""
Contract Type Classifier using OpenRouter API
Sends the same query to multiple small/efficient LLMs and returns structured JSON output.
"""

import os
import json
import time
from pathlib import Path
from typing import Optional

import httpx
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv

# ── Load environment ──────────────────────────────────────────────────────────
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise EnvironmentError("OPENROUTER_API_KEY not found in .env file.")

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# ── Models to try ─────────────────────────────────────────────────────────────
# Selected for strong instruction-following + text/legal parsing at small scale.
# Ordered roughly by expected legal-text capability (best first).
MODELS = [
    # Strong instruction followers with good reasoning at small size
    "mistralai/mistral-7b-instruct",  # Reliable, strong instruction following
    "mistralai/mistral-nemo",  # 12B, excellent at structured output
    "google/gemma-2-9b-it",  # Strong at classification tasks
    "qwen/qwen-2.5-7b-instruct",  # Qwen 2.5 excels at structured JSON
    "meta-llama/llama-3.2-3b-instruct",  # Llama 3.2 3B — very small but capable
    "meta-llama/llama-3.1-8b-instruct",  # Llama 3.1 8B — solid all-rounder
]


# ── Pydantic output model ─────────────────────────────────────────────────────
class ContractClassification(BaseModel):
    contract_type: str

    @field_validator("contract_type")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip()


# ── Prompt builder ────────────────────────────────────────────────────────────
def build_system_prompt(contract_types: list[str]) -> str:
    """
    Optimisation tricks applied:
    - Role assignment ("You are an expert legal analyst") primes legal reasoning.
    - Exhaustive contract type list is injected clearly with numbered items.
    - Strict output format is specified with a concrete example.
    - Model is told to reason internally but output ONLY JSON (reduces preamble noise).
    - "N/A" fallback is explicitly described to avoid hallucination of new categories.
    """
    types_block = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(contract_types))
    return f"""You are an expert legal analyst specialising in contract classification.

Your task is to read the provided contract text and identify its type from the list below.

ALLOWED CONTRACT TYPES:
{types_block}

RULES:
- You MUST respond with ONLY a valid JSON object. No explanation, no markdown, no preamble.
- The JSON must have exactly one key: "contract_type".
- The value must be EXACTLY one of the allowed contract types listed above (copy it verbatim), or "N/A" if none match.
- Do not invent new categories. Do not combine types. Do not add commentary.

EXAMPLE OUTPUT:
{{"contract_type": "Agency Agreement (Commercial Agency)"}}

If the document does not match any listed type, respond:
{{"contract_type": "N/A"}}"""


def build_user_prompt(contract_text: str) -> str:
    """
    Optimisation tricks applied:
    - Contract text is clearly delimited with XML-style tags to prevent prompt injection
      and help models separate instruction from content.
    - Reminder of output format at the end (recency bias in attention).
    """
    return f"""Classify the following contract document.

<contract_text>
{contract_text.strip()}
</contract_text>

Respond with ONLY the JSON object as specified. No other text."""


# ── API call ──────────────────────────────────────────────────────────────────
def call_openrouter(
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: int = 60,
) -> dict:
    """Call OpenRouter and return parsed result."""
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/contract-classifier",  # Optional but good practice
        "X-Title": "Contract Classifier",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        # Optimisation: low temperature for deterministic classification
        "temperature": 0.0,
        # Optimisation: limit tokens — a JSON response needs very few
        "max_tokens": 64,
        # Ask for JSON output where the API supports it (not all models do via OpenRouter)
        "response_format": {"type": "json_object"},
    }

    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(OPENROUTER_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        raw_content = data["choices"][0]["message"]["content"].strip()

        # Robustness: strip markdown code fences if model ignores the instruction
        if raw_content.startswith("```"):
            raw_content = raw_content.split("```")[1]
            if raw_content.startswith("json"):
                raw_content = raw_content[4:]
            raw_content = raw_content.strip()

        parsed_json = json.loads(raw_content)
        result = ContractClassification(**parsed_json)

        return {
            "model": model,
            "contract_type": result.contract_type,
            "raw_response": raw_content,
            "error": None,
        }

    except httpx.HTTPStatusError as e:
        return {
            "model": model,
            "contract_type": None,
            "raw_response": None,
            "error": f"HTTP {e.response.status_code}: {e.response.text}",
        }
    except json.JSONDecodeError as e:
        return {
            "model": model,
            "contract_type": None,
            "raw_response": raw_content if "raw_content" in dir() else None,
            "error": f"JSON parse error: {e}",
        }
    except Exception as e:
        return {
            "model": model,
            "contract_type": None,
            "raw_response": None,
            "error": str(e),
        }


# ── File loaders ──────────────────────────────────────────────────────────────
def load_contract_text(filepath: str) -> str:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Contract file not found: {filepath}")
    return path.read_text(encoding="utf-8")


def load_contract_types(filepath: str) -> list[str]:
    """
    Expects a plain text file with one contract type per line.
    Blank lines and lines starting with # are ignored.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Contract types file not found: {filepath}")
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


# ── Main runner ───────────────────────────────────────────────────────────────
def classify_contract(
    contract_file: str,
    types_file: str,
    models: Optional[list[str]] = None,
    delay_between_calls: float = 1.0,
) -> list[dict]:
    """
    Run classification across all models and return a list of results.

    Args:
        contract_file:        Path to the contract text file.
        types_file:           Path to the contract types list (one per line).
        models:               List of OpenRouter model IDs. Defaults to MODELS.
        delay_between_calls:  Seconds to wait between API calls (rate limit safety).
    """
    contract_text = load_contract_text(contract_file)
    contract_types = load_contract_types(types_file)
    selected_models = models or MODELS

    print(f"📄 Contract file   : {contract_file}")
    print(f"📋 Types file      : {types_file} ({len(contract_types)} types loaded)")
    print(f"🤖 Models to query : {len(selected_models)}")
    print("-" * 60)

    system_prompt = build_system_prompt(contract_types)
    user_prompt = build_user_prompt(contract_text)

    results = []
    for i, model in enumerate(selected_models, 1):
        print(f"[{i}/{len(selected_models)}] Querying {model} ...", end=" ", flush=True)
        result = call_openrouter(model, system_prompt, user_prompt)
        results.append(result)

        if result["error"]:
            print(f"❌ ERROR: {result['error']}")
        else:
            print(f"✅ → {result['contract_type']}")

        if i < len(selected_models):
            time.sleep(delay_between_calls)

    return results


def print_summary(results: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for r in results:
        status = r["contract_type"] if not r["error"] else f"ERROR: {r['error']}"
        print(f"  {r['model']:<50} → {status}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Classify a contract using multiple LLMs via OpenRouter."
    )
    parser.add_argument(
        "--contract", required=True, help="Path to the contract text file"
    )
    parser.add_argument(
        "--types",
        required=True,
        help="Path to the contract types list file (one per line)",
    )
    parser.add_argument(
        "--output",
        default="results.json",
        help="Path to save JSON results (default: results.json)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between API calls (default: 1.0)",
    )
    args = parser.parse_args()

    results = classify_contract(
        contract_file=args.contract,
        types_file=args.types,
        delay_between_calls=args.delay,
    )

    print_summary(results)

    output_path = Path(args.output)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n💾 Full results saved to: {output_path}")
