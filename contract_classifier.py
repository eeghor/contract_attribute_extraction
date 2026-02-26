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
    "microsoft/phi-4",
    "mistralai/mistral-nemo",  # 12B, excellent at structured output
    "google/gemma-2-9b-it",  # Strong at classification tasks
    "qwen/qwen-2.5-7b-instruct",  # Qwen 2.5 excels at structured JSON
    "meta-llama/llama-3.2-3b-instruct",  # Llama 3.2 3B — very small but capable
    "meta-llama/llama-3.1-8b-instruct",  # Llama 3.1 8B — solid all-rounder
]


# ── Pydantic output model ─────────────────────────────────────────────────────
class ContractClassification(BaseModel):
    contract_type: str
    governing_law: str
    subject_matter: str
    jurisdiction: str

    @field_validator("contract_type", "governing_law", "subject_matter", "jurisdiction")
    def strip_whitespace(cls, v: str) -> str:
        if isinstance(v, str):
            return v.strip()
        return v


# ── Prompt builder ────────────────────────────────────────────────────────────
def build_system_prompt(contract_types: list[str], subject_matters: list[str]) -> str:
    """
    Optimisation tricks applied:
    - Role assignment ("You are an expert lawyer") primes legal reasoning.
    - Exhaustive contract type list is injected clearly with numbered items.
    - Strict output format is specified with a concrete example.
    - Model is told to reason internally but output ONLY JSON (reduces preamble noise).
    - "N/A" fallback is explicitly described to avoid hallucination of new categories.
    """
    types_block = "\n".join(f"  {i+1}. {t}" for i, t in enumerate(contract_types))
    subject_matter_block = "\n".join(
        f"  {i+1}. {s}" for i, s in enumerate(subject_matters)
    )
    return f"""You are an expert lawyer specialising in contract classification.

Your task is to read the provided contract text and identify its type, main subject matter, governing law and jurisdiction. 
The contract types and subject matters you can choose from are strictly limited to the lists provided below.
Each contract type name is followed by a brief explanation separated by < to assist your decision.
Each subject matter name is followed by a brief refined definition separated by < to assist your decision.

ALLOWED CONTRACT TYPES:
{types_block}

ALLOWED SUBJECT MATTERS:
{subject_matter_block}

RULES:
- You MUST respond with ONLY a valid JSON object. No explanation, no markdown, no preamble.
- The JSON must have exactly four keys: "contract_type", "subject_matter", "governing_law", and "jurisdiction".
- The contract type value must be EXACTLY one of the allowed contract types listed above (copy it verbatim), or "N/A" if none match.
- Decide on contract type by legal mechanism first (MSA, distribution agreement, license agreement, etc.), not by subject matter. 
- The subject matter must be EXACTLY one of the allowed subject matters listed above (copy it verbatim), or "N/A" if none match.
- Do not invent new categories. Do not combine types. Do not add commentary.
- If several contract types are higly relevant, choose the type explicitly stated in the contract title.
- If several subject matters are higly relevant, choose the type with the more general scope.
- If the document is addendum or amendment, assign contract type based on the original contract type if possible.
- The governing law value should be the exact text from the contract translated to English and stripped of articles (e.g. "French law", "laws of State of California", etc.)
- To identify jurisdiction, scan the text for keywords: "Jurisdiction", "Forum", "Courts of", "Submit to", "Venue".
- If the document does not match any listed contract type, it's contract type is "N/A".
- If the document does not match any listed subject matter, it's main subject matter is "N/A".
- If the document does not explicitly state a governing law, it's governing law is "N/A".
- If the document does not explicitly state a jurisdiction, it's jurisdiction is "N/A".
- The jurisdiction value should be the exact text from the contract translated to English and stripped of articles (e.g. "Paris courts")
- Do NOT confuse Governing Law with Jurisdiction.

EXAMPLE OUTPUT:
{{"contract_type": "Agency Agreement",
  "subject_matter": "Workforce & Labor Relations",
  "governing_law": "French law",
  "jurisdiction": "Paris courts"}}
"""


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
        "Referer": "https://github.com/contract-classifier",  # Optional but good practice
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
        "max_tokens": 80,
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
            "subject_matter": result.subject_matter,
            "governing_law": result.governing_law,
            "jurisdiction": result.jurisdiction,
            "raw_response": raw_content,
            "error": None,
        }

    except httpx.HTTPStatusError as e:
        return {
            "model": model,
            "contract_type": None,
            "raw_response": None,
            "subject_matter": None,
            "governing_law": None,
            "jurisdiction": None,
            "error": f"HTTP {e.response.status_code}: {e.response.text}",
        }
    except json.JSONDecodeError as e:
        return {
            "model": model,
            "contract_type": None,
            "raw_response": raw_content if "raw_content" in locals() else None,
            "subject_matter": None,
            "governing_law": None,
            "jurisdiction": None,
            "error": f"JSON parse error: {e}",
        }
    except Exception as e:
        return {
            "model": model,
            "contract_type": None,
            "raw_response": None,
            "subject_matter": None,
            "governing_law": None,
            "jurisdiction": None,
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
    cleaned = []
    for line in lines:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        cleaned.append(s)
    return cleaned


# ── Main runner ───────────────────────────────────────────────────────────────
def classify_contract(
    contract_file: str,
    types_file: str,
    subjects_file: Optional[str] = None,
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
    subject_matters = load_contract_types(subjects_file) if subjects_file else []
    selected_models = models or MODELS

    print(f"📄 Contract file   : {contract_file}")
    print(f"📋 Types file      : {types_file} ({len(contract_types)} types loaded)")
    print(f"🤖 Models to query : {len(selected_models)}")
    print("-" * 60)

    system_prompt = build_system_prompt(contract_types, subject_matters)
    user_prompt = build_user_prompt(contract_text)

    results = []
    for i, model in enumerate(selected_models, 1):
        print(f"[{i}/{len(selected_models)}] Querying {model} ...", end=" ", flush=True)
        result = call_openrouter(model, system_prompt, user_prompt)
        results.append(result)

        if result["error"]:
            print(f"❌ ERROR: {result['error']}")
        else:
            print(
                f"✅ → {result['contract_type']} → {result.get('subject_matter')} → {result['governing_law']} → {result.get('jurisdiction')}"
            )

        if i < len(selected_models):
            time.sleep(delay_between_calls)

    return results


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
        "--subjects",
        required=False,
        help="Path to the subject matters list file (one per line)",
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
        subjects_file=args.subjects,
        delay_between_calls=args.delay,
    )

    output_path = Path(args.output)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n💾 Full results saved to: {output_path}")
