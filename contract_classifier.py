"""
Contract Type Classifier using OpenRouter API
Sends the same query to multiple small/efficient LLMs and returns structured JSON output.

Changes from previous version:
- contract_type: inferred ONLY from explicit contract title; "N/A" if no title found.
- subject_matter: free-text concise summary (no fixed list), 10 words or fewer.
- regulated_sectors: new field — list of EU-regulated sectors (NIS2/DORA/CER) detected
  in the contract, or ["N/A"] if none found.
- types_file / subjects_file CLI args and prompt injection removed accordingly.
- max_tokens raised to 150 to accommodate the regulated_sectors array.
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
MODELS = [
    "mistralai/mistral-nemo",
    "google/gemma-2-9b-it",
    "qwen/qwen-2.5-7b-instruct",
]

# ── EU regulated sectors (NIS2 / DORA / CER) ─────────────────────────────────
# Injected into the prompt as the closed list for regulated_sectors.
REGULATED_SECTORS = [
    "banking",
    "insurance",
    "financial markets",
    "digital infrastructure",
    "managed ICT services",
    "energy",
    "transport",
    "health",
    "pharmaceuticals",
    "water",
    "public administration",
    "defence",
    "space",
    "food",
    "waste management",
    "postal services",
    "chemicals",
    "nuclear",
]


# ── Pydantic output model ─────────────────────────────────────────────────────
class ContractClassification(BaseModel):
    contract_type: str
    subject_matter: str
    governing_law: str
    jurisdiction: str
    regulated_sectors: list[str]

    @field_validator("contract_type", "subject_matter", "governing_law", "jurisdiction")
    def strip_whitespace(cls, v: str) -> str:
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator("regulated_sectors")
    def strip_sector_whitespace(cls, v: list) -> list:
        return [s.strip() for s in v if isinstance(s, str)]


# ── Prompt builder ────────────────────────────────────────────────────────────
def build_system_prompt(regulated_sectors: list[str]) -> str:
    sectors_block = ", ".join(f'"{s}"' for s in regulated_sectors)
    return f"""You are an expert lawyer specialising in contract analysis.

Your task is to read the provided contract text and extract five fields, returned as a single JSON object.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIELD 1 — contract_type
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Extract the contract type from the EXPLICIT TITLE of the document only. Apply the following priority order:

1. STANDARD LEGAL INSTRUMENT NAME (preferred): If the title contains or clearly implies a recognised legal instrument name, extract and return that name in English (e.g. "Master Services Agreement", "Non-Disclosure Agreement", "Distribution Agreement"). A standard name is a short, conventional label for a legal instrument type — not a description of what the contract covers.

2. DESCRIPTIVE TITLE (fallback): If the title exists but does not contain a standard legal instrument name — for example it describes the subject or arrangement (e.g. "Conditions for Registration with the Good Seat Aggregator", "Terms of Access to the Mobility Platform") — return the full title translated to English verbatim.

3. "N/A": Only if the document has no identifiable title whatsoever.

- Look for the title in the first two pages.
- Do NOT infer a standard name from the contract body — only from the title itself.
- Do NOT invent a standard name if the title is purely descriptive.
- SELF-CHECK before outputting: ask yourself "does the standard name I am about to return appear as words in the document title?" If no, do not use it.
- A single generic word such as "Agreement" or "Contract" alone does NOT qualify as a standard legal instrument name. It must be a compound name identifying a specific instrument type (e.g. "Distribution Agreement", "Master Services Agreement", "Non-Disclosure Agreement"). A title containing only the word "Agreement" falls through to rule 2 (descriptive title).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIELD 2 — subject_matter
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Write a free-text summary of the contract's PRIMARY commercial subject in 10 words or fewer.
- Focus on what is being delivered, exchanged, or governed as the main obligation.
- Ignore ancillary clauses (confidentiality, data protection, IP boilerplate).
- Do not use legal mechanism language — describe the commercial activity (e.g. "cloud software subscription for HR management", "distribution of consumer electronics in France").
Write the subject_matter summary in English regardless of the contract's language.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIELD 3 — governing_law
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The legal system chosen to interpret the contract.
- Translate to English and strip definite articles (e.g. "French law", "laws of State of New York").
- If not explicitly stated in the contract, return "N/A".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIELD 4 — jurisdiction
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The court or forum chosen for dispute resolution.
- Scan for keywords: "jurisdiction", "forum", "courts of", "submit to", "venue", "competent court".
- Translate to English and strip articles (e.g. "Paris courts", "courts of England and Wales").
- Do NOT confuse with governing law. A contract may state one without the other.
- If not explicitly stated, return "N/A".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIELD 5 — regulated_sectors
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Identify all EU-regulated sectors (under NIS2, DORA, or CER Directive) that this contract involves.
- A sector is "involved" only if the contract text contains a CONCRETE SIGNAL: the party operates in that sector, the subject matter is that sector's infrastructure or services, or the contract explicitly references that sector's regulations.
- Do NOT infer from generic commercial activity alone. A generic IT services contract with no sector-specific party or subject is not "digital infrastructure".
- Select ONLY from this list: {sectors_block}
- Return all that apply as a JSON array of strings using the exact values above.
- If no regulated sector is identifiable, return ["N/A"].

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Respond with ONLY a valid JSON object. No explanation, no markdown, no preamble.
- The JSON must have exactly five keys: "contract_type", "subject_matter", "governing_law", "jurisdiction", "regulated_sectors".
- "regulated_sectors" must always be a JSON array, even if it contains only one item or ["N/A"].
- All values in the JSON output MUST be translated to English. 

EXAMPLE OUTPUT:
{{"contract_type": "Master Services Agreement",
  "subject_matter": "IT outsourcing services for payment processing platform",
  "governing_law": "French law",
  "jurisdiction": "Paris courts",
  "regulated_sectors": ["banking", "digital infrastructure"]}}
"""


def build_user_prompt(contract_text: str) -> str:
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
        "Referer": "https://github.com/contract-classifier",
        "X-Title": "Contract Classifier",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 150,  # Raised from 80 to accommodate regulated_sectors array
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

        sectors_display = ", ".join(result.regulated_sectors)
        return {
            "model": model,
            "contract_type": result.contract_type,
            "subject_matter": result.subject_matter,
            "governing_law": result.governing_law,
            "jurisdiction": result.jurisdiction,
            "regulated_sectors": result.regulated_sectors,
            "raw_response": raw_content,
            "error": None,
        }

    except httpx.HTTPStatusError as e:
        return _error_result(model, f"HTTP {e.response.status_code}: {e.response.text}")
    except json.JSONDecodeError as e:
        raw = raw_content if "raw_content" in locals() else None
        return _error_result(model, f"JSON parse error: {e}", raw)
    except Exception as e:
        return _error_result(model, str(e))


def _error_result(model: str, error: str, raw_response: Optional[str] = None) -> dict:
    return {
        "model": model,
        "contract_type": None,
        "subject_matter": None,
        "governing_law": None,
        "jurisdiction": None,
        "regulated_sectors": None,
        "raw_response": raw_response,
        "error": error,
    }


# ── File loader ───────────────────────────────────────────────────────────────
def load_contract_text(filepath: str) -> str:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Contract file not found: {filepath}")
    return path.read_text(encoding="utf-8")


# ── Main runner ───────────────────────────────────────────────────────────────
def classify_contract(
    contract_file: str,
    models: Optional[list[str]] = None,
    delay_between_calls: float = 1.0,
) -> list[dict]:
    """
    Run classification across all models and return a list of results.

    Args:
        contract_file:        Path to the contract text file.
        models:               List of OpenRouter model IDs. Defaults to MODELS.
        delay_between_calls:  Seconds to wait between API calls (rate limit safety).
    """
    contract_text = load_contract_text(contract_file)
    selected_models = models or MODELS

    print(f"📄 Contract file   : {contract_file}")
    print(f"🤖 Models to query : {len(selected_models)}")
    print("-" * 60)

    system_prompt = build_system_prompt(REGULATED_SECTORS)
    user_prompt = build_user_prompt(contract_text)

    results = []
    for i, model in enumerate(selected_models, 1):
        print(f"[{i}/{len(selected_models)}] Querying {model} ...", end=" ", flush=True)
        result = call_openrouter(model, system_prompt, user_prompt)
        results.append(result)

        if result["error"]:
            print(f"❌ ERROR: {result['error']}")
        else:
            sectors = ", ".join(result["regulated_sectors"])
            print(
                f"✅ → {result['contract_type']}"
                f" | {result['subject_matter']}"
                f" | {result['governing_law']}"
                f" | {result['jurisdiction']}"
                f" | [{sectors}]"
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
        delay_between_calls=args.delay,
    )

    output_path = Path(args.output)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n💾 Full results saved to: {output_path}")
