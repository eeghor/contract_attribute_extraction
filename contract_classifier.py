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
    # "qwen/qwen3.5-122b-a10b",
    "qwen/qwen3.5-flash-02-23",
]


# ── Pydantic output model ─────────────────────────────────────────────────────
class ContractClassification(BaseModel):
    contract_type_primary: str
    contract_type_secondary: list[str]
    subject_matter: str
    governing_law: str
    jurisdiction: str
    contract_language: str

    @field_validator(
        "contract_type_primary",
        "subject_matter",
        "governing_law",
        "jurisdiction",
        "contract_language",
    )
    def strip_whitespace(cls, v: str) -> str:
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator("contract_type_secondary")
    def ensure_list_of_str(cls, v: list) -> list[str]:
        if v is None:
            return []
        return [str(x).strip() for x in v]


# ── Prompt builder ────────────────────────────────────────────────────────────
def build_system_prompt() -> str:
    """
    Returns the built-in system prompt for contract classification. No file-based taxonomy.
    """
    return """You are an expert lawyer specialising in contract classification. Your task is to extract metadata from the provided contract text into a structured JSON format.

CONTRACT TYPE TAXONOMY

PRIMARY TYPES (Strict Hierarchy: If multiple types apply, pick the one highest on this list)
1. MODIFICATION_AND_CLOSURE: Documents that modify, extend, or terminate an existing agreement (e.g., Amendment, Addendum, Change Order, Termination Notice).
2. CORPORATE_AND_STRUCTURAL: Governing ownership, equity, or internal entity rules (e.g., Shareholders' Agreements, Articles of Association, SPAs, Board Resolutions).
3. FINANCE_AND_TREASURY: Documents where the "product" is capital or debt (e.g., Loan Agreements, Guarantees, Intercompany Loans).
4. INDIVIDUAL_LABOUR: Agreements with a natural person involving payroll or subordination (e.g., Employment Contracts, Executive Service Agreements). Note: If the worker is a company/Ltd, it is SERVICES.
5. REAL_ESTATE_AND_FACILITIES: Regarding physical land, buildings, or fixed infrastructure (e.g., Commercial Leases, Deeds, Construction/EPC).
6. IP_LICENSING_AND_TECH: Permission to use intangible property or digital platforms (e.g., SaaS Terms, Software Licenses, Trademark Licensing).
7. SUPPLY_AND_TRADE: The sale, distribution, or movement of physical goods (e.g., Purchase Orders, Supply Agreements, Logistics).
8. SERVICES: B2B or B2C agreements for human effort, time, or consulting (e.g., MSAs, Marketing Agency contracts).
9. LEGAL_SETTLEMENT_AND_RIGHTS: Standalone documents managing legal risk/disputes (e.g., Settlement & Release, Standalone NDAs, Power of Attorney).

SECONDARY TYPES (Tag All That Apply)
- DATA_PRIVACY: Includes GDPR, CCPA, or data processing clauses (DPA).
- CONFIDENTIALITY: Includes non-disclosure or secrecy obligations.
- IP_TRANSFER: Clauses transferring ownership of "Work Product" or "Inventions" to the client.
- RESTRICTIVE_COVENANTS: Includes Non-Compete, Non-Solicitation, or Non-Poaching.
- FINANCIAL_COMMITMENT: Contains a specific price, fee schedule, or fixed payment/invoicing terms.
- CROSS_BORDER: Parties have registered offices in different national jurisdictions.
- AUTOMATIC_RENEWAL: Contains "Evergreen" or auto-extension clauses.

TYPE TIE-BREAKER LOGIC:
1. THE MODIFICATION RULE: If the document amends or ends a previous contract, it is ALWAYS MODIFICATION_AND_CLOSURE, regardless of the subject matter.
2. THE SAAS VS. SERVICE RULE: If the buyer pays for access to an existing tool/platform, it is IP_LICENSING_AND_TECH. If they pay for custom development/person-hours, it is SERVICES.
3. THE INDIVIDUAL RULE: If the provider is a natural person and the contract mentions "Reporting lines," "Company equipment," or "Paid leave," it MUST be INDIVIDUAL_LABOUR.
4. THE ASSET RULE: If physical ownership (title) of a machine/good transfers, it is SUPPLY_AND_TRADE.

ALLOWED SUBJECT MATTERS (Pick exactly one):
Tangible Assets & Equipment: Procurement/maintenance of internal-use machinery, hardware, or vehicles (CapEx).
Real Estate & Facilities: Physical space: leasing, buying, or managing buildings and construction sites.
Professional & Operational Services: General B2B labor (Consulting, HR, Cleaning) NOT centered on a digital system.
Information Technology & Digital Systems: Software, SaaS, Cloud, and IT-implementation services.
Data Privacy & Cybersecurity: Focused strictly on the management and protection of personal/sensitive data.
Intellectual Property & Intangibles: Ownership/licensing of Trademarks, Patents, and Media (excluding Software).
Commercial Sales & Supply Chain: Trade, distribution, and logistics of goods for resale or manufacturing inputs (OpEx).
Workforce & Labor Relations: All relationships with natural persons (Employees, Executives, Individual Freelancers).
Corporate Governance & M&A: Entity structure, shareholder rights, and the buying/selling of companies.
Finance, Treasury & Banking: Debt, equity financing, banking services, and capital markets.
Research, Development & Innovation: Scientific research, clinical trials, and collaborative innovation.
Utilities, Energy & Infrastructure: Power, water, waste, and large-scale public works.
Marketing, Media & Sponsorship: External-facing growth: Advertising, PR, and Brand partnerships.
Legal, Regulatory & Risk Management: Settlements, compliance mandates, insurance, and standalone NDAs.

SUBJECT MATTER TIE-BREAKER LOGIC:
- Digital Primacy: If the subject involves software or digital infrastructure implementation, always use Information Technology & Digital Systems, even if it involves consulting services.
- The Services vs. Workforce Rule: If the provider is a company (Ltd/GmbH), use Professional & Operational Services. If the provider is a natural person, use Workforce & Labor Relations.
- The Supply Chain vs. Assets Rule: Use Commercial Sales & Supply Chain for recurring trade of goods (e.g. inventory). Use Tangible Assets & Equipment for one-off high-value purchases (e.g. factory robots).

RULES:
- Respond ONLY with a valid JSON object. No preamble or markdown.
- "contract_type_primary": Must be EXACTLY one label from the PRIMARY TYPES list. Follow the Hierarchy strictly.
- "contract_type_secondary": A JSON array of labels from the SECONDARY TYPES list. Empty array [] if none.
- "subject_matter": Must be verbatim from the ALLOWED SUBJECT MATTERS list (text before the colon).
- "governing_law": The exact law mentioned (e.g., "English law"). If not stated, "N/A".
- "jurisdiction": Extract the court venue and format it strictly as [CITY]_[ISO2_COUNTRY_CODE]_[COURT_TYPE] (all uppercase for city and country code). Rules:
    - CITY: the city name in UPPERCASE. If no city is mentioned (e.g., "Courts of France"), use NULL.
    - ISO2_COUNTRY_CODE: the 2-letter ISO 3166-1 alpha-2 country code in UPPERCASE (e.g., FR, DE, GB, IT, NL).
    - COURT_TYPE: must be exactly one of: General, Commercial, High, State, Federal, Chancery, International Commercial Court, Tribunal, Small Claims, Arbitration, IP, National.
      * If the text says "courts of [City]" with no further specificity: use General.
      * If the text says "Competent courts": use General.
      * If no city is mentioned but a country is (e.g., "Courts of France", "Italian courts"): use NULL for city and National for type.
      * Otherwise match the court type from context (e.g., "Commercial Court of Paris" → PARIS_FR_Commercial, "High Court of London" → LONDON_GB_High).
    - If no jurisdiction is stated at all: use "N/A".
    - Examples: PARIS_FR_Commercial, LONDON_GB_High, NULL_IT_National, AMSTERDAM_NL_General.
- "contract_language": The natural language the contract is written in (e.g., "English", "French", "German"). Detect from the document text itself.

EXAMPLE OUTPUT:
{{
  "contract_type_primary": "IP_LICENSING_AND_TECH",
  "contract_type_secondary": ["DATA_PRIVACY", "FINANCIAL_COMMITMENT"],
  "subject_matter": "Information Technology & Digital Systems",
  "governing_law": "Dutch law",
  "jurisdiction": "AMSTERDAM_NL_General",
  "contract_language": "English"
}}
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
        "max_tokens": 150,
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
            "contract_type_primary": result.contract_type_primary,
            "contract_type_secondary": result.contract_type_secondary,
            "subject_matter": result.subject_matter,
            "governing_law": result.governing_law,
            "jurisdiction": result.jurisdiction,
            "contract_language": result.contract_language,
            "raw_response": raw_content,
            "error": None,
        }

    except httpx.HTTPStatusError as e:
        return {
            "model": model,
            "contract_type_primary": None,
            "contract_type_secondary": [],
            "raw_response": None,
            "subject_matter": None,
            "governing_law": None,
            "jurisdiction": None,
            "contract_language": None,
            "error": f"HTTP {e.response.status_code}: {e.response.text}",
        }
    except json.JSONDecodeError as e:
        return {
            "model": model,
            "contract_type_primary": None,
            "contract_type_secondary": [],
            "raw_response": raw_content if "raw_content" in locals() else None,
            "subject_matter": None,
            "governing_law": None,
            "jurisdiction": None,
            "contract_language": None,
            "error": f"JSON parse error: {e}",
        }
    except Exception as e:
        return {
            "model": model,
            "contract_type_primary": None,
            "contract_type_secondary": [],
            "raw_response": None,
            "subject_matter": None,
            "governing_law": None,
            "jurisdiction": None,
            "contract_language": None,
            "error": str(e),
        }


# ── File loaders ──────────────────────────────────────────────────────────────
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

    print(f"\U0001f4c4 Contract file   : {contract_file}")
    print(f"\U0001f916 Models to query : {len(selected_models)}")
    print("-" * 60)

    system_prompt = build_system_prompt()
    user_prompt = build_user_prompt(contract_text)

    results = []
    for i, model in enumerate(selected_models, 1):
        print(f"[{i}/{len(selected_models)}] Querying {model} ...", end=" ", flush=True)
        result = call_openrouter(model, system_prompt, user_prompt)
        results.append(result)

        if result["error"]:
            print(f"❌ ERROR: {result['error']}")
        else:
            primary = result.get("contract_type_primary")
            secondary = result.get("contract_type_secondary")
            print(
                f"✅ → {primary} → {secondary} → {result.get('subject_matter')} → {result.get('governing_law')} → {result.get('jurisdiction')} → {result.get('contract_language')}"
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
    parser.add_argument(
        "--contract",
        required=True,
        help="Path to the contract text file to classify.",
    )
    args = parser.parse_args()

    results = classify_contract(
        contract_file=args.contract,
        delay_between_calls=args.delay,
    )

    output_path = Path(args.output)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n💾 Full results saved to: {output_path}")
