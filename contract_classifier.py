"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                       CONTRACT CLASSIFIER — OVERVIEW                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

WHAT THIS TOOL DOES
───────────────────
This script reads a contract document (plain text or Markdown exported from OCS)
and automatically extracts the following structured metadata from it:

  • Contract type       – What kind of agreement is this? (e.g. Employment,
                          SaaS licence, Shareholders' Agreement, NDA...)
  • Secondary tags      – What notable clauses does it contain? (e.g. GDPR
                          data-processing obligations, non-compete, auto-renewal...)
  • Subject matter      – What is the commercial topic of the contract?
                          (e.g. IT & Digital Systems, Real Estate, Finance...)
  • Governing law       – Which country's law applies? (e.g. "English law")
  • Jurisdiction        – Which court has authority to settle disputes?
                          Stored as three separate fields ready for SQL filtering:
                            - jurisdiction_city        (e.g. "PARIS")
                            - jurisdiction_country     (e.g. "FR")
                            - jurisdiction_court_type  (e.g. "COMMERCIAL")
  • Contract language   – The natural language the document is written in.

HOW IT WORKS (step by step)
────────────────────────────
  1. LOAD   – The contract file is read from disk.
  2. CLEAN  – Markdown formatting artefacts (headings, bold, rules) are stripped
              so they do not confuse the AI model.
  3. PROMPT – A detailed instruction set (the "system prompt") tells the AI
              exactly what to look for and how to label each field, including
              tie-breaker rules for ambiguous cases.
  4. QUERY  – The cleaned contract text is sent to one or more AI language
              models via the OpenRouter API (a gateway to multiple AI providers).
              Temperature is set to 0 so the model responds deterministically.
  5. PARSE  – The AI response is expected to be a strict JSON object. The code
              validates every field (types, allowed values, internal consistency)
              using Pydantic — a data-validation library. Malformed or partial
              responses are caught and surfaced as errors rather than silently
              producing wrong data.
  6. SAVE   – Results are written to a JSON file for downstream processing
              (e.g. import into a SQL database or review dashboard).

AI MODEL USED
─────────────
By default, Qwen 3.5 Flash is used via OpenRouter. This is a fast, cost-efficient
model well suited to structured extraction tasks. The MODELS list at the top of
the file can be extended to query multiple models and compare their outputs.

CONFIGURATION
─────────────
The script expects a single environment variable:
  OPENROUTER_API_KEY  – Your OpenRouter API key, typically stored in a .env file.

USAGE (command line)
────────────────────
  python contract_classifier.py --contract path/to/contract.txt
  python contract_classifier.py --contract path/to/contract.md --output results.json

OUTPUT FORMAT (one record per model queried)
────────────────────────────────────────────
  {
    "model":                  "qwen/qwen3.5-flash-02-23",
    "contract_type_primary":  "IP_LICENSING_AND_TECH",
    "contract_type_secondary": ["DATA_PRIVACY", "FINANCIAL_COMMITMENT"],
    "subject_matter":         "Information Technology & Digital Systems",
    "governing_law":          "Dutch law",
    "jurisdiction_city":      "AMSTERDAM",
    "jurisdiction_country":   "NL",
    "jurisdiction_court_type": "GENERAL",
    "contract_language":      "English",
    "raw_response":           "<raw JSON string from the model>",
    "error":                  null
  }

  If extraction fails for any reason, all classification fields are null and
  the "error" field explains what went wrong.
"""

import os
import re
import json
import time
from pathlib import Path
from typing import Optional

import httpx
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic.functional_validators import BeforeValidator
from typing import Annotated
from dotenv import load_dotenv

# ── Load environment ──────────────────────────────────────────────────────────
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise EnvironmentError("OPENROUTER_API_KEY not found in .env file.")

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# ── Models to try ─────────────────────────────────────────────────────────────
# The AI models that will be queried. Each model in this list processes the same
# contract independently, allowing you to compare outputs or build consensus logic.
# Models are identified by their OpenRouter model ID (provider/model-name format).
# Ordered roughly by expected legal-text capability (best first).
# To query multiple models, uncomment additional entries or add new ones.
MODELS = [
    # "qwen/qwen3.5-122b-a10b",  # Larger, slower, higher-quality alternative
    "qwen/qwen3.5-flash-02-23",  # Default: fast, cost-efficient, good at structured extraction
]


# ── Pydantic output model ─────────────────────────────────────────────────────
class ContractClassification(BaseModel):
    """
    The structured data record produced for each contract.

    This class defines the exact shape of the AI's output and enforces data
    quality rules before any result is accepted. Think of it as a form with
    strict validation: if the AI fills in a field incorrectly (wrong format,
    inconsistent values, missing required entry), the record is rejected and
    an error is returned instead of silently storing bad data.

    Fields
    ──────
    contract_type_primary    : The single best-fit contract category from a
                               fixed 9-item taxonomy (e.g. SERVICES, INDIVIDUAL_LABOUR).
                               A strict hierarchy resolves ambiguity — the highest-
                               ranked matching type always wins.

    contract_type_secondary  : Zero or more tags describing notable clause types
                               present in the contract (e.g. DATA_PRIVACY,
                               RESTRICTIVE_COVENANTS). Defaults to an empty list
                               if the AI returns null or omits the field.

    subject_matter           : The commercial topic of the contract, chosen verbatim
                               from a fixed 14-item list (e.g. "Real Estate & Facilities").

    governing_law            : The legal system that governs the contract
                               (e.g. "English law", "French law"). Stored as None
                               if not stated.

    jurisdiction_city        : The city where disputes must be brought
                               (e.g. "PARIS"). None if only a country is named
                               or if no jurisdiction clause exists.

    jurisdiction_country     : ISO 3166-1 alpha-2 country code of the court
                               (e.g. "FR", "GB"). None only if no jurisdiction
                               clause exists at all.

    jurisdiction_court_type  : The type of court specified (e.g. "COMMERCIAL",
                               "HIGH", "GENERAL"). None only if no jurisdiction
                               clause exists at all.

    contract_language        : The natural language the contract is written in,
                               detected from the document text (e.g. "English").

    Validation rules enforced automatically
    ───────────────────────────────────────
    - All string fields are stripped of leading/trailing whitespace.
    - Sentinel strings "NULL" and "N/A" returned by the AI are converted to
      Python None so downstream SQL storage receives proper NULL values.
    - jurisdiction_country and jurisdiction_court_type must both be present
      or both be absent — a partial jurisdiction entry is rejected.
    - jurisdiction_city cannot be set without a jurisdiction_country.
    """
    contract_type_primary: str
    contract_type_secondary: Annotated[list[str], BeforeValidator(lambda v: [] if v is None else v)] = Field(default_factory=list)
    subject_matter: str
    governing_law: str
    jurisdiction_city: Optional[str]
    jurisdiction_country: Optional[str]
    jurisdiction_court_type: Optional[str]
    contract_language: str

    @field_validator(
        "contract_type_primary",
        "subject_matter",
        "contract_language",
    )
    def strip_whitespace(cls, v: str) -> str:
        """Remove accidental leading/trailing whitespace from core string fields."""
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator("governing_law")
    def normalise_governing_law(cls, v: Optional[str]) -> Optional[str]:
        """Coerce NULL / N/A sentinel strings to None for consistency with jurisdiction fields."""
        if v is None:
            return None
        v = str(v).strip()
        return None if v.upper() in ("NULL", "N/A", "") else v


    @field_validator("jurisdiction_city", "jurisdiction_country", "jurisdiction_court_type")
    def normalise_jurisdiction_field(cls, v: Optional[str]) -> Optional[str]:
        """Uppercase, strip whitespace, and coerce empty / null strings to None."""
        if v is None:
            return None
        v = str(v).strip().upper()
        return None if v in ("", "NULL", "N/A") else v

    @model_validator(mode="after")
    def validate_jurisdiction_coherence(self) -> "ContractClassification":
        """
        Enforce that jurisdiction fields are coherent:
          - If all three are None → fine (no jurisdiction stated).
          - If jurisdiction_country or jurisdiction_court_type is populated,
            the other must also be populated (city is allowed to be None).
          - jurisdiction_city must not be set without country + court_type.
        """
        city = self.jurisdiction_city
        country = self.jurisdiction_country
        court_type = self.jurisdiction_court_type
        populated = [f for f in (country, court_type) if f is not None]
        if len(populated) not in (0, 2):
            raise ValueError(
                f"jurisdiction_country and jurisdiction_court_type must both be set or both be NULL. "
                f"Got: country={country!r}, court_type={court_type!r}"
            )
        if city is not None and country is None:
            raise ValueError(
                f"jurisdiction_city is set to {city!r} but jurisdiction_country is NULL — "
                "a city cannot be provided without a country."
            )
        return self


# ── Prompt builder ────────────────────────────────────────────────────────────
def build_system_prompt() -> str:
    """
    Build the instruction set sent to the AI model before it sees any contract text.

    The system prompt is the core of the classifier. It defines:
      - The contract type taxonomy and strict priority hierarchy.
      - The allowed subject matter categories.
      - Tie-breaker rules for genuinely ambiguous contracts
        (e.g. a SaaS agreement where a company also provides implementation services).
      - Precise output field definitions including allowed values, NULL semantics,
        and worked examples covering every jurisdiction pattern.

    The prompt is written once and reused for every model and every contract in a
    given run, so any change here affects all results immediately.
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
- "governing_law": The exact law mentioned (e.g., "English law"). If not stated, use NULL.
- "jurisdiction_city": The city of the court venue in UPPERCASE (e.g., "PARIS", "LONDON"). Use NULL if no city is mentioned (e.g. "Courts of France") or if no jurisdiction is stated at all.
- "jurisdiction_country": The 2-letter ISO 3166-1 alpha-2 country code in UPPERCASE (e.g., "FR", "DE", "GB", "IT", "NL"). Extract the country even when no city is mentioned (e.g. "Courts of France" → "FR", "Italian courts" → "IT"). Use NULL only if no jurisdiction is stated at all.
- "jurisdiction_court_type": Must be exactly one of (in UPPERCASE): "GENERAL", "COMMERCIAL", "HIGH", "STATE", "FEDERAL", "CHANCERY", "INTERNATIONAL COMMERCIAL COURT", "TRIBUNAL", "SMALL CLAIMS", "ARBITRATION", "IP", "NATIONAL". Rules:
      * If the text says "courts of [City]" with no further specificity: use "GENERAL".
      * If the text says "Competent courts": use "GENERAL".
      * If no city is mentioned but a country is (e.g., "Courts of France", "Italian courts"): use "NATIONAL".
      * Otherwise match from context (e.g., "Commercial Court of Paris" → "COMMERCIAL").
      * Use NULL only if no jurisdiction is stated at all.
    - Examples: jurisdiction_city="PARIS" jurisdiction_country="FR" jurisdiction_court_type="COMMERCIAL" | jurisdiction_city=NULL jurisdiction_country="IT" jurisdiction_court_type="NATIONAL" | jurisdiction_city="AMSTERDAM" jurisdiction_country="NL" jurisdiction_court_type="GENERAL".
- "contract_language": The natural language the contract is written in (e.g., "English", "French", "German"). Detect from the document text itself.

EXAMPLE OUTPUT (jurisdiction with city):
{{
  "contract_type_primary": "IP_LICENSING_AND_TECH",
  "contract_type_secondary": ["DATA_PRIVACY", "FINANCIAL_COMMITMENT"],
  "subject_matter": "Information Technology & Digital Systems",
  "governing_law": "Dutch law",
  "jurisdiction_city": "AMSTERDAM",
  "jurisdiction_country": "NL",
  "jurisdiction_court_type": "GENERAL",
  "contract_language": "English"
}}

EXAMPLE OUTPUT (jurisdiction country-only, no city):
{{
  "contract_type_primary": "SERVICES",
  "contract_type_secondary": ["FINANCIAL_COMMITMENT"],
  "subject_matter": "Professional & Operational Services",
  "governing_law": "French law",
  "jurisdiction_city": NULL,
  "jurisdiction_country": "FR",
  "jurisdiction_court_type": "NATIONAL",
  "contract_language": "French"
}}

EXAMPLE OUTPUT (no jurisdiction stated):
{{
  "contract_type_primary": "SERVICES",
  "contract_type_secondary": [],
  "subject_matter": "Professional & Operational Services",
  "governing_law": NULL,
  "jurisdiction_city": NULL,
  "jurisdiction_country": NULL,
  "jurisdiction_court_type": NULL,
  "contract_language": "English"
}}
"""


def clean_contract_text(text: str) -> str:
    """
    Normalise contract text exported from OCS before sending it to the AI model.

    OCS can export contracts as plain text or as Markdown (a lightweight text
    format that uses symbols like # for headings and ** for bold). While Markdown
    is useful for human reading, its formatting symbols are noise from the AI's
    perspective — they consume context space without adding legal meaning.

    Artefacts removed
    ─────────────────
    - Heading markers (# / ## / ###)      — kept as plain text, symbol removed.
    - Horizontal rules  (--- / ***)       — visual dividers, removed entirely.
    - Bold/italic markers (** / * / __ / _) — formatting only, markers removed.
    - Redundant blank lines (3 or more)   — collapsed to a single blank line.
    - Unicode whitespace variants          — non-breaking spaces, zero-width spaces,
                                            and similar invisible characters that
                                            can silently break text matching.

    Plain-text exports are returned unchanged except for whitespace normalisation.
    """
    # Normalise Unicode whitespace (NBSP, thin space, zero-width space, etc.)
    text = re.sub(u"[\u00a0\u200b\u200c\u200d\u2009\u202f\ufeff]", " ", text)
    # Strip Markdown headings (keep the heading text itself)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Strip horizontal rules
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    # Strip bold / italic markers (**, *, __, _)
    text = re.sub(r"(\*\*|__)(.+?)\1", r"\2", text, flags=re.DOTALL)
    text = re.sub(r"(\*|_)(.+?)\1", r"\2", text, flags=re.DOTALL)
    # Collapse 3+ consecutive blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_user_prompt(contract_text: str) -> str:
    """
    Prepare the contract text as the user-side message sent to the AI model.

    The contract is first cleaned (see clean_contract_text), then wrapped in
    XML-style <contract_text> tags. These tags serve two purposes:
      1. They create a clear boundary between the AI's instructions and the
         contract content, preventing the contract text from being mistaken
         for additional instructions (a risk known as "prompt injection").
      2. They help the model focus its attention on the right section of input.

    A short reminder to respond only with JSON is appended at the end, exploiting
    the model's tendency to give extra weight to the most recent instructions.
    """
    contract_text = clean_contract_text(contract_text)
    return f"""Classify the following contract document.

<contract_text>
{contract_text}
</contract_text>

Respond with ONLY the JSON object as specified. No other text."""


# ── API call ──────────────────────────────────────────────────────────────────
def call_openrouter(
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: int = 60,
) -> dict:
    """
    Send a classification request to a single AI model via OpenRouter and return
    a validated, normalised result dictionary.

    OpenRouter is an API gateway that provides access to many AI models through
    a single, unified interface. This function handles the full request lifecycle:

      1. Sends the system prompt (instructions) and user prompt (contract text)
         to the specified model.
      2. Strips any Markdown code fences the model may have wrapped around its
         JSON output, despite being instructed not to.
      3. Parses the raw JSON response.
      4. Validates and normalises every field via the ContractClassification model.
      5. Returns a flat dictionary ready for serialisation or database insertion.

    Temperature is set to 0.0 so the model's output is as deterministic as
    possible — the same contract should produce the same result on each run.
    max_tokens is capped at 300, which is sufficient for a full JSON response
    with all fields populated, while preventing runaway generation.

    On any failure (network error, malformed JSON, validation error), all
    classification fields are set to None and the "error" key explains what
    went wrong, so the caller can log and continue rather than crashing.

    Args
    ────
    model          : OpenRouter model ID (e.g. "qwen/qwen3.5-flash-02-23").
    system_prompt  : The classification instructions built by build_system_prompt().
    user_prompt    : The contract text wrapped by build_user_prompt().
    timeout        : Maximum seconds to wait for an API response (default: 60).

    Returns
    ───────
    A dict with keys: model, contract_type_primary, contract_type_secondary,
    subject_matter, governing_law, jurisdiction_city, jurisdiction_country,
    jurisdiction_court_type, contract_language, raw_response, error.
    """
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
        # Optimisation: limit tokens — raised to 300 to avoid truncation on verbose responses
        "max_tokens": 300,
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
        raw_content = re.sub(r"^```(?:json)?\s*", "", raw_content, flags=re.IGNORECASE)
        raw_content = re.sub(r"\s*```$", "", raw_content).strip()

        parsed_json = json.loads(raw_content)
        result = ContractClassification(**parsed_json)

        return {
            "model": model,
            "contract_type_primary": result.contract_type_primary,
            "contract_type_secondary": result.contract_type_secondary,
            "subject_matter": result.subject_matter,
            "governing_law": result.governing_law,
            "jurisdiction_city": result.jurisdiction_city,
            "jurisdiction_country": result.jurisdiction_country,
            "jurisdiction_court_type": result.jurisdiction_court_type,
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
            "jurisdiction_city": None,
            "jurisdiction_country": None,
            "jurisdiction_court_type": None,
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
            "jurisdiction_city": None,
            "jurisdiction_country": None,
            "jurisdiction_court_type": None,
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
            "jurisdiction_city": None,
            "jurisdiction_country": None,
            "jurisdiction_court_type": None,
            "contract_language": None,
            "error": str(e),
        }


# ── File loaders ──────────────────────────────────────────────────────────────
def load_contract_text(filepath: str) -> str:
    """
    Read a contract file from disk and return its full text content.

    Accepts plain text (.txt) or Markdown (.md) files exported from OCS.
    The file must be UTF-8 encoded, which is the standard for OCS exports.
    Raises a clear FileNotFoundError if the path does not exist, rather than
    letting a cryptic OS error propagate to the caller.
    """
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
    Classify a single contract file using one or more AI models and return all results.

    This is the main entry point for programmatic use of the classifier. It
    orchestrates the full pipeline: loading the file, building prompts, querying
    each model in sequence, and collecting results.

    A configurable delay between API calls prevents hitting OpenRouter's rate
    limits when querying multiple models back-to-back.

    Args
    ────
    contract_file        : Path to the contract text file (.txt or .md from OCS).
    models               : List of OpenRouter model IDs to query. If not provided,
                           defaults to the MODELS list defined at the top of this file.
    delay_between_calls  : Seconds to pause between successive API calls.
                           Increase this if you encounter rate-limit errors (default: 1.0).

    Returns
    ───────
    A list of result dictionaries, one per model queried. Each dict contains all
    extracted classification fields plus a "raw_response" and "error" key.
    Failed model calls are included in the list with error details rather than
    being silently dropped, so you always get one record per model per contract.
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
                f"✅ → {primary} → {secondary} → {result.get('subject_matter')} → {result.get('governing_law')} → {result.get('jurisdiction_city')}|{result.get('jurisdiction_country')}|{result.get('jurisdiction_court_type')} → {result.get('contract_language')}"
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
