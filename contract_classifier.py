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
  • Regulated sectors   – Which EU-regulated sectors (NIS2 / DORA / CER / AI Act)
                          are touched by this contract, if any. Extracted by a
                          dedicated second AI call so it does not interfere with
                          the type classification.

HOW IT WORKS (step by step)
────────────────────────────
  1. LOAD    – The contract file is read from disk.
  2. CLEAN   – Markdown formatting artefacts (headings, bold, rules) are stripped
               so they do not confuse the AI model.
  3. PROMPT  – Three separate instruction sets are prepared:
               (a) Type classifier: what kind of contract is this, what does it
                   cover, which law and court apply, what language is it in?
               (b) Sector classifier: does this contract touch any EU-regulated
                   sector under NIS2, DORA, CER, or the AI Act?
               (c) Risk vector extractor: fill the eight interpretive risk
                   dimensions that deterministic rule-based tools cannot handle
                   (counterparty leverage, termination risk, data sensitivity,
                   IP ownership risk, payment risk, auto-renewal risk,
                   confidentiality scope, liability cap adequacy).
  4. QUERY   – For each AI model, all three prompts are sent sequentially
               (chained): call (a) completes first, then call (b), then call (c),
               each on the same cleaned text. Temperature is set to 0 for
               deterministic results.
  5. PARSE   – Each response is validated against its own Pydantic model.
               Malformed or partial responses surface as errors rather than
               silently producing wrong data.
  6. MERGE   – The two result dicts are merged into one flat record per model.
  7. SAVE    – Results are written to a JSON file for downstream processing
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
  python contract_classifier.py --contract path/to/scanned_contract_121.txt
      → saves to path/to/results-scanned_contract_121.json (auto-derived)
  python contract_classifier.py --contract path/to/contract.md --output my_output.json
      → saves to my_output.json (explicit override)

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
        "regulated_sectors":       ["Digital Infrastructure", "Banking & Financial Markets"],
    "raw_response":            "<raw JSON string from the type classifier>",
    "sector_raw_response":     "<raw JSON string from the sector classifier>",
    "error":                   null,
    "elapsed_type_s":          2.41,
    "elapsed_sector_s":        1.87,
    "elapsed_total_s":         4.28,
    "sector_error":            null
  }

  The pipeline makes three sequential AI calls per model per contract:
    1. Contract type classifier   -> fills all fields except regulated_sectors
                                     and risk_vector_*.
    2. Regulated sector classifier -> fills regulated_sectors using the EU
       NIS2 / DORA / CER / AI Act framework.
    3. Risk vector extractor      -> fills eight interpretive risk dimensions
       that deterministic extraction cannot handle (counterparty_leverage,
       termination_risk, data_sensitivity, ip_ownership_risk, payment_risk,
       auto_renewal_risk, confidentiality_scope, liability_cap_adequacy).

  If any call fails, its fields are null/[] and the corresponding error
  key explains what went wrong. A failure in one call does not suppress
  the results of the others.
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
    "nvidia/nemotron-3-super-120b-a12b:free"
    # "qwen/qwen3.5-27b"
    # "qwen/qwen3.5-flash-02-23",  # Default: fast, cost-efficient, good at structured extraction
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
    - If jurisdiction_city is set but jurisdiction_country is None (model
      failed to infer the country from the city), jurisdiction_city is
      silently set to None so the record remains coherent. The prompt
      instructs the model to always infer country from city, so this
      fallback should be rare in practice.
    """

    contract_type_primary: str
    contract_type_secondary: Annotated[
        list[str], BeforeValidator(lambda v: [] if v is None else v)
    ] = Field(default_factory=list)
    subject_matter: str
    governing_law: Annotated[
        Optional[str],
        BeforeValidator(
            lambda v: (
                None
                if v is None or str(v).strip().upper() in ("NULL", "N/A", "")
                else v
            )
        ),
    ] = None
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

    @field_validator(
        "jurisdiction_city", "jurisdiction_country", "jurisdiction_court_type"
    )
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
          - If jurisdiction_city is set but jurisdiction_country is None,
            the model failed to infer the country from the city name.
            Rather than rejecting the whole record, jurisdiction_city is
            silently set to None so the country/court_type pair remains
            coherent. The prompt instructs the model to always infer
            country from city, so this path should be rare in practice.
        """
        city = self.jurisdiction_city
        country = self.jurisdiction_country
        court_type = self.jurisdiction_court_type

        # Model failed to infer country from the city name: degrade
        # gracefully by nulling out the city rather than rejecting the record.
        if city is not None and country is None:
            self.jurisdiction_city = None
            city = None

        populated = [f for f in (country, court_type) if f is not None]
        if len(populated) not in (0, 2):
            raise ValueError(
                f"jurisdiction_country and jurisdiction_court_type must both be set or both be NULL. "
                f"Got: country={country!r}, court_type={court_type!r}"
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
- "jurisdiction_country": The 2-letter ISO 3166-1 alpha-2 country code in UPPERCASE (e.g., "FR", "DE", "GB", "IT", "NL"). ALWAYS infer the country — even when the country is not explicitly named — from either (a) an explicit country name, or (b) the city named in the jurisdiction clause (e.g., "Courts of Paris" → "FR", "Milan courts" → "IT", "courts of Amsterdam" → "NL", "London courts" → "GB"). Extract the country even when no city is mentioned (e.g. "Courts of France" → "FR", "Italian courts" → "IT"). Use NULL only if no jurisdiction is stated at all and no country can be inferred from any named city or region.
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
    text = re.sub("[\u00a0\u200b\u200c\u200d\u2009\u202f\ufeff]", " ", text)
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


# ── Allowed regulated sector labels ──────────────────────────────────────────
# Canonical labels matching the NIS2 / DORA / CER / AI Act taxonomy.
# The sector validator uses this set to reject any label the model invents.
VALID_SECTORS: frozenset = frozenset(
    {
        "Banking & Financial Markets",
        "Insurance & Pensions",
        "Digital Infrastructure",
        "Managed ICT & Security Services (MSP/MSSP)",
        "Energy",
        "Transport",
        "Health & Life Sciences",
        "Manufacturing (Critical Goods)",
        "Water & Wastewater",
        "Public Administration & Defence",
        "Space & Satellite Infrastructure",
        "Digital Providers (Platforms)",
        "Food, Waste & Postal Services",
        "Chemical & Nuclear",
        "Research Organizations",
        "High-Risk AI",  # AI Act horizontal layer — co-exists with any primary sector
    }
)


# ── Pydantic sector model ─────────────────────────────────────────────────────
class SectorClassification(BaseModel):
    """
    The structured output of the regulated-sector classifier.

    A single contract may touch multiple regulated sectors simultaneously
    (e.g. a cloud hosting agreement for a bank is both Digital Infrastructure
    and Banking & Financial Markets). All applicable sectors are returned.

    Fields
    ──────
    regulated_sectors : A list of sector labels from the EU NIS2 / DORA / CER /
                        AI Act taxonomy. Each label must exactly match one of the
                        entries in VALID_SECTORS. null / missing / ["N/A"] are all
                        normalised to [] to keep downstream SQL clean.

    Validation rules enforced automatically
    ───────────────────────────────────────
    - null / missing field → empty list (same pattern as contract_type_secondary).
    - ["N/A"] sentinel → empty list (no regulated sector identified).
    - Any label not in VALID_SECTORS is rejected, surfacing a clear validation
      error rather than silently storing a hallucinated sector name.
    - Each label is stripped of whitespace before matching.
    """

    regulated_sectors: Annotated[
        list[str],
        BeforeValidator(lambda v: [] if v is None else v),
    ] = Field(default_factory=list)

    @field_validator("regulated_sectors")
    @classmethod
    def validate_sector_labels(cls, v: list) -> list:
        """
        Strip whitespace from each label, normalise ["N/A"] to [],
        and reject any label not in the canonical VALID_SECTORS set.
        """
        cleaned = [str(s).strip() for s in v]
        if cleaned == ["N/A"]:
            return []
        invalid = [s for s in cleaned if s not in VALID_SECTORS]
        if invalid:
            raise ValueError(
                f"Unrecognised sector label(s): {invalid}. "
                f"Must be one of: {sorted(VALID_SECTORS)}"
            )
        return cleaned



# ── Risk vector Pydantic model ────────────────────────────────────────────────

# Canonical labels for categorical risk-vector fields.
# Using frozensets here mirrors the VALID_SECTORS pattern so validation is
# consistent and the allowed values are a single source of truth.
_LEVERAGE_LABELS: frozenset = frozenset({"SUPPLIER", "CUSTOMER", "BALANCED", "UNCLEAR"})
_OWNERSHIP_LABELS: frozenset = frozenset({"SUPPLIER", "CUSTOMER", "JOINT", "UNCLEAR"})
_SCOPE_LABELS: frozenset = frozenset({"UNILATERAL_SUPPLIER", "UNILATERAL_CUSTOMER", "MUTUAL", "NONE", "UNCLEAR"})


class RiskVectorExtraction(BaseModel):
    """
    The structured output of the risk-vector extractor (call 3).

    This model captures eight interpretive dimensions that deterministic
    rule-based tools (universal_claims, temporal_parser) cannot reliably
    extract because they require holistic reading of multiple clauses and
    legal judgment about context.  Each dimension is validated tightly to
    prevent hallucinated values from reaching downstream consumers.

    The eight dimensions are designed to be orthogonal — each captures a
    distinct risk axis — and together they form the "interpretive layer" of
    the full risk vector.  The remaining dimensions (liability cap amount,
    warranty period, temporal obligations, etc.) are filled deterministically
    by universal_claims and temporal_parser without LLM involvement.

    Fields
    ------
    counterparty_leverage : Which party holds the stronger negotiating position
        based on the asymmetry of obligations and protections in the contract?
        Assessed holistically across all clauses, not just one.
        Allowed values: "SUPPLIER" | "CUSTOMER" | "BALANCED" | "UNCLEAR"

    termination_risk : How exposed is the weaker party to sudden or unilateral
        termination?  Score 1 (very low) to 5 (very high).

    data_sensitivity : What is the sensitivity level of personal or confidential
        data exchanged or processed under this contract?
        Score 1 (no personal data) to 5 (special-category GDPR Art. 9 data).

    ip_ownership_risk : Who owns IP created in the course of performing this
        contract (work product, deliverables, inventions)?
        Allowed values: "SUPPLIER" | "CUSTOMER" | "JOINT" | "UNCLEAR"

    payment_risk : How exposed is the supplying party to non-payment or delayed
        payment?  Score 1 (very low) to 5 (very high).

    auto_renewal_risk : How difficult is it to prevent automatic renewal?
        Score 1 (no auto-renewal) to 5 (very short opt-out window or multi-year
        lock-in on renewal).

    confidentiality_scope : Structure of the confidentiality obligation.
        Allowed values: "MUTUAL" | "UNILATERAL_SUPPLIER" | "UNILATERAL_CUSTOMER"
        | "NONE" | "UNCLEAR"

    liability_cap_adequacy : Is the liability cap commercially adequate relative
        to the value and risk profile of the contract?
        Score 1 (no cap — unlimited exposure) to 5 (cap clearly adequate and
        proportionate, mutual, with standard carve-outs).

    Validation rules enforced automatically
    ----------------------------------------
    - Integer score fields are clamped to [1, 5].  Out-of-range values are
      clipped rather than rejected — an off-by-one is less harmful than a null.
    - Categorical string fields are uppercased and checked against their
      allowed label set.  Unknown values are coerced to "UNCLEAR" rather than
      raising an error.
    - null / missing score fields default to 3 (neutral midpoint).
    - null / missing categorical fields default to "UNCLEAR".
    - auto_renewal_risk defaults to 1 (not 3) because the absence of an
      auto-renewal clause is a concrete fact, not an unknown.
    """

    counterparty_leverage: Annotated[
        str,
        BeforeValidator(lambda v: "UNCLEAR" if v is None else str(v).strip().upper()),
    ] = "UNCLEAR"

    termination_risk: Annotated[
        int,
        BeforeValidator(lambda v: 3 if v is None else max(1, min(5, int(v)))),
    ] = 3

    data_sensitivity: Annotated[
        int,
        BeforeValidator(lambda v: 3 if v is None else max(1, min(5, int(v)))),
    ] = 3

    ip_ownership_risk: Annotated[
        str,
        BeforeValidator(lambda v: "UNCLEAR" if v is None else str(v).strip().upper()),
    ] = "UNCLEAR"

    payment_risk: Annotated[
        int,
        BeforeValidator(lambda v: 3 if v is None else max(1, min(5, int(v)))),
    ] = 3

    auto_renewal_risk: Annotated[
        int,
        BeforeValidator(lambda v: 1 if v is None else max(1, min(5, int(v)))),
    ] = 1

    confidentiality_scope: Annotated[
        str,
        BeforeValidator(lambda v: "UNCLEAR" if v is None else str(v).strip().upper()),
    ] = "UNCLEAR"

    liability_cap_adequacy: Annotated[
        int,
        BeforeValidator(lambda v: 3 if v is None else max(1, min(5, int(v)))),
    ] = 3

    @field_validator("counterparty_leverage")
    @classmethod
    def validate_leverage(cls, v: str) -> str:
        """Coerce unrecognised leverage labels to UNCLEAR rather than rejecting."""
        return v if v in _LEVERAGE_LABELS else "UNCLEAR"

    @field_validator("ip_ownership_risk")
    @classmethod
    def validate_ip_ownership(cls, v: str) -> str:
        """Coerce unrecognised IP-ownership labels to UNCLEAR rather than rejecting."""
        return v if v in _OWNERSHIP_LABELS else "UNCLEAR"

    @field_validator("confidentiality_scope")
    @classmethod
    def validate_confidentiality_scope(cls, v: str) -> str:
        """Coerce unrecognised scope labels to UNCLEAR rather than rejecting."""
        return v if v in _SCOPE_LABELS else "UNCLEAR"


def build_sector_prompt() -> str:
    """
    Build the system prompt for the regulated-sector classifier.

    This prompt is entirely separate from the contract type classifier and
    encodes the full EU NIS2 / DORA / CER / AI Act sector taxonomy, four
    critical cross-cutting classification rules, and per-sector signal lists.

    Key rules embedded
    ──────────────────
    - ACTIVITY OVER ENTITY: Classify by what the contract's subject matter
      does, not what industry label the signing party carries.
    - NO GROUP PRIVILEGE: Intra-group IT/Cloud/Security services are still
      classified as Digital Infrastructure or Managed ICT.
    - LEX SPECIALIS: Banking/Finance signals invoke DORA as the primary
      regulatory anchor, taking precedence over general NIS2 rules.
    - AI HORIZONTAL LAYER: AI used for recruitment, education, or biometric
      identification triggers "High-Risk AI" regardless of primary sector.
    """
    return """You are a regulatory compliance expert specialising in EU critical infrastructure law (NIS2, DORA, CER Directive, AI Act). Your task is to identify which regulated sectors are touched by the provided contract.

CLASSIFICATION LOGIC (CRITICAL — apply before assigning any sector)
1. ACTIVITY OVER ENTITY: Classify based on what the contract's subject matter involves, not the brand of the signing party. If a retail company is building a power plant, classify as Energy.
2. NO GROUP PRIVILEGE: Legal entities providing IT/Cloud/Security services to their own corporate group are classified as Digital Infrastructure or Managed ICT — not exempt.
3. LEX SPECIALIS: If Banking/Finance signals are present, DORA is the primary regulatory anchor and takes precedence over general NIS2 rules.
4. AI HORIZONTAL LAYER: Any contract involving AI for recruitment, education, or biometric identification must include "High-Risk AI" regardless of the primary sector.

REGULATED SECTORS AND THEIR SIGNALS

Each sector is described with three tags:
  <def>    — what entities and activities this sector covers
  <signals> — contract keywords and phrases that indicate this sector applies
  <note>   — special classification instructions (only where applicable)

<sector label="Banking & Financial Markets">
  <def>Credit institutions, investment firms, payment service providers, and financial market infrastructures.</def>
  <signals>banking licenses, payment processing, clearing, settlement, trading platforms, credit facilities, banks, insurers, asset managers, financial market operators</signals>
</sector>

<sector label="Insurance & Pensions">
  <def>Insurance undertakings, reinsurance companies, and occupational pension funds (Solvency II / IORP II).</def>
  <signals>insurance underwriting, policy administration, actuarial systems, claims processing, licensed insurers, pension fund administrators</signals>
</sector>

<sector label="Digital Infrastructure">
  <def>Internet exchange points (IXPs), DNS providers, TLD registries, cloud computing providers, data centres, and CDN providers.</def>
  <signals>data centre operations, cloud hosting (IaaS/PaaS), DNS/TLD management, internet exchange, CDN services</signals>
</sector>

<sector label="Managed ICT & Security Services (MSP/MSSP)">
  <def>Managed service providers and managed security service providers delivering outsourced IT or security operations.</def>
  <signals>outsourced IT management, managed SOC, remote infrastructure monitoring, third-party cybersecurity administration</signals>
</sector>

<sector label="Energy">
  <def>Electricity, gas, oil, hydrogen, and district heating operators — covering transmission, distribution, supply, and generation.</def>
  <signals>power generation, grid operation, gas transmission, oil pipelines, energy trading, smart metering, TSOs, DSOs, licensed energy suppliers</signals>
</sector>

<sector label="Transport">
  <def>Air, rail, maritime, and road transport operators and infrastructure managers.</def>
  <signals>airport operations, air traffic management, railway network management, port operations, vessel traffic services, road traffic management systems</signals>
</sector>

<sector label="Health & Life Sciences">
  <def>Hospitals, healthcare networks, and diagnostic or research laboratories.</def>
  <signals>patient data systems (EHR), clinical operations, licensed healthcare providers, medical diagnostics</signals>
</sector>

<sector label="Manufacturing (Critical Goods)">
  <def>Manufacturers of medical devices (MDR/IVDR), chemicals, computers and electronics, electrical equipment, machinery, and motor vehicles.</def>
  <signals>NACE codes C26–C30, medical device hardware, pharmaceutical manufacturing, automotive supply chains, industrial machinery production</signals>
</sector>

<sector label="Water & Wastewater">
  <def>Drinking water suppliers and wastewater treatment operators.</def>
  <signals>water treatment, distribution network management, SCADA systems for water infrastructure, public water utilities</signals>
</sector>

<sector label="Public Administration & Defence">
  <def>Central and regional government bodies and entities processing sensitive government information.</def>
  <signals>government agencies, classified information (EUCI), defence procurement, military systems</signals>
</sector>

<sector label="Space & Satellite Infrastructure">
  <def>Operators of ground-based infrastructure supporting space-based services.</def>
  <signals>satellite operations, ground station management, space-based positioning (PNT), earth observation infrastructure</signals>
</sector>

<sector label="Digital Providers (Platforms)">
  <def>Online marketplaces, search engines, and social networking service platforms.</def>
  <signals>user-generated content hosting, e-commerce platform operations, search algorithms, digital advertising marketplaces</signals>
</sector>

<sector label="Food, Waste & Postal Services">
  <def>Large-scale food production and processing companies, hazardous waste management operators, and universal postal service providers.</def>
  <signals>food safety certification (large scale), municipal waste systems, universal service obligation (USO) postal networks</signals>
</sector>

<sector label="Chemical & Nuclear">
  <def>Manufacturers of hazardous chemicals and operators of nuclear facilities.</def>
  <signals>REACH-regulated substances, chemical plant operations, nuclear power generation, radioactive material handling</signals>
</sector>

<sector label="Research Organizations">
  <def>Organisations conducting research activities directly related to any of the above regulated sectors.</def>
  <signals>scientific research grants in energy/health/space, laboratory services for critical entities</signals>
</sector>

<sector label="High-Risk AI">
  <def>AI systems used for recruitment or HR screening, educational assessment, or biometric identification and categorisation (EU AI Act Annex III).</def>
  <signals>AI-powered CV screening, automated hiring tools, student assessment AI, facial recognition, biometric access control</signals>
  <note>This is a HORIZONTAL label. It must be added alongside any primary sector label — never as the sole entry unless no other sector applies. Regulatory anchor: EU AI Act Annex III.</note>
</sector>

RULES:
- Respond ONLY with a valid JSON object. No preamble or markdown.
- "regulated_sectors": A JSON array of sector labels. You MUST use the label strings EXACTLY as they appear in the following canonical list — copy them character-for-character, including parenthetical suffixes:
    "Banking & Financial Markets"
    "Insurance & Pensions"
    "Digital Infrastructure"
    "Managed ICT & Security Services (MSP/MSSP)"
    "Energy"
    "Transport"
    "Health & Life Sciences"
    "Manufacturing (Critical Goods)"
    "Water & Wastewater"
    "Public Administration & Defence"
    "Space & Satellite Infrastructure"
    "Digital Providers (Platforms)"
    "Food, Waste & Postal Services"
    "Chemical & Nuclear"
    "Research Organizations"
    "High-Risk AI"
- Return [] if no sector applies. Do NOT invent, abbreviate, or paraphrase any label.
- Apply ALL sectors that are triggered — a single contract may touch several.
- Do NOT infer a sector from the party's general industry if the contract's subject matter does not involve that sector's regulated activities.

EXAMPLE OUTPUT (cloud hosting agreement for a bank):
{{
  "regulated_sectors": ["Digital Infrastructure", "Banking & Financial Markets"]
}}

EXAMPLE OUTPUT (standard marketing services agreement):
{{
  "regulated_sectors": []
}}

EXAMPLE OUTPUT (AI recruitment tool for a hospital):
{{
  "regulated_sectors": ["Health & Life Sciences", "High-Risk AI"]
}}
"""



def build_risk_vector_prompt() -> str:
    """
    Build the system prompt for the risk-vector extractor (call 3).

    This prompt asks the model to fill eight interpretive risk dimensions that
    deterministic rule-based tools cannot handle.  Every dimension has a
    bounded output type (integer 1-5 or a fixed categorical label) and worked
    examples calibrated to real contract patterns.

    Design principles
    -----------------
    - Bounded outputs only: integer scores and named categorical labels.
      Open-ended text is not used because it cannot be validated or compared.
    - Calibrated anchors: each score field includes explicit descriptions of
      what 1 and 5 mean so the model applies a consistent scale.
    - Null handling: the model is instructed to use specific sentinel values
      (null for categorical, 3 for scores) when a dimension cannot be assessed.
    - Independence from calls 1 and 2: this prompt stands alone; the risk
      vector is derived purely from the contract text, not from the earlier
      classifier outputs.
    """
    return """You are an expert legal risk analyst specialising in commercial contract review. Your task is to assess eight interpretive risk dimensions of the provided contract and return them as a structured JSON object.

These dimensions require legal judgment and holistic reading of the contract — they cannot be extracted mechanically from individual clauses.

RISK DIMENSIONS

1. counterparty_leverage
   Which party holds the stronger negotiating position, inferred from the asymmetry of obligations, protections, liability exposure, and termination rights across the whole contract?
   Allowed values (UPPERCASE, copy exactly):
     "SUPPLIER"  — the supplier/service provider/licensor bears disproportionately more obligations and risk
     "CUSTOMER"  — the customer/buyer/licensee holds most of the protective rights and limited obligations
     "BALANCED"  — obligations and protections are roughly symmetric between both parties
     "UNCLEAR"   — the contract is too short, too generic, or too ambiguous to assess

2. termination_risk
   How exposed is the weaker party to sudden or unilateral termination?
   Integer 1-5:
     1 = Very low: long bilateral notice (>=90 days), cure period, mutual specific triggers
     3 = Moderate: 30-60 days notice, some cure period, termination for convenience present but balanced
     5 = Very high: termination for convenience <=7 days notice, no cure, broad unilateral triggers
   Use 3 if no termination clause exists.

3. data_sensitivity
   What is the sensitivity level of personal or confidential data exchanged or processed under this contract?
   Integer 1-5:
     1 = No personal data; only B2B operational data
     2 = Basic contact/business data (name, email, role)
     3 = Standard personal data (profiles, usage data)
     4 = Sensitive personal data (financial accounts, HR/employment records, location)
     5 = Special-category data under GDPR Art. 9 (health, biometrics, race/ethnicity, criminal records)
   Use 1 if no data processing clause and no personal data signals.

4. ip_ownership_risk
   Who owns IP created in the course of performing this contract (deliverables, work product, inventions)?
   Allowed values (UPPERCASE, copy exactly):
     "SUPPLIER"  — supplier retains ownership; customer receives a licence only
     "CUSTOMER"  — IP is assigned to the customer (work-for-hire or explicit assignment clause)
     "JOINT"     — shared or joint ownership of newly created IP
     "UNCLEAR"   — no IP ownership clause, or clause is ambiguous

5. payment_risk
   How exposed is the supplying party to non-payment or delayed payment?
   Integer 1-5:
     1 = Very low: clear milestones, <=15-day terms, automatic late-payment interest, right to suspend
     3 = Moderate: standard 30-day terms, late-payment interest present, no suspension right
     5 = Very high: vague payment triggers, >=60-day terms, no late-payment remedy, no suspension right
   Use 3 if no payment clause exists (e.g. a standalone NDA).

6. auto_renewal_risk
   How difficult is it to prevent the contract from automatically renewing?
   Integer 1-5:
     1 = No auto-renewal clause at all
     2 = Auto-renewal present; opt-out window >=90 days
     3 = Auto-renewal; opt-out window 30-89 days
     4 = Auto-renewal; opt-out window <30 days
     5 = Very short or unclear opt-out window, multi-year lock-in on renewal, or no opt-out right
   Use 1 if no auto-renewal clause exists.

7. confidentiality_scope
   What is the structure of the confidentiality obligation?
   Allowed values (UPPERCASE, copy exactly):
     "MUTUAL"                — both parties bound by symmetric confidentiality obligations
     "UNILATERAL_SUPPLIER"   — only the supplier must keep the customer's information confidential
     "UNILATERAL_CUSTOMER"   — only the customer must keep the supplier's information confidential
     "NONE"                  — no confidentiality clause present
     "UNCLEAR"               — a confidentiality clause exists but its scope is ambiguous

8. liability_cap_adequacy
   Is the liability cap commercially adequate and proportionate to the value and risk of this contract?
   Integer 1-5:
     1 = No cap: unlimited exposure for at least one party
     2 = Cap present but very low relative to contract value (e.g. 1x one month's fees)
     3 = Cap roughly proportionate (e.g. 12x monthly fees or total contract value)
     4 = Cap generous (well above likely loss scenario, multiple carve-outs)
     5 = Cap clearly adequate and balanced: proportionate ceiling, mutual, standard carve-outs
   Use 3 if no cap exists and contract value is unclear.

RULES:
- Respond ONLY with a valid JSON object. No preamble or markdown.
- Use null for categorical fields when a clause is absent or too ambiguous to classify.
- Use integer 3 for score fields when the relevant clause is absent or the contract is too short.
- Use integers only for score fields — do not return strings like "3" in quotes.
- Do NOT invent new categorical labels. Use ONLY the values listed above.

EXAMPLE OUTPUT (SaaS agreement, customer-favourable):
{{
  "counterparty_leverage": "CUSTOMER",
  "termination_risk": 3,
  "data_sensitivity": 3,
  "ip_ownership_risk": "CUSTOMER",
  "payment_risk": 2,
  "auto_renewal_risk": 1,
  "confidentiality_scope": "MUTUAL",
  "liability_cap_adequacy": 3
}}

EXAMPLE OUTPUT (employment contract):
{{
  "counterparty_leverage": "CUSTOMER",
  "termination_risk": 2,
  "data_sensitivity": 4,
  "ip_ownership_risk": "CUSTOMER",
  "payment_risk": 1,
  "auto_renewal_risk": 1,
  "confidentiality_scope": "UNILATERAL_SUPPLIER",
  "liability_cap_adequacy": 1
}}

EXAMPLE OUTPUT (standalone NDA, no payment terms):
{{
  "counterparty_leverage": "BALANCED",
  "termination_risk": 2,
  "data_sensitivity": 3,
  "ip_ownership_risk": "UNCLEAR",
  "payment_risk": 3,
  "auto_renewal_risk": 1,
  "confidentiality_scope": "MUTUAL",
  "liability_cap_adequacy": 3
}}
"""


def call_openrouter_risk_vector(
    model: str,
    risk_vector_system_prompt: str,
    user_prompt: str,
    timeout: int = 60,
) -> dict:
    """
    Send a risk-vector extraction request to a single AI model and return a
    validated result dictionary.

    Structurally mirrors call_openrouter_sector() but validates against
    RiskVectorExtraction.  Keeping the three call functions separate ensures
    a failure in any one never suppresses the results of the other two.

    max_tokens is set to 200 — sufficient for all eight fields with their
    values, with headroom for any extra whitespace the model may emit.

    Args
    ----
    model                    : OpenRouter model ID.
    risk_vector_system_prompt: Instructions built by build_risk_vector_prompt().
    user_prompt              : The same contract text wrapped by build_user_prompt(),
                               reused from the earlier two calls.
    timeout                  : Maximum seconds to wait (default: 60).

    Returns
    -------
    A dict with keys:
      risk_vector_counterparty_leverage
      risk_vector_termination_risk
      risk_vector_data_sensitivity
      risk_vector_ip_ownership_risk
      risk_vector_payment_risk
      risk_vector_auto_renewal_risk
      risk_vector_confidentiality_scope
      risk_vector_liability_cap_adequacy
      risk_vector_raw_response
      elapsed_risk_vector_s
      risk_vector_error

    All risk_vector_* keys are prefixed to avoid collisions with the flat
    output dict merged in classify_contract().
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "Referer": "https://github.com/contract-classifier",
        "X-Title": "Contract Classifier",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": risk_vector_system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 200,
        "response_format": {"type": "json_object"},
    }

    # Sentinel values used when the call fails — mirrors the pattern used by
    # call_openrouter_sector for sector failures.
    _null_risk_vector = {
        "risk_vector_counterparty_leverage": None,
        "risk_vector_termination_risk": None,
        "risk_vector_data_sensitivity": None,
        "risk_vector_ip_ownership_risk": None,
        "risk_vector_payment_risk": None,
        "risk_vector_auto_renewal_risk": None,
        "risk_vector_confidentiality_scope": None,
        "risk_vector_liability_cap_adequacy": None,
        "risk_vector_raw_response": None,
    }

    t0 = time.perf_counter()
    raw_content = None  # ensure it exists in exception scope
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(OPENROUTER_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        content = data["choices"][0]["message"]["content"]
        if content is None:
            raise ValueError(
                "Model returned null content (possible content filter or empty response)"
            )
        raw_content = _extract_json(content)

        parsed_json = json.loads(raw_content)
        result = RiskVectorExtraction(**parsed_json)

        return {
            "risk_vector_counterparty_leverage": result.counterparty_leverage,
            "risk_vector_termination_risk": result.termination_risk,
            "risk_vector_data_sensitivity": result.data_sensitivity,
            "risk_vector_ip_ownership_risk": result.ip_ownership_risk,
            "risk_vector_payment_risk": result.payment_risk,
            "risk_vector_auto_renewal_risk": result.auto_renewal_risk,
            "risk_vector_confidentiality_scope": result.confidentiality_scope,
            "risk_vector_liability_cap_adequacy": result.liability_cap_adequacy,
            "risk_vector_raw_response": raw_content,
            "elapsed_risk_vector_s": round(time.perf_counter() - t0, 2),
            "risk_vector_error": None,
        }

    except httpx.HTTPStatusError as e:
        return {
            **_null_risk_vector,
            "elapsed_risk_vector_s": round(time.perf_counter() - t0, 2),
            "risk_vector_error": f"HTTP {e.response.status_code}: {e.response.text}",
        }
    except json.JSONDecodeError as e:
        return {
            **_null_risk_vector,
            "risk_vector_raw_response": raw_content,
            "elapsed_risk_vector_s": round(time.perf_counter() - t0, 2),
            "risk_vector_error": f"JSON parse error: {e}",
        }
    except Exception as e:
        return {
            **_null_risk_vector,
            "elapsed_risk_vector_s": round(time.perf_counter() - t0, 2),
            "risk_vector_error": str(e),
        }


def _extract_json(text: str) -> str:
    """
    Extract the first top-level JSON object from a model response.

    Some models emit reasoning or thinking tokens (e.g. <think>...</think>)
    before or after the JSON payload, or include trailing commentary. This
    function strips those artefacts and returns only the JSON object string,
    ready for json.loads().
    """
    # Remove <think>...</think> blocks (and similar reasoning wrappers)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Strip markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()
    # Find the first '{' and the matching closing '}'
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model response")
    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    raise ValueError("Unbalanced braces in model response — JSON object not closed")


def call_openrouter_sector(
    model: str,
    sector_system_prompt: str,
    user_prompt: str,
    timeout: int = 60,
) -> dict:
    """
    Send a regulated-sector classification request to a single AI model and
    return a validated result dictionary.

    Structurally mirrors call_openrouter() but uses the sector system prompt
    and validates against SectorClassification. Keeping the two functions
    separate ensures a sector failure never suppresses the type result.

    max_tokens is 150 (vs 300 for the type call) because the sector response
    contains only one field; even all 16 sector labels fit within this limit.

    Args
    ────
    model               : OpenRouter model ID.
    sector_system_prompt: The sector instructions built by build_sector_prompt().
    user_prompt         : The same contract text wrapped by build_user_prompt(),
                          reused from the type classification call.
    timeout             : Maximum seconds to wait for the API response (default: 60).

    Returns
    ───────
    A dict with keys: regulated_sectors, sector_raw_response, sector_error.
    """
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "Referer": "https://github.com/contract-classifier",
        "X-Title": "Contract Classifier",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": sector_system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 150,
        "response_format": {"type": "json_object"},
    }

    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(OPENROUTER_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        content = data["choices"][0]["message"]["content"]
        if content is None:
            raise ValueError(
                "Model returned null content (possible content filter or empty response)"
            )
        raw_content = _extract_json(content)

        parsed_json = json.loads(raw_content)
        result = SectorClassification(**parsed_json)

        return {
            "regulated_sectors": result.regulated_sectors,
            "sector_raw_response": raw_content,
            "elapsed_sector_s": round(time.perf_counter() - t0, 2),
            "sector_error": None,
        }

    except httpx.HTTPStatusError as e:
        return {
            "regulated_sectors": [],
            "sector_raw_response": None,
            "elapsed_sector_s": round(time.perf_counter() - t0, 2),
            "sector_error": f"HTTP {e.response.status_code}: {e.response.text}",
        }
    except json.JSONDecodeError as e:
        return {
            "regulated_sectors": [],
            "sector_raw_response": raw_content if "raw_content" in locals() else None,
            "elapsed_sector_s": round(time.perf_counter() - t0, 2),
            "sector_error": f"JSON parse error: {e}",
        }
    except Exception as e:
        return {
            "regulated_sectors": [],
            "sector_raw_response": None,
            "elapsed_sector_s": round(time.perf_counter() - t0, 2),
            "sector_error": str(e),
        }


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

    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(OPENROUTER_API_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        content = data["choices"][0]["message"]["content"]
        if content is None:
            raise ValueError(
                "Model returned null content (possible content filter or empty response)"
            )
        raw_content = _extract_json(content)

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
            "elapsed_type_s": round(time.perf_counter() - t0, 2),
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
            "elapsed_type_s": round(time.perf_counter() - t0, 2),
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
            "elapsed_type_s": round(time.perf_counter() - t0, 2),
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
            "elapsed_type_s": round(time.perf_counter() - t0, 2),
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

    Orchestrates the full chained pipeline for each model:
      1. Load and clean the contract text (once, shared across all three calls).
      2. Run the contract type classifier  (call 1).
      3. Run the regulated sector classifier on the same text (call 2).
      4. Run the risk-vector extractor on the same text (call 3).
      5. Merge all three results into one flat record.

    A failure in any call is recorded in its own error key and does not
    suppress the other calls' results.

    Args
    ────
    contract_file        : Path to the contract text file (.txt or .md from OCS).
    models               : List of OpenRouter model IDs to query. Defaults to MODELS.
    delay_between_calls  : Seconds to pause between successive API calls. Applied
                           between each of the three calls per model and between
                           models. Increase if you encounter rate-limit errors
                           (default: 1.0).

    Returns
    ───────
    A list of merged result dictionaries, one per model queried. Each dict contains
    all type fields, all sector fields, all risk-vector fields, and all three error
    keys. Failed calls appear with null/[] fields rather than being silently dropped.
    """
    contract_text = load_contract_text(contract_file)
    selected_models = models or MODELS

    print(f"\U0001f4c4 Contract file   : {contract_file}")
    print(f"\U0001f916 Models to query : {len(selected_models)}")
    print("-" * 60)

    type_system_prompt = build_system_prompt()
    sector_system_prompt = build_sector_prompt()
    risk_vector_system_prompt = build_risk_vector_prompt()
    user_prompt = build_user_prompt(contract_text)

    # Derive contract_id from filename (replace dashes with underscores, remove extension)
    contract_id = Path(contract_file).stem.replace("-", "_")

    results = []
    for i, model in enumerate(selected_models, 1):
        # ── Call 1: contract type classification ────────────────────────────
        print(
            f"[{i}/{len(selected_models)}] {model} — type classifier ...",
            end=" ",
            flush=True,
        )
        type_result = call_openrouter(model, type_system_prompt, user_prompt)

        if type_result["error"]:
            print(
                f"❌ TYPE ERROR: {type_result['error']} ({type_result.get('elapsed_type_s', '?')}s)"
            )
        else:
            print(
                f"✅ {type_result.get('contract_type_primary')} | "
                f"{type_result.get('contract_type_secondary')} | "
                f"{type_result.get('subject_matter')} | "
                f"{type_result.get('governing_law')} | "
                f"{type_result.get('jurisdiction_city')}|"
                f"{type_result.get('jurisdiction_country')}|"
                f"{type_result.get('jurisdiction_court_type')} | "
                f"{type_result.get('contract_language')} "
                f"({type_result.get('elapsed_type_s', '?')}s)"
            )

        # Respect rate limits between calls for this model
        time.sleep(delay_between_calls)

        # ── Call 2: regulated sector classification ─────────────────────────
        print(
            f"[{i}/{len(selected_models)}] {model} — sector classifier ...",
            end=" ",
            flush=True,
        )
        sector_result = call_openrouter_sector(model, sector_system_prompt, user_prompt)

        if sector_result["sector_error"]:
            print(
                f"❌ SECTOR ERROR: {sector_result['sector_error']} ({sector_result.get('elapsed_sector_s', '?')}s)"
            )
        else:
            print(
                f"✅ {sector_result.get('regulated_sectors')} ({sector_result.get('elapsed_sector_s', '?')}s)"
            )

        time.sleep(delay_between_calls)

        # ── Call 3: risk vector extraction ──────────────────────────────────
        print(
            f"[{i}/{len(selected_models)}] {model} — risk vector ...",
            end=" ",
            flush=True,
        )
        risk_result = call_openrouter_risk_vector(
            model, risk_vector_system_prompt, user_prompt
        )

        if risk_result["risk_vector_error"]:
            print(
                f"❌ RISK VECTOR ERROR: {risk_result['risk_vector_error']} "
                f"({risk_result.get('elapsed_risk_vector_s', '?')}s)"
            )
        else:
            print(
                f"✅ leverage={risk_result.get('risk_vector_counterparty_leverage')} | "
                f"term={risk_result.get('risk_vector_termination_risk')} | "
                f"data={risk_result.get('risk_vector_data_sensitivity')} | "
                f"ip={risk_result.get('risk_vector_ip_ownership_risk')} | "
                f"pay={risk_result.get('risk_vector_payment_risk')} | "
                f"renew={risk_result.get('risk_vector_auto_renewal_risk')} | "
                f"conf={risk_result.get('risk_vector_confidentiality_scope')} | "
                f"cap={risk_result.get('risk_vector_liability_cap_adequacy')} "
                f"({risk_result.get('elapsed_risk_vector_s', '?')}s)"
            )

        # ── Merge all three results into one flat record ─────────────────────
        merged = {
            **type_result,
            "regulated_sectors": sector_result["regulated_sectors"],
            "sector_raw_response": sector_result["sector_raw_response"],
            "elapsed_sector_s": sector_result["elapsed_sector_s"],
            "sector_error": sector_result["sector_error"],
            # Risk vector fields — prefixed to avoid collisions
            "risk_vector_counterparty_leverage": risk_result["risk_vector_counterparty_leverage"],
            "risk_vector_termination_risk": risk_result["risk_vector_termination_risk"],
            "risk_vector_data_sensitivity": risk_result["risk_vector_data_sensitivity"],
            "risk_vector_ip_ownership_risk": risk_result["risk_vector_ip_ownership_risk"],
            "risk_vector_payment_risk": risk_result["risk_vector_payment_risk"],
            "risk_vector_auto_renewal_risk": risk_result["risk_vector_auto_renewal_risk"],
            "risk_vector_confidentiality_scope": risk_result["risk_vector_confidentiality_scope"],
            "risk_vector_liability_cap_adequacy": risk_result["risk_vector_liability_cap_adequacy"],
            "risk_vector_raw_response": risk_result["risk_vector_raw_response"],
            "elapsed_risk_vector_s": risk_result["elapsed_risk_vector_s"],
            "risk_vector_error": risk_result["risk_vector_error"],
            # Total elapsed is the sum of all three calls
            "elapsed_total_s": round(
                (type_result.get("elapsed_type_s") or 0.0)
                + (sector_result.get("elapsed_sector_s") or 0.0)
                + (risk_result.get("elapsed_risk_vector_s") or 0.0),
                2,
            ),
            "contract_id": contract_id,
        }
        results.append(merged)

        # Delay before moving to next model (skip after the last)
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
        default=None,
        help=(
            "Path to save JSON results. "
            "Defaults to 'results-<contract stem>.json' for single file, or 'results-batch.json' for batch mode."
        ),
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay in seconds between API calls (default: 1.0)",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--contract",
        help="Path to the contract text file to classify.",
    )
    group.add_argument(
        "--batch_dir",
        help="Directory containing .md or .txt files to classify in batch mode.",
    )
    args = parser.parse_args()

    labelled_dir = Path(__file__).parent / "labelled_contracts"
    labelled_dir.mkdir(exist_ok=True)

    if args.batch_dir:
        batch_dir = Path(args.batch_dir)
        files = sorted(
            [
                f
                for f in batch_dir.iterdir()
                if f.suffix in {".md", ".txt"} and f.is_file()
            ]
        )
        print(f"Batch mode: {len(files)} files found in {batch_dir}")
        all_results = []
        total_files = len(files)
        for i, file in enumerate(files, 1):
            print(f"\nProcessing contract {i}/{total_files}: {file.name}")
            results = classify_contract(
                contract_file=str(file),
                delay_between_calls=args.delay,
            )
            all_results.extend(results)
        print(f"\nBatch processing complete: {total_files} contracts processed.")
        # Determine model name for output file
        model_name = None
        if all_results:
            # Use the first result's model field, get the part after '/'
            model_field = all_results[0].get("model", "model")
            model_name = model_field.split("/")[-1]
        else:
            model_name = "model"
        output_filename = f"results-{total_files}-{model_name}.json"
        output_path = (
            Path(args.output) if args.output else labelled_dir / output_filename
        )
        output_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print(f"\n💾 Batch results saved to: {output_path}")
    else:
        contract_path = Path(args.contract)
        results = classify_contract(
            contract_file=args.contract,
            delay_between_calls=args.delay,
        )
        # Determine model name for output file
        model_name = None
        if results:
            model_field = results[0].get("model", "model")
            model_name = model_field.split("/")[-1]
        else:
            model_name = "model"
        output_filename = f"results-{len(results)}-{model_name}.json"
        output_path = (
            Path(args.output) if args.output else labelled_dir / output_filename
        )
        output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\n💾 Full results saved to: {output_path}")
